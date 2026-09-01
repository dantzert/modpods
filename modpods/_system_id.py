"""Lightweight system identification model.

This module provides SystemIdModel, which implements the core operations
used by modpods:
  - Polynomial feature expansion
  - Finite-difference time differentiation
  - Ordinary least squares
  - Constrained least squares (equality via closed-form Lagrange multipliers,
    inequality via an active-set QP solver)
  - ODE simulation via scipy.integrate.solve_ivp

This lightweight implementation avoids external dependencies and yields
significant speedups on the operations that matter (fit+score, simulate).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any

import numpy as np
import pandas as pd
import scipy.signal
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d

try:
    from numba import njit  # type: ignore[import-not-found]

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

_JIT_THRESHOLD = 16

_savgol_coeffs_cache: dict[tuple[int, int, float], np.ndarray] = {}


def _get_savgol_coeffs(width: int, order: int, dt: float) -> np.ndarray:
    """Return cached Savitzky-Golay first-derivative coefficients.

    The coefficients depend only on (window_length, polyorder, delta) —
    not the data — so caching avoids the expensive ``savgol_coeffs``
    call (which internally does polyfit/polyval/lstsq) on every invocation.
    """
    key = (width, order, dt)
    if key not in _savgol_coeffs_cache:
        _savgol_coeffs_cache[key] = scipy.signal.savgol_coeffs(
            window_length=width,
            polyorder=order,
            deriv=1,
            delta=dt,
        )
    return _savgol_coeffs_cache[key]


def _polynomial_feature_names(
    input_names: list[str],
    degree: int,
    include_bias: bool,
    include_interaction: bool,
) -> list[str]:
    """Generate polynomial feature names matching pysindy's PolynomialLibrary.

    Ordering:
      - If include_bias: ``["1"]`` is prepended.
      - For d in range(1, degree+1):
        - include_interaction=False: each *input* variable raised to power d.
        - include_interaction=True: all combinations_with_replacement
          of input indices with repetition d.
    """
    names: list[str] = []
    if include_bias:
        names.append("1")
    for d in range(1, degree + 1):
        if not include_interaction:
            for j in range(len(input_names)):
                if d == 1:
                    names.append(input_names[j])
                else:
                    names.append(f"{input_names[j]}^{d}")
        else:
            for combo in combinations_with_replacement(range(len(input_names)), d):
                parts: list[str] = []
                unique: dict[int, int] = {}
                for idx in combo:
                    unique[idx] = unique.get(idx, 0) + 1
                for idx, count in unique.items():
                    if count == 1:
                        parts.append(input_names[idx])
                    else:
                        parts.append(f"{input_names[idx]}^{count}")
                names.append(" ".join(parts))
    return names


def _n_polynomial_features(
    n_inputs: int,
    degree: int,
    include_bias: bool,
    include_interaction: bool,
) -> int:
    """Return the number of polynomial features (matches pysindy)."""
    if include_interaction:
        total = 0
        for d in range(0 if include_bias else 1, degree + 1):
            n = 1
            for i in range(d):
                n = n * (n_inputs + i) // (i + 1)
            total += n
    else:
        total = sum(n_inputs for _ in range(1, degree + 1))
        if include_bias:
            total += 1
    return total


if _HAS_NUMBA:

    @njit(cache=True)
    def _expand_poly_no_interaction_numba(
        data: np.ndarray, degree: int, include_bias: bool
    ) -> np.ndarray:
        n_samples, n_features = data.shape
        n_cols = n_features * degree
        total = n_cols + 1 if include_bias else n_cols
        result = np.empty((n_samples, total))
        col = 0
        if include_bias:
            for i in range(n_samples):
                result[i, 0] = 1.0
            col = 1
        for d in range(1, degree + 1):
            for j in range(n_features):
                for i in range(n_samples):
                    v = data[i, j]
                    result[i, col] = v
                    for _ in range(d - 1):
                        result[i, col] *= v
                col += 1
        return result


def _expand_polynomial(
    data: np.ndarray,
    degree: int,
    include_bias: bool,
    include_interaction: bool,
) -> np.ndarray:
    """Expand *data* into polynomial features (matches PolynomialLibrary).

    Uses numba JIT when available and the input is large enough to
    amortise the ~1 µs Python→numba dispatch overhead. For small inputs
    (e.g. the single-sample calls from ``simulate``'s per-step RHS),
    vectorised numpy is faster.

    Args:
        data: shape (n_samples, n_input_features)
        degree: maximum polynomial degree.
        include_bias: prepend a constant column.
        include_interaction: include cross-terms.

    Returns:
        shape (n_samples, n_output_features)
    """
    n_samples, n_features = data.shape

    if not include_interaction:
        if _HAS_NUMBA and n_samples > _JIT_THRESHOLD:
            result = _expand_poly_no_interaction_numba(data, degree, include_bias)
            return np.asarray(result)

        col_indices = np.tile(np.arange(n_features), degree)
        powers = np.repeat(np.arange(1, degree + 1), n_features)
        cols = data[:, col_indices] ** powers
        if include_bias:
            cols = np.hstack([np.ones((n_samples, 1)), cols])
        return np.asarray(cols)

    # include_interaction=True
    columns: list[np.ndarray] = []
    if include_bias:
        columns.append(np.ones((n_samples, 1)))
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            term = np.ones(n_samples)
            for idx in combo:
                term = term * data[:, idx]
            columns.append(term.reshape(-1, 1))
    if len(columns) == 0:
        return np.empty((n_samples, 0))
    return np.hstack(columns)


def _finite_difference(
    x: np.ndarray, t: np.ndarray, order: int, drop_endpoints: bool
) -> np.ndarray:
    """Compute time derivatives via finite differences.

    - order=2 (default): centered differences via numpy.gradient
      (edge_order=2 matches pysindy FiniteDifference exactly).
    - order=10: 11-point Savitzky-Golay filter
      (matches pysindy FiniteDifference(order=10) at interior points).

    If drop_endpoints is True, endpoint rows are set to NaN so they are
    dropped before least-squares fitting (matching pysindy's behaviour).
    """
    dt = float(np.asarray(np.diff(t))[0])

    if order == 2 and not drop_endpoints:
        return np.asarray(np.gradient(x, dt, axis=0, edge_order=2))

    width = 2 * (order // 2) + 1
    half = width // 2
    coeffs = _get_savgol_coeffs(width, order, dt)

    if x.shape[1] == 1:
        deriv = np.empty_like(x, dtype=float)
        deriv[:, 0] = convolve1d(x[:, 0], coeffs, mode="constant")
        if half > 0 and not drop_endpoints:
            p = np.polyfit(np.arange(width), x[:width, 0], order)
            deriv[:half, 0] = np.polyval(np.polyder(p, 1), np.arange(0, half)) / dt
            p = np.polyfit(np.arange(width), x[-width:, 0], order)
            deriv[-half:, 0] = (
                np.polyval(np.polyder(p, 1), np.arange(half + 1, width)) / dt
            )
        deriv = deriv.reshape(-1, 1)
    else:
        deriv = np.empty_like(x, dtype=float)
        for j in range(x.shape[1]):
            col = x[:, j]
            deriv[:, j] = convolve1d(col, coeffs, mode="constant")
            if half > 0 and not drop_endpoints:
                p = np.polyfit(np.arange(width), col[:width], order)
                deriv[:half, j] = np.polyval(np.polyder(p, 1), np.arange(0, half)) / dt
                p = np.polyfit(np.arange(width), col[-width:], order)
                deriv[-half:, j] = (
                    np.polyval(np.polyder(p, 1), np.arange(half + 1, width)) / dt
                )

    if drop_endpoints:
        deriv[:half] = np.nan
        deriv[-half:] = np.nan

    return np.asarray(deriv)


def _active_set_qp(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
    ridge_lambda: float = 1e-8,
) -> np.ndarray:
    """Solve  min ||A w - b||^2  s.t.  C w <= d  via the active-set method.

    Fast for the small problems encountered in modpods (a few dozen
    features at most).  Falls back gracefully when no QP solver is
    available — cvxpy is an explicit dependency already.
    """
    n = A.shape[1]
    # Use regularized least squares for better numerical stability
    AtA = A.T @ A + ridge_lambda * np.eye(n)
    Atb = A.T @ b
    w = np.linalg.solve(AtA, Atb)
    active: set[int] = set()

    for _ in range(max_iter):
        violation = C @ w - d
        violated = np.where(violation > tol)[0]
        if len(violated) == 0:
            break

        most_violated = int(np.argmax(violation[violated]))
        active.add(int(violated[most_violated]))

        C_active = C[list(active)]
        d_active = d[list(active)]

        # Equality-constrained least-squares via Lagrange multipliers
        AtA_reg = A.T @ A + ridge_lambda * np.eye(n)
        Atb_reg = A.T @ b
        w_ls = np.linalg.solve(AtA_reg, Atb_reg)
        A_inv = np.linalg.inv(AtA_reg)
        CAt = C_active @ A_inv
        denom = CAt @ C_active.T
        if denom.size == 1:
            denom_inv = 1.0 / denom
        else:
            denom_inv = np.linalg.inv(denom)
        mult = denom_inv @ (C_active @ w_ls - d_active)
        w = w_ls - A_inv @ C_active.T @ mult

        # Remove inactive constraints
        violation = C @ w - d
        to_remove = [i for i in active if violation[i] < -tol]
        for i in to_remove:
            active.remove(i)

    return np.asarray(w)


class SystemIdModel:
    """Lightweight ODE/transfer-function model.

    Supports polynomial features, finite-difference differentiation,
    ordinary least squares, and constrained least squares.
    """

    def __init__(
        self,
        poly_degree: int = 3,
        include_bias: bool = False,
        include_interaction: bool = False,
        fd_order: int = 2,
        fd_drop_endpoints: bool = False,
        constraint_lhs: np.ndarray | None = None,
        constraint_rhs: np.ndarray | None = None,
        inequality_constraints: bool = False,
        initial_guess: np.ndarray | None = None,
        relax_coeff_nu: float | None = None,
        max_iter: int | None = None,
    ) -> None:
        self.poly_degree = poly_degree
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.fd_order = fd_order
        self.fd_drop_endpoints = fd_drop_endpoints
        self.constraint_lhs = (
            np.array(constraint_lhs, dtype=float)
            if constraint_lhs is not None
            else None
        )
        self.constraint_rhs = (
            np.array(constraint_rhs, dtype=float)
            if constraint_rhs is not None
            else None
        )
        self.inequality_constraints = inequality_constraints
        self.initial_guess = (
            np.array(initial_guess, dtype=float) if initial_guess is not None else None
        )
        self.relax_coeff_nu = relax_coeff_nu
        self.max_iter = max_iter

        self._coef: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._poly_feature_names: list[str] | None = None
        self._n_input_features: int = 0
        self._n_output_features: int = 0
        self._n_targets: int = 0
        self._is_fitted: bool = False
        self._cached_x_hash: int | None = None
        self._cached_t_hash: int | None = None
        self._cached_x_dot: np.ndarray | None = None
        self._cached_theta: np.ndarray | None = None
        self._cached_valid: np.ndarray | None = None

    # -- public API ---------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Names of the input variables (x columns + u columns)."""
        return self._feature_names if self._feature_names is not None else []

    @feature_names.setter
    def feature_names(self, value: list[str]) -> None:
        self._feature_names = list(value)

    def get_feature_names(self) -> list[str]:
        """Names of the polynomial-library (output) features."""
        return self._poly_feature_names if self._poly_feature_names is not None else []

    @property
    def n_features_in_(self) -> int:
        return self._n_input_features

    @property
    def n_output_features_(self) -> int:
        return self._n_output_features

    def coefficients(self) -> np.ndarray:
        """Return the fitted coefficient matrix, shape (n_targets, n_library_features)."""
        if self._coef is None:
            raise RuntimeError("Model is not fitted yet.")
        return self._coef

    def fit(
        self,
        x: np.ndarray | pd.DataFrame | pd.Series,
        t: np.ndarray | float,
        u: np.ndarray | pd.DataFrame | pd.Series | None = None,
        x_dot: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> SystemIdModel:
        """Fit the model.

        Args:
            x: target time-series, shape (n,) or (n, n_targets).
            t: time points (n,) or scalar dt.
            u: optional control inputs, shape (n,) or (n, n_controls).
            x_dot: pre-computed derivative (if known).
            feature_names: names for x and u columns.

        Returns:
            self (for chaining).
        """
        x_arr = self._to_array(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        n_samples, n_targets = x_arr.shape

        t_arr = self._to_time_array(t, n_samples)

        if u is not None:
            u_arr = self._to_array(u)
            if u_arr.ndim == 1:
                u_arr = u_arr.reshape(-1, 1)
        else:
            u_arr = None

        # Feature names
        if feature_names is not None:
            self._feature_names = list(feature_names)
        elif self._feature_names is None:
            self._feature_names = [f"x{i}" for i in range(x_arr.shape[1])]
            if u_arr is not None:
                self._feature_names += [f"u{i}" for i in range(u_arr.shape[1])]

        # Input features for polynomial library = [x_columns, u_columns]
        if u_arr is not None:
            data = np.hstack([x_arr, u_arr])
            input_names = self._feature_names
        else:
            data = x_arr
            input_names = self._feature_names[: x_arr.shape[1]]

        self._n_input_features = data.shape[1]
        self._n_targets = n_targets

        # Polynomial feature names
        self._poly_feature_names = _polynomial_feature_names(
            input_names,
            self.poly_degree,
            self.include_bias,
            self.include_interaction,
        )
        self._n_output_features = len(self._poly_feature_names)

        # Derivative
        if x_dot is not None:
            x_dot_arr = self._to_array(x_dot)
            if x_dot_arr.ndim == 1:
                x_dot_arr = x_dot_arr.reshape(-1, 1)
        else:
            x_dot_arr = _finite_difference(
                x_arr, t_arr, self.fd_order, self.fd_drop_endpoints
            )

        # Polynomial expansion
        theta = _expand_polynomial(
            data, self.poly_degree, self.include_bias, self.include_interaction
        )

        # Drop NaN rows (from drop_endpoints=True)
        valid = ~np.isnan(x_dot_arr).any(axis=1) & ~np.isnan(theta).any(axis=1)
        theta_valid = theta[valid]
        x_dot_valid = x_dot_arr[valid]

        # Solve with regularization
        self._coef = self._solve(theta_valid, x_dot_valid)

        # Cache computed arrays for potential reuse in score()
        self._cached_x_hash = hash(x_arr.tobytes())
        self._cached_t_hash = hash(t_arr.tobytes())
        self._cached_x_dot = x_dot_arr
        self._cached_theta = theta
        self._cached_valid = valid

        self._is_fitted = True
        return self

    def _solve(self, theta: np.ndarray, x_dot: np.ndarray) -> np.ndarray:
        """Return coefficient matrix of shape (n_targets, n_features)."""
        if self.constraint_lhs is None or self.constraint_rhs is None:
            # Regularized OLS (ridge regression) for better numerical stability
            # This avoids SVD convergence issues with ill-conditioned matrices
            ridge_lambda = 1e-8
            AtA = theta.T @ theta + ridge_lambda * np.eye(theta.shape[1])
            Atb = theta.T @ x_dot
            coef = np.linalg.solve(AtA, Atb)
            return coef.T
        else:
            C = self.constraint_lhs
            d = self.constraint_rhs.flatten()

            if not self.inequality_constraints:
                return self._solve_equality_constrained(theta, x_dot, C, d)
            else:
                return self._solve_inequality_constrained(theta, x_dot, C, d)

    def _solve_equality_constrained(
        self, theta: np.ndarray, x_dot: np.ndarray, C: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """Solve min ||(I⊗Θ) w − vec(Xd)||²  s.t.  C w = d  via Lagrange.

        Returns coefficient matrix of shape (n_targets, n_feat).
        """
        n_feat = theta.shape[1]
        n_targets = x_dot.shape[1] if x_dot.ndim > 1 else 1
        x_dot_2d = x_dot.reshape(-1, 1) if x_dot.ndim == 1 else x_dot

        # Add regularization for numerical stability
        ridge_lambda = 1e-8
        AtA = theta.T @ theta + ridge_lambda * np.eye(n_feat)
        Atb = theta.T @ x_dot_2d  # (n_feat, n_targets)
        w_ls = np.linalg.solve(AtA, Atb)  # (n_feat, n_targets)
        A_inv = np.linalg.inv(AtA)

        # Target-major vectorisation: [target 0 coeffs, target 1 coeffs, ...]
        w_ls_vec = w_ls.T.flatten()

        # I ⊗ A_inv (block-diagonal, one block per target)
        kron_A_inv = np.kron(np.eye(n_targets), A_inv) if n_targets > 1 else A_inv
        C_A_inv = C @ kron_A_inv
        denom = C_A_inv @ C.T
        denom_inv = 1.0 / denom if denom.size == 1 else np.linalg.inv(denom)
        mult = denom_inv @ (C @ w_ls_vec - d)
        w = w_ls_vec - kron_A_inv @ C.T @ mult

        return np.asarray(w.reshape(n_targets, n_feat))

    def _solve_inequality_constrained(
        self, theta: np.ndarray, x_dot: np.ndarray, C: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """Solve  min ||(I⊗Theta) w - vec(X_dot)||^2  s.t.  C w <= d."""
        n_feat = theta.shape[1]
        n_targets = x_dot.shape[1] if x_dot.ndim > 1 else 1
        x_dot_2d = x_dot.reshape(-1, 1) if x_dot.ndim == 1 else x_dot

        if n_targets == 1:
            w = _active_set_qp(theta, x_dot_2d.flatten(), C, d)
            return np.asarray(w.reshape(1, n_feat))

        A = np.kron(np.eye(n_targets), theta)
        b = x_dot_2d.flatten(order="F")
        w = _active_set_qp(A, b, C, d)
        return np.asarray(w.reshape(n_targets, n_feat))

    def score(
        self,
        x: np.ndarray | pd.DataFrame | pd.Series,
        t: np.ndarray | float,
        u: np.ndarray | pd.DataFrame | pd.Series | None = None,
        **kwargs: Any,
    ) -> float:
        """R² score on the finite-difference derivative (variance_weighted)."""
        x_arr = self._to_array(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        t_arr = self._to_time_array(t, x_arr.shape[0])

        # Reuse cached derivative & theta if inputs match the last fit()
        x_hash = hash(x_arr.tobytes())
        t_hash = hash(t_arr.tobytes())
        if (
            self._cached_x_hash == x_hash
            and self._cached_t_hash == t_hash
            and self._cached_x_dot is not None
            and self._cached_theta is not None
            and self._cached_valid is not None
        ):
            x_dot = self._cached_x_dot
            theta = self._cached_theta
            valid = self._cached_valid
        else:
            if u is not None:
                u_arr = self._to_array(u)
                if u_arr.ndim == 1:
                    u_arr = u_arr.reshape(-1, 1)
                data = np.hstack([x_arr, u_arr])
            else:
                data = x_arr

            x_dot = _finite_difference(
                x_arr, t_arr, self.fd_order, self.fd_drop_endpoints
            )
            theta = _expand_polynomial(
                data, self.poly_degree, self.include_bias, self.include_interaction
            )
            valid = ~np.isnan(x_dot).any(axis=1) & ~np.isnan(theta).any(axis=1)

        x_dot_valid = x_dot[valid]
        theta_valid = theta[valid]

        x_dot_pred = theta_valid @ self._coef.T
        # Variance-weighted R² across targets
        ss_res = np.sum((x_dot_valid - x_dot_pred) ** 2, axis=0)
        ss_tot = np.sum((x_dot_valid - x_dot_valid.mean(axis=0)) ** 2, axis=0)
        var_weights = ss_tot / ss_tot.sum()
        return float(
            1.0 - np.sum(var_weights * ss_res / np.where(ss_tot > 0, ss_tot, 1))
        )

    def predict(
        self,
        x: np.ndarray | pd.DataFrame | pd.Series,
        u: np.ndarray | pd.DataFrame | pd.Series | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Evaluate the model RHS for the given state / control.

        Returns d/dt(x) with shape (n_samples, n_targets).
        """
        x_arr = self._to_array(x)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        if u is not None:
            u_arr = self._to_array(u)
            if u_arr.ndim == 1:
                u_arr = u_arr.reshape(-1, 1)
            data = np.hstack([x_arr, u_arr])
        else:
            data = x_arr

        theta = _expand_polynomial(
            data, self.poly_degree, self.include_bias, self.include_interaction
        )
        return np.asarray(theta @ self._coef.T)

    def simulate(
        self,
        x0: np.ndarray | float,
        t: np.ndarray,
        u: np.ndarray | pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Integrate the ODE forward in time.

        Args:
            x0: Initial condition, shape (n_targets,) or (n_targets, 1).
            t: Time points array.
            u: Control inputs, shape (n_samples,) or (n_samples, n_controls).

        Returns:
            Simulated trajectory, shape (n_samples - 1, n_targets).
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        t_arr = np.asarray(t, dtype=float).flatten()
        x0_flat = np.asarray(x0, dtype=float).flatten()
        if x0_flat.size == 1:
            x0_flat = x0_flat.reshape(1)

        coef_t = self._coef.T  # (n_feat, n_target) — pre-transposed
        poly_degree = self.poly_degree
        include_bias = self.include_bias
        include_interaction = self.include_interaction

        if u is not None:
            u_arr = self._to_array(u)
            if u_arr.ndim == 1:
                u_arr = u_arr.reshape(-1, 1)
            u_fun = interp1d(
                t_arr,
                u_arr,
                axis=0,
                kind="cubic",
                fill_value="extrapolate",
            )
        else:
            u_fun = None

        t_sim = t_arr[:-1]

        if not include_interaction:
            _degrees = np.arange(1, poly_degree + 1)

            if u_fun is not None:

                def _rhs(t_val: float, x_arr: np.ndarray) -> np.ndarray:
                    data = np.concatenate([x_arr.ravel(), u_fun(t_val).ravel()])
                    terms = (data[:, None] ** _degrees).T.ravel()
                    if include_bias:
                        return np.asarray(
                            (coef_t[0, :] + terms @ coef_t[1:, :]).ravel()
                        )
                    return np.asarray((terms @ coef_t).ravel())

            else:

                def _rhs(t_val: float, x_arr: np.ndarray) -> np.ndarray:
                    data = x_arr.ravel()
                    terms = (data[:, None] ** _degrees).T.ravel()
                    if include_bias:
                        return np.asarray(
                            (coef_t[0, :] + terms @ coef_t[1:, :]).ravel()
                        )
                    return np.asarray((terms @ coef_t).ravel())

        else:

            def _rhs(t_val: float, x_arr: np.ndarray) -> np.ndarray:
                if u_fun is not None:
                    u_t = u_fun(t_val).reshape(1, -1)
                    state = np.hstack([x_arr.reshape(1, -1), u_t])
                else:
                    state = x_arr.reshape(1, -1)
                theta = _expand_polynomial(
                    state, poly_degree, include_bias, include_interaction
                )
                return np.asarray((theta @ coef_t).flatten())

        sol = solve_ivp(
            _rhs,
            (t_sim[0], t_sim[-1]),
            x0_flat,
            t_eval=t_sim,
            method="LSODA",
            rtol=1e-12,
            atol=1e-12,
        )
        return np.asarray(sol.y.T)

    def print(self, precision: int = 3) -> None:
        """Print the model equations in a human-readable format."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        feature_names = self._poly_feature_names
        coef = self._coef  # (n_targets, n_feat)
        target_names = self._feature_names[: self._n_targets]

        for i, target in enumerate(target_names):
            terms: list[str] = []
            for j, name in enumerate(feature_names):
                c = coef[i, j]
                if abs(c) > 10 ** (-(precision + 1)):
                    terms.append(f"{c: .{precision}f} {name}")
            rhs = " + ".join(terms) if terms else f"{0:.{precision}f}"
            print(f"({target})' = {rhs}")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _to_array(
        val: np.ndarray | pd.DataFrame | pd.Series | float | None,
    ) -> np.ndarray:
        if val is None:
            return np.empty((0, 0))
        if isinstance(val, pd.DataFrame):
            return np.asarray(val.to_numpy(dtype=float))
        if isinstance(val, pd.Series):
            return np.asarray(val.to_numpy(dtype=float).reshape(-1, 1))
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _to_time_array(t: np.ndarray | float, n_samples: int) -> np.ndarray:
        if np.isscalar(t):
            return np.arange(n_samples, dtype=float) * float(np.asarray(t))
        return np.asarray(t, dtype=float).flatten()
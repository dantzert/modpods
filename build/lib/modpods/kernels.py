"""Convolution kernel definitions and registry for modpods.

Supports pluggable convolution kernels for delayed input transformation.
Each kernel defines a parametric impulse response h(t) that is convolved
with forcing inputs via FFT.  The default kernel is gamma (shape, scale, loc).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import scipy.stats as stats
from scipy.linalg import expm


class ConvolutionKernel(ABC):
    """Abstract base class for convolution kernels.

    Subclasses define a parametric impulse response h(t) that is convolved
    with forcing inputs.  The kernel is normalized such that sum(h(t)) = 1
    over the simulation time horizon.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this kernel type."""
        ...

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Number of free parameters for this kernel."""
        ...

    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """Human-readable names for the parameters, in order."""
        ...

    @property
    @abstractmethod
    def default_bounds(self) -> np.ndarray:
        """Array of [lower, upper] bounds for each parameter, shape (num_params, 2)."""
        ...

    @property
    @abstractmethod
    def default_init(self) -> np.ndarray:
        """Default initial parameter values, shape (num_params,)."""
        ...

    @abstractmethod
    def kernel_fn(self, t: np.ndarray, *params: float) -> np.ndarray:
        """Compute the kernel values at time points t.

        Args:
            t: Time array, shape (n,).
            *params: Kernel parameters in the order defined by param_names.

        Returns:
            Kernel values, shape (n,). Should integrate to ~1 over t.
        """
        ...

    @property
    def is_unstable(self) -> bool:
        """Whether this kernel represents an unstable impulse response."""
        return False

    def is_unstable_params(self, *params: float) -> bool:
        return self.is_unstable

    def is_stable_delay(self, *params: float) -> bool:
        return True

    def to_lti(self, *params: float) -> tuple:
        return None

    def make_kwargs(self, params: np.ndarray) -> dict:
        return dict(zip(self.param_names, params.tolist()))


class GammaKernel(ConvolutionKernel):
    """Gamma distribution kernel (default).

    h(t) = Gamma.pdf(t; shape, scale, loc)
    """

    @property
    def name(self) -> str:
        return "gamma"

    @property
    def num_params(self) -> int:
        return 3

    @property
    def param_names(self) -> List[str]:
        return ["shape", "scale", "loc"]

    @property
    def default_bounds(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 50.0],
                [0.1, 5.0],
                [0.0, 20.0],
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([1.0, 1.0, 0.0])

    def kernel_fn(  # type: ignore[override]
        self, t: np.ndarray, shape: float, scale: float, loc: float
    ) -> np.ndarray:
        return stats.gamma.pdf(t, shape, scale=scale, loc=loc)  # type: ignore[no-any-return]


class LogNormalKernel(ConvolutionKernel):
    """Log-normal distribution kernel.

    h(t) = Lognormal.pdf(t; mu, sigma)
    """

    @property
    def name(self) -> str:
        return "lognormal"

    @property
    def num_params(self) -> int:
        return 2

    @property
    def param_names(self) -> List[str]:
        return ["mu", "sigma"]

    @property
    def default_bounds(self) -> np.ndarray:
        return np.array(
            [
                [0.1, 5.0],
                [0.1, 5.0],
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([0.0, 1.0])

    def kernel_fn(self, t: np.ndarray, mu: float, sigma: float) -> np.ndarray:  # type: ignore[override]
        return stats.lognorm.pdf(t, sigma, scale=np.exp(mu))  # type: ignore[no-any-return]


class BimodalGammaKernel(ConvolutionKernel):
    """Sum of two gamma distribution kernels.

    h(t) = 0.5 * Gamma1.pdf(t) + 0.5 * Gamma2.pdf(t)
    """

    @property
    def name(self) -> str:
        return "bimodal_gamma"

    @property
    def num_params(self) -> int:
        return 6

    @property
    def param_names(self) -> List[str]:
        return ["shape1", "scale1", "loc1", "shape2", "scale2", "loc2"]

    @property
    def default_bounds(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 50.0],
                [0.1, 5.0],
                [0.0, 20.0],
                [1.0, 50.0],
                [0.1, 5.0],
                [0.0, 20.0],
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([2.0, 1.0, 0.0, 5.0, 1.0, 5.0])

    def kernel_fn(  # type: ignore[override]
        self,
        t: np.ndarray,
        shape1: float,
        scale1: float,
        loc1: float,
        shape2: float,
        scale2: float,
        loc2: float,
    ) -> np.ndarray:
        k1 = stats.gamma.pdf(t, shape1, scale=scale1, loc=loc1)
        k2 = stats.gamma.pdf(t, shape2, scale=scale2, loc=loc2)
        return 0.5 * (k1 + k2)  # type: ignore[no-any-return]


class UnderdampedOscillatorKernel(ConvolutionKernel):
    """Damped sinusoidal impulse response (underdamped LTI system).

    h(t) = (omega_n / sqrt(1 - zeta^2)) * exp(-zeta * omega_n * t) * sin(omega_d * t)
    where omega_d = omega_n * sqrt(1 - zeta^2)

    Parameters are physical: zeta (damping ratio) and omega_n (natural frequency).
    Positive zeta produces decaying oscillations; negative zeta produces growing
    (unstable) oscillations. The kernel is truncated to non-negative values for
    causality when zeta >= 0.

    Note: This does NOT construct LTI state-space matrices.  It only uses the
    impulse response for convolution.  Arbitrary pole placements may be an
    interesting extension but are out of scope for this PR.
    """

    @property
    def name(self) -> str:
        return "underdamped"

    @property
    def num_params(self) -> int:
        return 2

    @property
    def param_names(self) -> List[str]:
        return ["zeta", "omega_n"]

    @property
    def default_bounds(self) -> np.ndarray:
        return np.array(
            [
                [0.001, 5.0],
                [0.001, 50.0],
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([0.1, 2.0])

    def kernel_fn(self, t: np.ndarray, zeta: float, omega_n: float) -> np.ndarray:  # type: ignore[override]
        if zeta < 1.0:
            omega_d = omega_n * np.sqrt(1.0 - zeta**2)
            amplitude = omega_n / omega_d
            exponent = -zeta * omega_n * t
            max_exponent = 700.0
            exponent = np.clip(exponent, -max_exponent, max_exponent)
            h = amplitude * np.exp(exponent) * np.sin(omega_d * t)
        elif zeta == 1.0:
            h = omega_n**2 * t * np.exp(-omega_n * t)
        else:
            s = omega_n * np.sqrt(zeta**2 - 1.0)
            h = omega_n * np.exp(-zeta * omega_n * t) * np.sinh(s * t) / s
        if zeta < 0:
            return h  # type: ignore[no-any-return]
        return np.maximum(h, 0.0)  # type: ignore[no-any-return]

    @property
    def is_unstable(self) -> bool:
        return True

    def is_unstable_params(self, zeta: float, omega_n: float) -> bool:
        return zeta < 0

    def is_stable_delay(self, zeta: float, omega_n: float) -> bool:
        return zeta > 0

    def to_lti(self, zeta: float, omega_n: float) -> tuple:
        A = np.array(
            [
                [0.0, 1.0],
                [-(omega_n**2), -2.0 * zeta * omega_n],
            ]
        )
        B = np.array([[0.0], [1.0]])
        C = np.array([[omega_n, 0.0]])
        D = np.array([[0.0]])
        return A, B, C, D


class ExponentialGrowthKernel(ConvolutionKernel):
    """Exponential growth impulse response.

    h(t) = exp(rate * t) / sum(exp(rate * t))

    The kernel is normalized so that the values sum to 1 over the simulation
    time horizon.  rate > 0 produces monotonically increasing weights.

    Parameters:
        rate: Growth rate controlling how quickly the kernel increases with t.
    """

    @property
    def name(self) -> str:
        return "exponential_growth"

    @property
    def num_params(self) -> int:
        return 1

    @property
    def param_names(self) -> List[str]:
        return ["rate"]

    @property
    def default_bounds(self) -> np.ndarray:
        return np.array(
            [
                [0.01, 5.0],
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([0.5])

    def kernel_fn(self, t: np.ndarray, rate: float) -> np.ndarray:  # type: ignore[override]
        h = np.exp(rate * t)
        return h / np.sum(h)  # type: ignore[no-any-return]

    @property
    def is_unstable(self) -> bool:
        return True

    def is_unstable_params(self, rate: float) -> bool:
        return rate > 0

    def is_stable_delay(self, rate: float) -> bool:
        return rate < 0

    def to_lti(self, rate: float) -> tuple:
        A = np.array([[rate]])
        B = np.array([[1.0]])
        C = np.array([[rate]])
        D = np.array([[0.0]])
        return A, B, C, D


class ExponentialDecayKernel(ConvolutionKernel):
    """Exponential decay kernel (positive lambda = decay).

    h(t) = lambda * exp(-lambda * t)

    This is the standard exponential decay kernel, equivalent to a first-order
    low-pass filter. Useful for modeling simple delay dynamics.

    Note: The kernel is normalized such that integral = 1 (for lambda > 0).
    """

    @property
    def name(self) -> str:
        return "exponential_decay"

    @property
    def num_params(self) -> int:
        return 1

    @property
    def param_names(self) -> List[str]:
        return ["lambda"]

    @property
    def default_bounds(self) -> np.ndarray:
        return np.array(
            [
                [0.01, 20.0],  # lambda > 0 for decay
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([1.0])

    def kernel_fn(self, t: np.ndarray, lam: float) -> np.ndarray:  # type: ignore[override]
        return lam * np.exp(-lam * t)  # type: ignore[no-any-return]

    @property
    def is_unstable(self) -> bool:
        return False

    def is_stable_delay(self, lam: float) -> bool:
        return lam > 0

    def to_lti(self, lam: float) -> tuple:
        A = np.array([[-lam]])
        B = np.array([[1.0]])
        C = np.array([[lam]])
        D = np.array([[0.0]])
        return A, B, C, D


class ExponentialKernel(ConvolutionKernel):
    """Exponential growth/decay impulse response (unnormalized).

    h(t) = lambda * exp(lambda * t) for t >= 0

    This models pure exponential growth (lambda > 0) or decay (lambda < 0).
    Useful for capturing unstable poles in system identification.

    Note: The kernel is NOT normalized to integrate to 1, as exponential
    growth does not have a finite integral. The growth rate is captured
    by the lambda parameter directly.
    """

    @property
    def name(self) -> str:
        return "exponential"

    @property
    def num_params(self) -> int:
        return 1

    @property
    def param_names(self) -> List[str]:
        return ["lambda"]

    @property
    def default_bounds(self) -> np.ndarray:
        return np.array(
            [
                [-10.0, 10.0],  # lambda: negative for decay, positive for growth
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([1.0])

    def kernel_fn(self, t: np.ndarray, lam: float) -> np.ndarray:  # type: ignore[override]
        h = lam * np.exp(lam * t)
        return np.maximum(h, 0.0)  # type: ignore[no-any-return]

    @property
    def is_unstable(self) -> bool:
        return True

    def is_unstable_params(self, lam: float) -> bool:
        return lam > 0

    def is_stable_delay(self, lam: float) -> bool:
        return lam < 0

    def to_lti(self, lam: float) -> tuple:
        A = np.array([[lam]])
        B = np.array([[1.0]])
        C = np.array([[lam]])
        D = np.array([[0.0]])
        return A, B, C, D


class CanonicalLTIKernel(ConvolutionKernel):
    """Canonical-form intervening LTI system with fixed state dimension.

    This kernel represents an intervening LTI system in controllable canonical form:
    A = [[-a1, -a2, ..., -an],
         [ 1,   0, ...,  0 ],
         [ 0,   1, ...,  0 ],
         ...
         [ 0,   0, ...,  1, 0 ]]
    B = [[1], [0], ..., [0]]
    C = [[c1, c2, ..., cn]]
    D = [[d]]

    The state dimension n is fixed (default 5).
    The parameters are: [a1, ..., an, c1, ..., cn, d] (2n + 1 parameters for n states).

    This form can represent any LTI system with the given state dimension
    (controllable canonical form), including unstable eigenvalues.

    Parameters:
        n: State dimension (1 to max_states)
        a1...an: A matrix coefficients (last row of controllable canonical form)
        c1...cn: C matrix coefficients
        d: Direct feedthrough term
    """

    def __init__(self, max_states: int = 5):
        self.max_states = max_states

    @property
    def name(self) -> str:
        return "canonical_lti"

    @property
    def num_params(self) -> int:
        return 2 * self.max_states + 1

    @property
    def param_names(self) -> List[str]:
        names = []
        for i in range(1, self.max_states + 1):
            names.append(f"a{i}")
        for i in range(1, self.max_states + 1):
            names.append(f"c{i}")
        names.append("d")
        return names

    @property
    def default_bounds(self) -> np.ndarray:
        bounds = []
        for _ in range(self.max_states):
            bounds.append([-50.0, 50.0])
        for _ in range(self.max_states):
            bounds.append([-50.0, 50.0])
        bounds.append([-10.0, 10.0])
        return np.array(bounds)

    @property
    def default_init(self) -> np.ndarray:
        init = np.zeros(2 * self.max_states + 1)
        for i in range(self.max_states):
            init[i] = -0.5 * (0.5 ** i)
        init[self.max_states] = 1.0
        init[-1] = 0.0
        return init

    def kernel_fn(self, t: np.ndarray, *params: float) -> np.ndarray:
        n = self.max_states
        A, B, C, D = self._build_lti(params, self.max_states)

        # Check if A has eigenvalues outside unit circle (discrete-time stability)
        try:
            eigvals = np.linalg.eigvals(A)
            if np.any(np.abs(eigvals) > 1.0):
                return np.zeros_like(t)
        except:
            pass

        from scipy.linalg import expm
        n_states = A.shape[0]
        h = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 0.0
            else:
                try:
                    expAt = expm(A * ti)
                    B_vec = np.zeros((n, 1))
                    B_vec[-1, 0] = 1.0
                    h[i] = (C @ expAt @ B_vec).item()
                except (OverflowError, ValueError, RuntimeError):
                    h[i] = 0.0

        h_sum = np.sum(h)
        if h_sum != 0:
            h = h / h_sum
        return h

    @property
    def is_unstable(self) -> bool:
        return True

    def is_unstable_params(self, *params: float) -> bool:
        return True

    def is_stable_delay(self, *params: float) -> bool:
        return False

    def _build_lti(self, params: np.ndarray, n: int):
        a = params[:n]
        c = params[n:2*n]
        d = params[2*n]

        A = np.zeros((n, n))
        A[-1, :] = -np.array(a)
        for i in range(n - 1):
            A[i, i + 1] = 1.0

        B = np.zeros((n, 1))
        B[-1, 0] = 1.0

        C = np.array([params[n:2*n]])
        D = np.array([[params[2*n]]])

        return A, B, C, D

    def is_unstable_params(self, *params: float) -> bool:
        return True

    def is_stable_delay(self, *params: float) -> bool:
        return False

    def to_lti(self, *params: float) -> tuple:
        return self._build_lti(params, self.max_states)


class DirectLTISystem(ConvolutionKernel):
    """Direct LTI system in controllable canonical form.

    x' = A*x + B*u
    y  = C*x + D*u

    Canonical form:
        A = [[-a1, -a2, ..., -an],
             [ 1,   0, ...,  0 ],
             ...
             [ 0,   0, ...,  1, 0 ]]
        B = [[1], [0], ..., [0]]
        C = [[c1, c2, ..., cn]]
        D = [[d]]

    Parameters: [a1...an, c1...cn, d] (2n + 1 parameters)
    """

    def __init__(self, max_states: int = 5):
        self.max_states = max_states

    @property
    def name(self) -> str:
        return "direct_lti"

    @property
    def num_params(self) -> int:
        return 2 * self.max_states + 1

    @property
    def param_names(self) -> List[str]:
        names = [f"a{i+1}" for i in range(self.max_states)]
        names += [f"c{i+1}" for i in range(self.max_states)]
        names.append("d")
        return names

    @property
    def default_bounds(self) -> np.ndarray:
        bounds = []
        for _ in range(self.max_states):
            bounds.append([-50.0, 50.0])
        for _ in range(self.max_states):
            bounds.append([-50.0, 50.0])
        bounds.append([-10.0, 10.0])
        return np.array(bounds)

    @property
    def default_init(self) -> np.ndarray:
        init = np.zeros(2 * self.max_states + 1)
        for i in range(self.max_states):
            init[i] = -0.5 * (0.5 ** i)
        init[self.max_states] = 1.0
        init[-1] = 0.0
        return init

    def kernel_fn(self, t: np.ndarray, *params: float) -> np.ndarray:
        n = self.max_states
        A, B, C, D = self._build_lti(params, self.max_states)

        try:
            eigvals = np.linalg.eigvals(A)
            if np.any(np.abs(eigvals) > 1.0):
                return np.zeros_like(t)
        except:
            pass

        from scipy.linalg import expm
        n_states = A.shape[0]
        h = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 0.0
            else:
                try:
                    expAt = expm(A * ti)
                    B_vec = np.zeros((n, 1))
                    B_vec[-1, 0] = 1.0
                    h[i] = (C @ expAt @ B_vec).item()
                except (OverflowError, ValueError, RuntimeError):
                    h[i] = 0.0

        h_sum = np.sum(h)
        if h_sum != 0:
            h = h / h_sum
        return h

    @property
    def is_unstable(self) -> bool:
        return True

    def is_unstable_params(self, *params: float) -> bool:
        return True

    def is_stable_delay(self, *params: float) -> bool:
        return False

    def _build_lti(self, params: np.ndarray, n: int):
        a = params[:n]
        c = params[n:2*n]
        d = params[2*n]

        A = np.zeros((n, n))
        A[-1, :] = -np.array(a)
        for i in range(n - 1):
            A[i, i + 1] = 1.0

        B = np.zeros((n, 1))
        B[-1, 0] = 1.0

        C = np.array([params[n:2*n]])
        D = np.array([[params[2*n]]])

        return A, B, C, D

    def is_unstable_params(self, *params: float) -> bool:
        return True

    def is_stable_delay(self, *params: float) -> bool:
        return False

    def to_lti(self, *params: float) -> tuple:
        return self._build_lti(params, self.max_states)


class DecoupledLTISystem(ConvolutionKernel):
    """Decoupled LTI system in controllable canonical form.
    
    The system has the form:
        x_lti' = A_lti * x_lti + B_lti * u
        y_lti   = C_lti * x_lti + D_lti * u
    
    where:
        A_lti = [[-a1, -a2, ..., -an],
                 [ 1,   0, ...,  0 ],
                 ...
                 [ 0,   0, ...,  1, 0 ]]
        B_lti = [[1], [0], ..., [0]]
        C_lti = [[c1, c2, ..., cn]]
        D_lti = [[d]]
    
    Parameters: [a1...an, c1...cn, d] (2n + 1 parameters per output)
    """
    
    def __init__(self, n_states: int = 5, n_inputs: int = 1, n_outputs: int = 1):
        self.max_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
    @property
    def name(self) -> str:
        return "decoupled_lti"
    
    @property
    def num_params(self) -> int:
        return (2 * self.max_states + 1) * self.n_outputs * self.n_inputs
    
    @property
    def param_names(self) -> List[str]:
        names = []
        for out in range(self.n_outputs):
            for inp in range(self.n_inputs):
                for i in range(self.max_states):
                    names.append(f"a_{inp}_{out}_{i+1}")
                for i in range(self.max_states):
                    names.append(f"c_{inp}_{out}_{i+1}")
                names.append(f"d_{inp}_{out}")
        return names
    
    @property
    def default_bounds(self) -> np.ndarray:
        bounds = []
        for _ in range(self.n_outputs * self.n_inputs):
            for _ in range(self.max_states):
                bounds.append([-50.0, 50.0])  # a coefficients
            for _ in range(self.max_states):
                bounds.append([-50.0, 50.0])  # c coefficients
            bounds.append([-10.0, 10.0])  # d
        return np.array(bounds)
    
    @property
    def default_init(self) -> np.ndarray:
        init = np.zeros(self.num_params)
        n = self.max_states
        for out in range(self.n_outputs):
            for inp in range(self.n_inputs):
                base = (out * self.n_inputs + inp) * (2 * self.max_states + 1)
                for i in range(self.max_states):
                    init[base + i] = -0.5 * (0.5 ** i)  # decaying coefficients
                init[base + self.max_states] = 1.0  # c1 = 1
                init[base + 2 * self.max_states] = 0.0  # d = 0
        return init
    
    def kernel_fn(self, t: np.ndarray, *params: float) -> np.ndarray:
        n = self.max_states * self.n_outputs * self.n_inputs
        # Use single output for impulse response computation
        A, B, C, D = self._build_single_lti(params[:2*self.max_states+1])
        
        # Check if A has eigenvalues outside unit circle (discrete-time stability)
        try:
            eigvals = np.linalg.eigvals(A)
            if np.any(np.abs(eigvals) > 1.0):
                return np.zeros_like(t)
        except:
            pass
        
        # Compute impulse response
        from scipy.linalg import expm
        h = np.zeros_like(t)
        
        for i, ti in enumerate(t):
            if ti == 0:
                h[i] = 0.0
            else:
                try:
                    expAt = expm(A * ti)
                    B_vec = np.zeros((n, 1))
                    B_vec[-1, 0] = 1.0
                    h[i] = (C @ expAt @ B_vec).item()
                except (OverflowError, ValueError, RuntimeError):
                    h[i] = 0.0
        
        h_sum = np.sum(h)
        if h_sum != 0:
            h = h / h_sum
        return h
    
    @property
    def is_unstable(self) -> bool:
        return True
    
    def is_unstable_params(self, *params: float) -> bool:
        return True
    
    def is_stable_delay(self, *params: float) -> bool:
        return False
    
    def _build_single_lti(self, params: np.ndarray) -> tuple:
        """Build LTI for a single input-output pair."""
        n = self.max_states
        a = params[:n]
        c = params[n:2*n]
        d = params[2*n]
        
        A = np.zeros((n, n))
        A[-1, :] = -np.array(a)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        
        B = np.zeros((n, 1))
        B[-1, 0] = 1.0
        
        C = np.array([params[n:2*n]])
        D = np.array([[params[2*n]]])
        
        return A, B, C, D
    
    def _build_lti(self, params: np.ndarray, n: int):
        """Build LTI matrices from parameters."""
        # For backward compatibility, use the first input-output pair
        return self._build_single_lti(params[:2*self.max_states+1])
    
    def to_lti(self, *params: float) -> tuple:
        return self._build_single_lti(params[:2*self.max_states+1])


_KERNEL_REGISTRY: Dict[str, type] = {}


def register_kernel(kernel_cls: type) -> type:
    """Register a ConvolutionKernel subclass in the global registry.

    Can be used as a class decorator.
    """
    instance = kernel_cls()
    _KERNEL_REGISTRY[instance.name] = kernel_cls
    return kernel_cls


def get_kernel(name_or_instance) -> ConvolutionKernel:
    """Resolve a kernel by name string or return an instance directly.

    Args:
        name_or_instance: Kernel name string, or a ConvolutionKernel instance.

    Returns:
        A fresh ConvolutionKernel instance.
    """
    if isinstance(name_or_instance, ConvolutionKernel):
        return name_or_instance
    cls = _KERNEL_REGISTRY.get(str(name_or_instance))
    if cls is None:
        raise ValueError(
            f"Unknown kernel '{name_or_instance}'. " f"Available: {list_kernels()}"
        )
    return cls()  # type: ignore[no-any-return]


def list_kernels() -> List[str]:
    """Return names of all registered kernels."""
    return list(_KERNEL_REGISTRY.keys())


register_kernel(GammaKernel)
register_kernel(LogNormalKernel)
register_kernel(BimodalGammaKernel)
register_kernel(UnderdampedOscillatorKernel)
register_kernel(ExponentialGrowthKernel)
register_kernel(ExponentialDecayKernel)
register_kernel(ExponentialKernel)
register_kernel(CanonicalLTIKernel)
register_kernel(DirectLTISystem)
register_kernel(DecoupledLTISystem)
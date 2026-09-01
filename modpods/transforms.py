from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from scipy.optimize import minimize

from .kernels import ConvolutionKernel


# Bayesian optimization helper functions
def _expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """Expected Improvement acquisition function for Bayesian optimization."""
    mu, sigma = gpr.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide="warn"):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def _propose_location(
    acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=10, rng=None
):
    """Propose next sampling point by optimizing acquisition function."""
    dim = X_sample.shape[1]
    min_val = float("inf")
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()

    if rng is not None:
        x0s = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim))
    else:
        x0s = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim))
    for x0 in x0s:
        res = minimize(min_obj, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(-1, 1)


def _safe_convolve(forcing_values, kernel_values, mode="full"):
    """Safely compute convolution with fallback to time-domain method.

    FFT-based convolution (signal.fftconvolve) can overflow for growing
    oscillations (e.g., underdamped kernel with zeta < 0). This function
    tries FFT first, then falls back to time-domain convolution using
    signal.oaconvolve which handles growing signals more robustly.
    """
    try:
        result = signal.fftconvolve(forcing_values, kernel_values, mode=mode)
        if not np.all(np.isfinite(result)):
            raise ValueError("FFT convolution produced non-finite values")
        return result
    except (ValueError, FloatingPointError, OverflowError):
        result = signal.oaconvolve(forcing_values, kernel_values, mode=mode)
        if not np.all(np.isfinite(result)):
            raise ValueError("Time-domain convolution also produced non-finite values")
        return result


# =============================================================================
# Transform Cache - memoizes single-input kernel transforms to avoid recomputation
# =============================================================================


class TransformCache:
    """LRU cache for kernel-transformed time series.

    Caches results of convolving a forcing series with a kernel impulse response.
    Keys are quantized (input_name, n, kernel_name, params...) tuples so
    near-identical parameter sets reuse cached results.
    """

    def __init__(self, max_entries: int = 2000, quantization: float = 1e-6):
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.max_entries = max_entries
        self.quantization = quantization
        self.hits = 0
        self.misses = 0

    def _quantize(self, value: float) -> float:
        """Quantize a float to reduce near-duplicate keys."""
        if self.quantization <= 0:
            return value
        return round(value / self.quantization) * self.quantization

    def _make_key(
        self,
        input_name: str,
        n: int,
        kernel_name: str,
        params: tuple,
    ) -> tuple:
        """Create a hashable cache key from input name, kernel, and params."""
        return (
            input_name,
            n,
            kernel_name,
        ) + tuple(self._quantize(p) for p in params)

    def get(
        self,
        input_name: str,
        forcing_values: np.ndarray,
        kernel: ConvolutionKernel,
        params: tuple,
    ) -> np.ndarray:
        """Get cached transform or compute and cache it.

        Returns a COPY of the cached array to prevent mutation issues.
        """
        n = len(forcing_values)
        key = self._make_key(input_name, n, kernel.name, params)

        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key].copy()

        self.misses += 1
        shape_time = np.arange(0, n, 1)
        kernel_values = kernel.kernel_fn(shape_time, *params)
        result = _safe_convolve(forcing_values, kernel_values, mode="full")[:n]

        self._cache[key] = result

        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        return result.copy()  # type: ignore[no-any-return]

    def clear(self):
        """Clear the cache and reset counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_entries": self.max_entries,
        }

    def __repr__(self):
        s = self.stats()
        return f"TransformCache(hits={s['hits']}, misses={s['misses']}, hit_rate={s['hit_rate']:.2%}, size={s['size']})"


# Global cache instance used throughout the module
_transform_cache = TransformCache(max_entries=2000, quantization=1e-6)


def make_kernel_params(
    kernel: ConvolutionKernel,
    columns: list,
    init_transforms: int = 1,
    max_transforms: int = 4,
) -> pd.DataFrame:
    """Create a kernel_params DataFrame with MultiIndex rows.

    The DataFrame has a MultiIndex on rows of (transform_idx, param_name)
    and input variable names as columns.  This generalizes the previous
    separate shape_factors / scale_factors / loc_factors DataFrames.

    Args:
        kernel: ConvolutionKernel instance defining the parameter schema.
        columns: List of input variable names (DataFrame columns).
        init_transforms: Starting transform index (usually 1).
        max_transforms: Ending transform index (inclusive).

    Returns:
        DataFrame with MultiIndex rows and input columns, initialized to
        kernel.default_init values.
    """
    transform_idx = list(range(init_transforms, max_transforms + 1))
    param_idx = [(t, p) for t in transform_idx for p in kernel.param_names]
    index = pd.MultiIndex.from_tuples(param_idx, names=["transform", "param"])
    kernel_params = pd.DataFrame(index=index, columns=columns, dtype=float)

    for t in transform_idx:
        for col in columns:
            for i, p_name in enumerate(kernel.param_names):
                kernel_params.loc[(t, p_name), col] = kernel.default_init[i]

    return kernel_params


def params_vector_to_dataframe(
    kernel: ConvolutionKernel,
    params_vector: np.ndarray,
    columns: list,
    init_transforms: int,
    max_transforms: int,
) -> pd.DataFrame:
    """Convert a flat parameter vector to a kernel_params DataFrame.

    Args:
        kernel: ConvolutionKernel instance.
        params_vector: Flat array of all parameters, ordered by
            (transform_idx * param_name * column).
        columns: List of input variable names.
        init_transforms: Starting transform index.
        max_transforms: Ending transform index (inclusive).

    Returns:
        DataFrame with MultiIndex rows (transform, param) and input columns.
    """
    transform_idx = list(range(init_transforms, max_transforms + 1))
    param_idx = [(t, p) for t in transform_idx for p in kernel.param_names]
    index = pd.MultiIndex.from_tuples(param_idx, names=["transform", "param"])
    kernel_params = pd.DataFrame(index=index, columns=columns, dtype=float)

    idx = 0
    for t in transform_idx:
        for col in columns:
            for p_name in kernel.param_names:
                kernel_params.loc[(t, p_name), col] = params_vector[idx]
                idx += 1

    return kernel_params


def transform_inputs(
    kernel: ConvolutionKernel,
    kernel_params: pd.DataFrame,
    index,
    forcing,
    *,
    cache=None,
):
    """Apply kernel convolution transformations to forcing inputs.

    Vectorized implementation using FFT-based convolution with time-domain
    fallback. Optional LRU cache avoids recomputation for near-identical
    parameters during optimization.

    Args:
        kernel: ConvolutionKernel instance defining the impulse response.
        kernel_params: DataFrame with MultiIndex rows (transform_idx, param_name)
            and input variable names as columns.
        index: Time index.
        forcing: DataFrame of forcing inputs.
        cache: Optional TransformCache instance for memoization (default None).
    """
    orig_forcing_columns = [col for col in forcing.columns if "_tr_" not in col]

    num_transforms = kernel_params.index.get_level_values("transform").nunique()

    n = len(index)

    for input_col in orig_forcing_columns:
        forcing_values = forcing[input_col].to_numpy(dtype=float)

        for transform_idx in range(1, num_transforms + 1):
            col_name = f"{input_col}_tr_{transform_idx}"

            params = tuple(
                float(kernel_params.loc[(transform_idx, p_name), input_col])
                for p_name in kernel.param_names
            )

            if cache is not None:
                result = cache.get(input_col, forcing_values, kernel, params)
            else:
                shape_time = np.arange(0, n, 1)
                kernel_values = kernel.kernel_fn(shape_time, *params)
                result = _safe_convolve(forcing_values, kernel_values, mode="full")[:n]

            # Replace NaN/Inf with large but finite values to avoid downstream NaN issues
            if not np.all(np.isfinite(result)):
                result = np.nan_to_num(result, nan=1e6, posinf=1e6, neginf=-1e6)

            forcing.loc[:, col_name] = result

    if forcing.isnull().values.any():
        raise ValueError("Transform inputs produced NaN values")
    return forcing
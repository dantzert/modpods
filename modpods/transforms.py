import warnings
from collections import OrderedDict

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy.optimize import minimize

# Suppress the specific AxesWarning from pysindy after import
warnings.filterwarnings(
    "ignore", message=".*axes labeled for array with.*", module="pysindy"
)


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


def _propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=10, rng=None):
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


# =============================================================================
# Transform Cache - memoizes single-input gamma transforms to avoid recomputation
# =============================================================================


class TransformCache:
    """LRU cache for gamma-transformed time series.

    Caches results of convolving a forcing series with a gamma PDF kernel.
    Keys are quantized (input_name, shape, scale, loc) tuples so near-identical
    parameter sets reuse cached results.
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
        self, input_name: str, n: int, shape: float, scale: float, loc: float
    ) -> tuple:
        """Create a hashable cache key from input name and gamma params."""
        return (
            input_name,
            n,
            self._quantize(shape),
            self._quantize(scale),
            self._quantize(loc),
        )

    def get(
        self,
        input_name: str,
        forcing_values: np.ndarray,
        shape: float,
        scale: float,
        loc: float,
    ) -> np.ndarray:
        """Get cached transform or compute and cache it.

        Returns a COPY of the cached array to prevent mutation issues.
        """
        n = len(forcing_values)
        key = self._make_key(input_name, n, shape, scale, loc)

        if key in self._cache:
            self.hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key].copy()

        # Cache miss - compute the transform using FFT convolution
        self.misses += 1
        shape_time = np.arange(0, n, 1)
        gamma_kernel = stats.gamma.pdf(shape_time, shape, scale=scale, loc=loc)
        result = signal.fftconvolve(forcing_values, gamma_kernel, mode="full")[:n]

        # Store in cache
        self._cache[key] = result

        # Evict oldest if over capacity
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


def transform_inputs(
    shape_factors, scale_factors, loc_factors, index, forcing, *, cache=None
):
    """Vectorized implementation of transform_inputs for greater speed.

    Applies gamma PDF transformations to forcing inputs using FFT-based convolution
    instead of element-wise iteration. Optional LRU cache avoids recomputation for
    near-identical parameters during optimization.

    Args:
        shape_factors: DataFrame of gamma shape parameters
        scale_factors: DataFrame of gamma scale parameters
        loc_factors: DataFrame of gamma location parameters
        index: Time index
        forcing: DataFrame of forcing inputs
        cache: Optional TransformCache instance for memoization (default None)
    """
    # original forcing columns -> columns of forcing that don't have _tr_ in their name
    orig_forcing_columns = [col for col in forcing.columns if "_tr_" not in col]

    # how many rows of shape_factors do not contain NaNs?
    num_transforms = int(shape_factors.count().iloc[0])

    n = len(index)

    for input_col in orig_forcing_columns:
        forcing_values = forcing[input_col].to_numpy(dtype=float)

        for transform_idx in range(1, num_transforms + 1):
            col_name = f"{input_col}_tr_{transform_idx}"

            # Get gamma parameters
            shape = float(shape_factors[input_col][transform_idx])
            scale = float(scale_factors[input_col][transform_idx])
            loc = float(loc_factors[input_col][transform_idx])

            if cache is not None:
                # Use cached transform
                result = cache.get(input_col, forcing_values, shape, scale, loc)
            else:
                # Compute directly using FFT convolution (faster than np.convolve for large arrays)
                shape_time = np.arange(0, n, 1)
                gamma_kernel = stats.gamma.pdf(shape_time, shape, scale=scale, loc=loc)
                result = signal.fftconvolve(forcing_values, gamma_kernel, mode="full")[
                    :n
                ]

            forcing.loc[:, col_name] = result

    # assert there are no NaNs in the forcing
    if forcing.isnull().values.any():
        raise ValueError("Transform inputs produced NaN values")
    return forcing

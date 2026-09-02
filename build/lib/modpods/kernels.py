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
        """Whether this kernel represents an unstable impulse response.

        Unstable kernels have impulse responses that grow without bound,
        making convolution numerically problematic. They should be handled
        via explicit LTI simulation instead of convolution.
        """
        return False

    def is_unstable_params(self, *params: float) -> bool:
        """Check if the kernel is unstable for the given parameters.

        Args:
            *params: Kernel parameters in the order defined by param_names.

        Returns:
            True if the kernel is unstable for these parameters.
        """
        return self.is_unstable

    def is_stable_delay(self, *params: float) -> bool:
        """Check if the delay dynamics are stable for the given parameters.

        Delay dynamics should be stable to avoid spurious unstable modes.
        By default, kernels have stable delay dynamics.
        Override in subclasses for kernels that can have unstable delay dynamics.

        Args:
            *params: Kernel parameters in the order defined by param_names.

        Returns:
            True if the delay dynamics are stable for these parameters.
        """
        return True

    def to_lti(self, *params: float) -> tuple:
        """Convert kernel parameters to intervening LTI system (A, B, C, D).

        This method creates the intervening LTI system that generates the
        kernel's impulse response. For unstable kernels, this LTI system
        should be simulated explicitly instead of using convolution.

        Args:
            *params: Kernel parameters in the order defined by param_names.

        Returns:
            Tuple of (A, B, C, D) matrices for the intervening LTI system.
            Returns None if the kernel cannot be represented as an LTI system
            or if it's stable (should use convolution instead).
        """
        return None

    def make_kwargs(self, params: np.ndarray) -> dict:
        """Convert flat parameter array to a kwargs dict keyed by param_names."""
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

    @property
    def is_unstable(self) -> bool:
        return False


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

    @property
    def is_unstable(self) -> bool:
        return False


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

    @property
    def is_unstable(self) -> bool:
        return False


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
                [0.001, 5.0],   # zeta: strictly positive for stable delay dynamics
                [0.001, 20.0],   # omega_n: tighter upper bound
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([0.1, 2.0])

    def kernel_fn(self, t: np.ndarray, zeta: float, omega_n: float) -> np.ndarray:  # type: ignore[override]
        # Handle different damping regimes
        if zeta < -1.0:
            # Unstable real poles (zeta < -1): pure exponential growth
            # Poles are at -zeta*omega_n +/- omega_n*sqrt(zeta^2 - 1)
            # The dominant pole has growth rate = -zeta*omega_n + omega_n*sqrt(zeta^2 - 1)
            s = omega_n * np.sqrt(zeta**2 - 1.0)
            growth_rate = -zeta * omega_n + s
            h = growth_rate * np.exp(growth_rate * t)
        elif -1.0 <= zeta < 1.0:
            # Underdamped or growing oscillatory (-1 < zeta < 1)
            omega_d = omega_n * np.sqrt(1.0 - zeta**2)
            amplitude = omega_n / omega_d
            exponent = -zeta * omega_n * t
            # Clip exponent to prevent overflow (exp(700) ~ 1e304, near float64 max)
            max_exponent = 700.0
            exponent = np.clip(exponent, -max_exponent, max_exponent)
            h = amplitude * np.exp(exponent) * np.sin(omega_d * t)
        elif zeta == 1.0:
            # Critically damped: h(t) = omega_n^2 * t * exp(-omega_n * t)
            h = omega_n**2 * t * np.exp(-omega_n * t)
        else:
            # Overdamped (zeta > 1): numerically stable form using difference of exponentials
            # h(t) = (omega_n/(2*s)) * [exp((-zeta*omega_n + s)*t) - exp((-zeta*omega_n - s)*t)]
            # where s = omega_n*sqrt(zeta^2 - 1)
            s = omega_n * np.sqrt(zeta**2 - 1.0)
            decay1 = -zeta * omega_n + s
            decay2 = -zeta * omega_n - s
            # Clip exponents to prevent overflow
            max_exponent = 700.0
            decay1 = np.clip(decay1, -max_exponent, max_exponent)
            decay2 = np.clip(decay2, -max_exponent, max_exponent)
            h = (omega_n / (2.0 * s)) * (np.exp(decay1 * t) - np.exp(decay2 * t))
        if zeta < 0:
            return h  # type: ignore[no-any-return]
        return np.maximum(h, 0.0)  # type: ignore[no-any-return]

    @property
    def is_unstable(self) -> bool:
        # This kernel can be unstable depending on parameters
        return True

    def is_unstable_params(self, zeta: float, omega_n: float) -> bool:
        return False  # With zeta > 0 bounds, underdamped is always stable delay

    def is_stable_delay(self, zeta: float, omega_n: float) -> bool:
        """Check if the delay dynamics are stable.
        
        For underdamped kernel, delay dynamics are stable when zeta > 0.
        For zeta <= 0, the delay dynamics are unstable.
        """
        return zeta > 0

    def to_lti(self, zeta: float, omega_n: float) -> tuple:
        """Convert underdamped oscillator parameters to intervening LTI system.

        The underdamped oscillator corresponds to a 2nd-order LTI system:
        A = [[0, 1], [-omega_n^2, -2*zeta*omega_n]]
        B = [[0], [1]]
        C = [[omega_n, 0]]  (for the standard impulse response)
        D = [[0]]
        """
        A = np.array([
            [0.0, 1.0],
            [-(omega_n**2), -2.0 * zeta * omega_n]
        ])
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
                [-5.0, -0.01],  # rate: negative for stable delay dynamics (decay)
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
        return False  # With rate < 0 bounds, always stable delay

    def is_stable_delay(self, rate: float) -> bool:
        """Check if the delay dynamics are stable.
        
        For exponential growth kernel, delay dynamics are stable when rate < 0 (decay).
        """
        return rate < 0

    def to_lti(self, rate: float) -> tuple:
        """Convert exponential growth kernel to intervening LTI system.

        The exponential growth kernel corresponds to a 1st-order LTI system:
        A = [[rate]]
        B = [[1]]
        C = [[rate]]  (so impulse response is rate * exp(rate * t))
        D = [[0]]
        """
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
        """Check if the delay dynamics are stable.
        
        For exponential decay kernel, delay dynamics are stable when lambda > 0 (decay).
        """
        return lam > 0

    def to_lti(self, lam: float) -> tuple:
        """Convert exponential decay kernel to intervening LTI system.

        The exponential decay kernel corresponds to a 1st-order LTI system:
        A = [[-lam]]
        B = [[1]]
        C = [[lam]]  (so impulse response is lam * exp(-lam * t))
        D = [[0]]
        """
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
                [-10.0, -0.01],  # lambda: negative for stable delay dynamics (decay)
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
        return False  # With lambda < 0 bounds, always stable delay

    def is_stable_delay(self, lam: float) -> bool:
        """Check if the delay dynamics are stable.
        
        For exponential kernel, delay dynamics are stable when lambda < 0 (decay).
        """
        return lam < 0

    def to_lti(self, lam: float) -> tuple:
        """Convert exponential kernel to intervening LTI system.

        The exponential kernel corresponds to a 1st-order LTI system:
        A = [[lam]]
        B = [[1]]
        C = [[lam]]  (so impulse response is lam * exp(lam * t))
        D = [[0]]
        """
        A = np.array([[lam]])
        B = np.array([[1.0]])
        C = np.array([[lam]])
        D = np.array([[0.0]])
        return A, B, C, D


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
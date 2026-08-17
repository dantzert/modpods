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
    zeta is clamped to (0, 1) for underdamped response.  The kernel is truncated
    to non-negative values for causality.

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
                [0.01, 0.99],
                [0.1, 10.0],
            ]
        )

    @property
    def default_init(self) -> np.ndarray:
        return np.array([0.1, 2.0])

    def kernel_fn(self, t: np.ndarray, zeta: float, omega_n: float) -> np.ndarray:  # type: ignore[override]
        omega_d = omega_n * np.sqrt(1.0 - zeta**2)
        amplitude = omega_n / omega_d
        h = amplitude * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)
        return np.maximum(h, 0.0)  # type: ignore[no-any-return]


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

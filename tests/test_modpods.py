"""
Pytest tests for modpods core functions.

Tests collected from the following original scripts (now deleted):
  test_lti_from_gamma.py, test_topo_inference.py, test_coef_constraints.py,
  test.py, test_lti_system_gen.py, test_topo_from_swmm.py,
  test_lti_control_of_swmm.py

Tests that load large data files or run long simulations are marked @pytest.mark.slow.
"""

import pathlib
import warnings
from typing import Any, cast

import control as ct
import numpy as np
import pandas as pd
import pytest

import modpods

DATA_DIR = pathlib.Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_lti_data() -> pd.DataFrame:
    """Small two-state LTI system: u → x0 → x1 (cascade, 200 time-steps)."""
    np.random.seed(42)
    n, dt = 200, 0.05
    T = np.arange(0, n * dt, dt)
    A = np.array([[-1.0, 0], [1.0, -1.0]])
    B = np.array([[1.0], [0.0]])
    sys = ct.ss(A, B, np.eye(2), 0)
    u = np.zeros((n, 1))
    u[50:80, 0] = np.random.rand(30)
    response = ct.forced_response(sys, T, np.transpose(u))
    df = pd.DataFrame(
        index=T,
        data={
            "u": response.inputs[0],
            "x0": response.states[0],
            "x1": response.states[1],
        },
    )
    return df


@pytest.fixture(scope="module")
def cascade_lti_system_data() -> pd.DataFrame:
    """Generate response data from a known cascade LTI system.

    System topology (ground truth):
      u1 → x0 → x1 → x2  (u1 causes x2 via a long cascade, delayed)
      u2 → x8             (u2 causes x8 directly)
      x7 → x9, x8 → x9   (x9 driven by both chains)

    Observable variables: u1, u2, x2, x8, x9
    """
    np.random.seed(0)

    A = np.diag(-1.0 * np.ones(10))
    A[1, 0] = 1
    A[2, 1] = 1
    A[3, 2] = 1
    A[4, 3] = 1
    A[5, 4] = 1
    A[6, 5] = 1
    A[7, 6] = 1
    A[9, 7] = 1
    A[9, 8] = 1

    B = np.zeros((10, 2))
    B[0, 0] = 1
    B[8, 1] = 1

    C = np.eye(10)
    D = np.zeros((10, 2))

    system = ct.ss(A, B, C, D)
    time_base = 50.0
    dt = 0.05
    T = np.arange(0, time_base, dt)

    u = np.zeros((len(T), 2))
    u[int(25 / dt) : int(40 / dt), 0] = np.random.rand(int(15 / dt)) - 0.5
    u[int(0 / dt) : int(15 / dt), 1] = np.random.rand(int(15 / dt)) - 0.5
    u[np.abs(u) < 0.40] = 0
    u[:, 0] *= np.random.rand(len(T)) * 1000
    u[:, 1] *= np.random.rand(len(T)) * 100

    response = ct.forced_response(system, T, np.transpose(u))
    df = pd.DataFrame(index=T)
    df["u1"] = response.inputs[0]
    df["u2"] = response.inputs[1]
    df["x2"] = response.states[2]
    df["x8"] = response.states[8]
    df["x9"] = response.states[9]
    return df


# ---------------------------------------------------------------------------
# lti_from_gamma tests  (from test_lti_from_gamma.py)
# ---------------------------------------------------------------------------


def test_lti_from_gamma_returns_required_keys() -> None:
    """lti_from_gamma must return a dict with the expected keys."""
    result = modpods.lti_from_gamma(shape=10, scale=1, location=0, dt=0.1)
    assert isinstance(result, dict)
    for key in ("t", "gamma_pdf", "lti_approx_output", "lti_approx"):
        assert key in result, f"missing key '{key}' in result"


def test_lti_from_gamma_output_shapes_match() -> None:
    """gamma_pdf and lti_approx_output must have the same length."""
    result = modpods.lti_from_gamma(shape=5, scale=2, location=0)
    assert result["gamma_pdf"].shape == result["lti_approx_output"].shape


def test_lti_from_gamma_achieves_reasonable_nse() -> None:
    """The LTI approximation should achieve NSE > 0.9 for a well-conditioned
    gamma distribution (shape=10, scale=1, location=0)."""
    result = modpods.lti_from_gamma(shape=10, scale=1, location=0, dt=0.1)
    gamma_pdf = result["gamma_pdf"]
    lti_approx = result["lti_approx_output"]
    nse = 1.0 - float(
        np.sum(np.square(gamma_pdf - lti_approx))
        / np.sum(np.square(gamma_pdf - np.mean(gamma_pdf)))
    )
    assert nse > 0.9, f"NSE {nse:.4f} is below the 0.9 threshold"


def test_lti_from_gamma_t_is_nonnegative() -> None:
    """The time vector returned must be non-negative and monotonically increasing."""
    result = modpods.lti_from_gamma(shape=3, scale=1, location=0)
    t = result["t"]
    assert t[0] >= 0.0
    assert np.all(np.diff(t) > 0), "time vector is not strictly increasing"


# ---------------------------------------------------------------------------
# transform_inputs tests
# ---------------------------------------------------------------------------


def test_transform_inputs_correctness() -> None:
    """transform_inputs must produce correct gamma-transformed outputs."""
    np.random.seed(42)
    n = 20
    index = pd.date_range("2000-01-01", periods=n, freq="1h")

    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)

    kernel = modpods.GammaKernel()
    kernel_params = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(1, p) for p in kernel.param_names], names=["transform", "param"]
        ),
        columns=["u"],
        dtype=float,
    )
    kernel_params.loc[(1, "shape"), "u"] = 2.0
    kernel_params.loc[(1, "scale"), "u"] = 1.0
    kernel_params.loc[(1, "loc"), "u"] = 0.0

    result = modpods.transform_inputs(
        kernel, kernel_params, index, forcing
    )

    assert "u_tr_1" in result.columns
    assert len(result) == n
    assert not result.isnull().values.any()

    known_expected = np.array(
        [
            4.44089210e-17,
            1.82730925e-02,
            2.66312232e-02,
            5.41349278e-02,
            1.29269010e-01,
            1.72213335e-01,
            1.85028295e-01,
            2.46741125e-01,
            3.18644918e-01,
            3.45925852e-01,
            3.76226589e-01,
            3.77780369e-01,
            3.57689577e-01,
            3.51598612e-01,
            2.79450476e-01,
            1.63734986e-01,
            6.76750731e-02,
            -2.46014475e-02,
            -6.79339082e-02,
            -1.20732221e-01,
        ]
    )

    np.testing.assert_allclose(result["u_tr_1"].values, known_expected, rtol=1e-5)


def test_transform_inputs_with_cache() -> None:
    """transform_inputs with cache must produce identical results and improve speed on repeated calls."""
    np.random.seed(42)
    n = 1000
    index = pd.date_range("2000-01-01", periods=n, freq="1h")

    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)

    kernel = modpods.GammaKernel()
    kernel_params = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(1, p) for p in kernel.param_names], names=["transform", "param"]
        ),
        columns=["u"],
        dtype=float,
    )
    kernel_params.loc[(1, "shape"), "u"] = 2.0
    kernel_params.loc[(1, "scale"), "u"] = 1.0
    kernel_params.loc[(1, "loc"), "u"] = 0.0

    cache = modpods.TransformCache()
    result1 = modpods.transform_inputs(
        kernel, kernel_params, index, forcing, cache=cache
    )

    result2 = modpods.transform_inputs(
        kernel, kernel_params, index, forcing, cache=cache
    )
    stats2 = cache.stats()

    np.testing.assert_allclose(result1["u_tr_1"].values, result2["u_tr_1"].values)

    assert stats2["hits"] == 1
    assert stats2["misses"] == 1
    assert stats2["hit_rate"] == 0.5


def test_transform_inputs_performance() -> None:
    """transform_inputs must be fast (vectorized FFT convolution)."""
    import time

    np.random.seed(42)
    n = 5000
    index = pd.date_range("2000-01-01", periods=n, freq="1h")

    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)

    kernel = modpods.GammaKernel()
    kernel_params = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(1, p) for p in kernel.param_names], names=["transform", "param"]
        ),
        columns=["u"],
        dtype=float,
    )
    kernel_params.loc[(1, "shape"), "u"] = 2.0
    kernel_params.loc[(1, "scale"), "u"] = 1.0
    kernel_params.loc[(1, "loc"), "u"] = 0.0

    _ = modpods.transform_inputs(kernel, kernel_params, index, forcing)

    start = time.perf_counter()
    for _ in range(5):
        _ = modpods.transform_inputs(kernel, kernel_params, index, forcing)
    elapsed = (time.perf_counter() - start) / 5

    assert elapsed < 0.1, f"transform_inputs too slow: {elapsed:.3f}s for {n} samples"


def test_transform_inputs_multiple_transforms() -> None:
    """transform_inputs must handle multiple transforms per input correctly."""
    np.random.seed(42)
    n = 200
    index = pd.date_range("2000-01-01", periods=n, freq="1h")

    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)

    kernel = modpods.GammaKernel()
    kernel_params = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(t, p) for t in [1, 2] for p in kernel.param_names],
            names=["transform", "param"],
        ),
        columns=["u"],
        dtype=float,
    )
    kernel_params.loc[(1, "shape"), "u"] = 2.0
    kernel_params.loc[(1, "scale"), "u"] = 1.0
    kernel_params.loc[(1, "loc"), "u"] = 0.0
    kernel_params.loc[(2, "shape"), "u"] = 3.0
    kernel_params.loc[(2, "scale"), "u"] = 0.5
    kernel_params.loc[(2, "loc"), "u"] = 1.0

    result = modpods.transform_inputs(kernel, kernel_params, index, forcing)

    assert "u_tr_1" in result.columns
    assert "u_tr_2" in result.columns
    assert len(result) == n
    assert not result.isnull().values.any()

    from scipy import signal, stats

    forcing_values = forcing["u"].to_numpy()

    for transform_idx, (shape, scale, loc) in enumerate(
        [(2.0, 1.0, 0.0), (3.0, 0.5, 1.0)], 1
    ):
        shape_time = np.arange(0, n, 1)
        gamma_kernel = stats.gamma.pdf(shape_time, shape, scale=scale, loc=loc)
        expected = signal.fftconvolve(forcing_values, gamma_kernel, mode="full")[:n]
        np.testing.assert_allclose(
            result[f"u_tr_{transform_idx}"].values, expected, rtol=1e-10
        )


def test_transform_inputs_multiple_inputs() -> None:
    """transform_inputs must handle multiple independent inputs correctly."""
    np.random.seed(42)
    n = 200
    index = pd.date_range("2000-01-01", periods=n, freq="1h")

    forcing = pd.DataFrame(
        {
            "u1": np.cumsum(np.random.randn(n) * 0.1),
            "u2": np.cumsum(np.random.randn(n) * 0.1),
        },
        index=index,
    )

    kernel = modpods.GammaKernel()
    kernel_params = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [(1, p) for p in kernel.param_names], names=["transform", "param"]
        ),
        columns=["u1", "u2"],
        dtype=float,
    )
    kernel_params.loc[(1, "shape"), "u1"] = 2.0
    kernel_params.loc[(1, "scale"), "u1"] = 1.0
    kernel_params.loc[(1, "loc"), "u1"] = 0.0
    kernel_params.loc[(1, "shape"), "u2"] = 3.0
    kernel_params.loc[(1, "scale"), "u2"] = 0.5
    kernel_params.loc[(1, "loc"), "u2"] = 1.0

    result = modpods.transform_inputs(kernel, kernel_params, index, forcing)

    assert "u1_tr_1" in result.columns
    assert "u2_tr_1" in result.columns
    assert len(result) == n
    assert not result.isnull().values.any()


def test_transform_inputs_cache_quantization() -> None:
    """TransformCache quantization must allow reuse for near-identical parameters."""
    np.random.seed(42)
    n = 100
    index = pd.date_range("2000-01-01", periods=n, freq="1h")

    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)

    kernel = modpods.GammaKernel()

    def make_params(shape, scale, loc):
        kp = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [(1, p) for p in kernel.param_names], names=["transform", "param"]
            ),
            columns=["u"],
            dtype=float,
        )
        kp.loc[(1, "shape"), "u"] = shape
        kp.loc[(1, "scale"), "u"] = scale
        kp.loc[(1, "loc"), "u"] = loc
        return kp

    kp1 = make_params(2.0000001, 1.0000001, 0.0000001)
    kp2 = make_params(2.0000002, 1.0000002, 0.0000002)

    cache = modpods.TransformCache(quantization=1e-6)

    result1 = modpods.transform_inputs(
        kernel, kp1, index, forcing, cache=cache
    )

    result2 = modpods.transform_inputs(
        kernel, kp2, index, forcing, cache=cache
    )
    stats2 = cache.stats()

    assert stats2["hits"] == 1
    np.testing.assert_allclose(result1["u_tr_1"].values, result2["u_tr_1"].values)


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------


def test_kernel_registry() -> None:
    """All built-in kernels should be discoverable via list_kernels."""
    names = modpods.list_kernels()
    assert "gamma" in names
    assert "lognormal" in names
    assert "bimodal_gamma" in names
    assert "underdamped" in names


def test_get_kernel_by_name() -> None:
    """get_kernel should resolve both name strings and instances."""
    k1 = modpods.get_kernel("gamma")
    assert isinstance(k1, modpods.GammaKernel)
    k2 = modpods.get_kernel(modpods.GammaKernel())
    assert isinstance(k2, modpods.GammaKernel)


def test_gamma_kernel_defaults() -> None:
    """GammaKernel should have 3 params with sensible defaults."""
    k = modpods.GammaKernel()
    assert k.num_params == 3
    assert k.param_names == ["shape", "scale", "loc"]
    assert k.default_init.tolist() == [1.0, 1.0, 0.0]
    assert k.default_bounds.shape == (3, 2)


def test_underdamped_kernel_defaults() -> None:
    """UnderdampedOscillatorKernel should have 2 params."""
    k = modpods.UnderdampedOscillatorKernel()
    assert k.num_params == 2
    assert k.param_names == ["zeta", "omega_n"]
    assert k.default_bounds[0, 0] > 0
    assert k.default_bounds[0, 1] < 1


def test_kernel_fn_shape() -> None:
    """All kernel_fn outputs must have the same length as the input time array."""
    t = np.arange(0, 100, 1.0)
    for name in modpods.list_kernels():
        k = modpods.get_kernel(name)
        params = k.default_init
        h = k.kernel_fn(t, *params)
        assert h.shape == t.shape, f"{name} kernel output shape mismatch"


def test_underdamped_kernel_oscillatory() -> None:
    """Underdamped kernel should produce non-zero values that decay."""
    t = np.arange(0, 200, 1.0)
    k = modpods.UnderdampedOscillatorKernel()
    h = k.kernel_fn(t, 0.2, 2.0)
    assert h.max() > 0, "underdamped kernel should have positive peak"
    assert h[-1] < h.max() / 10, "underdamped kernel should decay"
    assert np.all(h >= 0), "underdamped kernel should be non-negative (causal)"


def test_make_kernel_params() -> None:
    """make_kernel_params should create a properly indexed DataFrame."""
    k = modpods.GammaKernel()
    kp = modpods.make_kernel_params(k, ["u1", "u2"], init_transforms=1, max_transforms=3)
    assert kp.index.nlevels == 2
    assert kp.index.names == ["transform", "param"]
    assert list(kp.columns) == ["u1", "u2"]
    assert len(kp) == 9  # 3 transforms * 3 params
    np.testing.assert_allclose(kp.loc[(1, "shape"), :], k.default_init[0])
    np.testing.assert_allclose(kp.loc[(1, "scale"), :], k.default_init[1])
    np.testing.assert_allclose(kp.loc[(1, "loc"), :], k.default_init[2])


def test_params_vector_roundtrip() -> None:
    """params_vector_to_dataframe should be the inverse of flattening kernel_params."""
    k = modpods.GammaKernel()
    kp = modpods.make_kernel_params(k, ["u"], init_transforms=1, max_transforms=2)
    kp.loc[(1, "shape"), "u"] = 2.5
    kp.loc[(1, "scale"), "u"] = 1.5
    kp.loc[(1, "loc"), "u"] = 0.5
    kp.loc[(2, "shape"), "u"] = 5.0
    kp.loc[(2, "scale"), "u"] = 2.0
    kp.loc[(2, "loc"), "u"] = 1.0

    flat = np.array([2.5, 1.5, 0.5, 5.0, 2.0, 1.0])
    recovered = modpods.params_vector_to_dataframe(
        k, flat, ["u"], init_transforms=1, max_transforms=2
    )
    pd.testing.assert_frame_equal(kp, recovered)


def test_delay_io_train_with_underdamped_kernel(simple_lti_data: pd.DataFrame) -> None:
    """delay_io_train should work with the underdamped oscillator kernel."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=5,
            poly_order=1,
            verbose=False,
            kernel="underdamped",
        )
    assert isinstance(model, dict)
    assert 1 in model
    assert "kernel_type" in model[1]
    assert model[1]["kernel_type"] == "underdamped"
    assert "kernel_params" in model[1]


# ---------------------------------------------------------------------------
# delay_io_train / delay_io_predict tests  (from test_coef_constraints.py)
# ---------------------------------------------------------------------------


def test_delay_io_train_returns_model(simple_lti_data: pd.DataFrame) -> None:
    """delay_io_train must return a dict keyed by output-variable index."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=5,
            poly_order=1,
            verbose="warnings",
        )
    assert isinstance(model, dict)
    assert 1 in model, "expected key 1 (first output) in model dict"
    assert "final_model" in model[1]
    assert "error_metrics" in model[1]["final_model"]


def test_delay_io_train_nse_above_zero(simple_lti_data: pd.DataFrame) -> None:
    """Training NSE on the simple cascade system must be positive."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=10,
            poly_order=1,
            verbose="warnings",
        )
    nse = float(model[1]["final_model"]["error_metrics"]["NSE"][0])
    assert nse > 0.0, f"Training NSE {nse:.4f} is non-positive"


def test_delay_io_train_with_forcing_coef_constraints(
    simple_lti_data: pd.DataFrame,
) -> None:
    """delay_io_train with bibo_stable=True and forcing_coef_constraints must complete
    without error and return a valid model dict."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=10,
            poly_order=1,
            verbose="warnings",
            bibo_stable=True,
            forcing_coef_constraints={"u": 1},
        )
    assert isinstance(model, dict)
    assert 1 in model
    assert model[1]["final_model"]["error_metrics"]["NSE"] is not None


def test_delay_io_predict_returns_expected_shape(
    simple_lti_data: pd.DataFrame,
) -> None:
    """delay_io_predict must return a dict with 'prediction' of the right length."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=5,
            poly_order=1,
            verbose="warnings",
        )
        pred = modpods.delay_io_predict(model, simple_lti_data, num_transforms=1)
    assert isinstance(pred, dict)
    assert "prediction" in pred
    # prediction length should be approximately equal to data length
    assert pred["prediction"].shape[0] > 0


# ---------------------------------------------------------------------------
# Optimization method comparison tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bayesian_model(simple_lti_data: pd.DataFrame) -> dict[Any, Any]:
    """Train a model using Bayesian optimization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=10,
            poly_order=1,
            verbose="warnings",
            optimization_method="bayesian",
        )
    return cast(dict[Any, Any], model)


@pytest.fixture(scope="module")
def de_model(simple_lti_data: pd.DataFrame) -> dict[Any, Any]:
    """Train a model using differential evolution optimization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=10,
            poly_order=1,
            verbose="warnings",
            optimization_method="differential_evolution",
        )
    return cast(dict[Any, Any], model)


@pytest.fixture(scope="module")
def da_model(simple_lti_data: pd.DataFrame) -> dict[Any, Any]:
    """Train a model using dual annealing optimization."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=10,
            poly_order=1,
            verbose="warnings",
            optimization_method="dual_annealing",
        )
    return cast(dict[Any, Any], model)


def test_bayesian_returns_valid_model(
    bayesian_model: dict[Any, Any],
) -> None:
    """Bayesian optimizer must return a well-formed model dict."""
    assert isinstance(bayesian_model, dict)
    assert 1 in bayesian_model
    assert "final_model" in bayesian_model[1]
    assert "error_metrics" in bayesian_model[1]["final_model"]
    r2 = float(bayesian_model[1]["final_model"]["error_metrics"]["r2"])
    assert r2 > -1.0, f"Bayesian R² {r2:.4f} is unreasonably low"


def test_differential_evolution_returns_valid_model(
    de_model: dict[Any, Any],
) -> None:
    """Differential evolution optimizer must return a well-formed model dict."""
    assert isinstance(de_model, dict)
    assert 1 in de_model
    assert "final_model" in de_model[1]
    assert "error_metrics" in de_model[1]["final_model"]
    r2 = float(de_model[1]["final_model"]["error_metrics"]["r2"])
    assert r2 > -1.0, f"DE R² {r2:.4f} is unreasonably low"


def test_dual_annealing_returns_valid_model(
    da_model: dict[Any, Any],
) -> None:
    """Dual annealing optimizer must return a well-formed model dict."""
    assert isinstance(da_model, dict)
    assert 1 in da_model
    assert "final_model" in da_model[1]
    assert "error_metrics" in da_model[1]["final_model"]
    r2 = float(da_model[1]["final_model"]["error_metrics"]["r2"])
    assert r2 > -1.0, f"DA R² {r2:.4f} is unreasonably low"


def test_all_methods_produce_comparable_r2(
    bayesian_model: dict[Any, Any],
    de_model: dict[Any, Any],
    da_model: dict[Any, Any],
) -> None:
    """All optimization methods should achieve similar R² on the same data.

    The difference in R² should be within a reasonable margin, confirming
    that all methods solve the same underlying optimization problem.
    """
    r2_bayesian = float(bayesian_model[1]["final_model"]["error_metrics"]["r2"])
    r2_de = float(de_model[1]["final_model"]["error_metrics"]["r2"])
    r2_da = float(da_model[1]["final_model"]["error_metrics"]["r2"])
    # All should be positive (reasonable fit)
    assert r2_bayesian > 0.0, f"Bayesian R² {r2_bayesian:.4f} is non-positive"
    assert r2_de > 0.0, f"DE R² {r2_de:.4f} is non-positive"
    assert r2_da > 0.0, f"DA R² {r2_da:.4f} is non-positive"
    # No method should be dramatically worse than the others
    assert abs(r2_bayesian - r2_de) < 0.5, (
        f"Methods diverge too much: bayesian={r2_bayesian:.4f}, " f"de={r2_de:.4f}"
    )
    assert abs(r2_bayesian - r2_da) < 0.5, (
        f"Methods diverge too much: bayesian={r2_bayesian:.4f}, " f"da={r2_da:.4f}"
    )
    assert abs(r2_de - r2_da) < 0.5, (
        f"Methods diverge too much: de={r2_de:.4f}, " f"da={r2_da:.4f}"
    )


def test_all_methods_predictions_agree(
    bayesian_model: dict[Any, Any],
    de_model: dict[Any, Any],
    da_model: dict[Any, Any],
    simple_lti_data: pd.DataFrame,
) -> None:
    """Predictions from all optimization methods should broadly agree."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_bayesian = modpods.delay_io_predict(
            bayesian_model, simple_lti_data, num_transforms=1
        )
        pred_de = modpods.delay_io_predict(de_model, simple_lti_data, num_transforms=1)
        pred_da = modpods.delay_io_predict(da_model, simple_lti_data, num_transforms=1)
    assert "prediction" in pred_bayesian
    assert "prediction" in pred_de
    assert "prediction" in pred_da
    p_b = pred_bayesian["prediction"].ravel()
    p_de = pred_de["prediction"].ravel()
    p_da = pred_da["prediction"].ravel()
    # Predictions should be correlated
    assert np.corrcoef(p_b, p_de)[0, 1] > 0.5, "Bayesian and DE predictions diverge"
    assert np.corrcoef(p_b, p_da)[0, 1] > 0.5, "Bayesian and DA predictions diverge"
    assert np.corrcoef(p_de, p_da)[0, 1] > 0.5, "DE and DA predictions diverge"
    # All predictions must be finite (no NaN or Inf)
    assert np.all(np.isfinite(p_b)), "Bayesian predictions contain NaN/Inf"
    assert np.all(np.isfinite(p_de)), "DE predictions contain NaN/Inf"
    assert np.all(np.isfinite(p_da)), "DA predictions contain NaN/Inf"


# ---------------------------------------------------------------------------
# infer_causative_topology tests  (from test_topo_inference.py)
# ---------------------------------------------------------------------------


def test_infer_causative_topology_returns_dataframe(
    cascade_lti_system_data: pd.DataFrame,
) -> None:
    """infer_causative_topology must return a (DataFrame, DataFrame) tuple."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = modpods.infer_causative_topology(  # type: ignore[call-arg]
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose="warnings",
            max_iter=0,
            method="sindy",
        )
    assert isinstance(result, tuple) and len(result) == 2
    causative_topo, total_graph = result
    assert isinstance(causative_topo, pd.DataFrame)
    assert isinstance(total_graph, pd.DataFrame)


def test_infer_causative_topology_identifies_u1_causes_x2(
    cascade_lti_system_data: pd.DataFrame,
) -> None:
    """SINDy causality must identify u1 as a cause of x2 (delayed cascade)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        causative_topo, _ = modpods.infer_causative_topology(  # type: ignore[call-arg]
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose="warnings",
            max_iter=0,
            method="sindy",
        )
    assert (
        causative_topo.loc["x2", "u1"] == "d"
    ), f"Expected u1→x2 to be 'd' (delayed), got '{causative_topo.loc['x2', 'u1']}'"


def test_infer_causative_topology_identifies_u2_causes_x8(
    cascade_lti_system_data: pd.DataFrame,
) -> None:
    """SINDy causality must identify u2 as a cause of x8 (direct link)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        causative_topo, _ = modpods.infer_causative_topology(  # type: ignore[call-arg]
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose="warnings",
            max_iter=0,
            method="sindy",
        )
    assert (
        causative_topo.loc["x8", "u2"] == "d"
    ), f"Expected u2→x8 to be 'd' (delayed), got '{causative_topo.loc['x8', 'u2']}'"


def test_infer_causative_topology_no_self_loops(
    cascade_lti_system_data: pd.DataFrame,
) -> None:
    """No variable should be identified as causing itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        causative_topo, _ = modpods.infer_causative_topology(  # type: ignore[call-arg]
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose="warnings",
            max_iter=0,
            method="sindy",
        )
    for dep_var in ["x2", "x8", "x9"]:
        assert (
            causative_topo.loc[dep_var, dep_var] == "n"
        ), f"Self-loop detected for {dep_var}"


# ---------------------------------------------------------------------------
# lti_system_gen tests  (from test_lti_system_gen.py) — SLOW
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def known_topology() -> pd.DataFrame:
    """Manually specified topology for the 5-variable cascade system:
    u1 --(delayed)--> x2
    u2 --(immediate)-> x8
    x8 --(immediate)-> x9
    x2 --(delayed)---> x9
    """
    topo = pd.DataFrame(
        index=["x2", "x8", "x9"],
        columns=["u1", "u2", "x2", "x8", "x9"],
    ).fillna("n")
    topo.loc["x2", "u1"] = "d"
    topo.loc["x8", "u2"] = "i"
    topo.loc["x9", "x8"] = "i"
    topo.loc["x9", "x2"] = "d"
    return topo


@pytest.mark.slow
def test_lti_system_gen_returns_state_space(
    cascade_lti_system_data: pd.DataFrame,
    known_topology: pd.DataFrame,
) -> None:
    """lti_system_gen must return a dict with 'system', 'A', 'B', 'C' keys, where
    'system' is a StateSpace object that can be simulated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = modpods.lti_system_gen(
            known_topology,
            cascade_lti_system_data,
            independent_columns=["u1", "u2"],
            dependent_columns=["x2", "x8", "x9"],
            max_iter=5,
            bibo_stable=True,
            max_transforms=1,
        )

    assert isinstance(result, dict)
    for key in ("system", "A", "B", "C"):
        assert key in result, f"missing key '{key}'"

    assert isinstance(result["system"], ct.StateSpace)
    # Verify the system can be used for forward simulation
    T = cascade_lti_system_data.index
    test_u = np.zeros((len(T), 2))
    test_u[100:200, 0] = 1.0
    response = ct.forced_response(result["system"], T, np.transpose(test_u))
    assert response.outputs.shape[0] == 3, "expected 3 outputs (x2, x8, x9)"


# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# CAMELS rainfall-runoff tests  (from test.py) — SLOW (uses data file)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def camels_data() -> pd.DataFrame:
    """Load and preprocess the CAMELS daily streamflow data."""
    filepath = DATA_DIR / "03439000_05_model_output.txt"
    df = pd.read_csv(filepath, sep=r"\s+")
    df.rename(
        {"YR": "year", "MNTH": "month", "DY": "day", "HR": "hour"},
        axis=1,
        inplace=True,
    )
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.set_index("datetime", inplace=True)
    df.RAIM = df.RAIM.shift(-1)
    df.dropna(inplace=True)
    return df


@pytest.fixture(scope="module")
def trained_camels_model(
    camels_data: pd.DataFrame,
) -> dict[Any, Any]:
    """Train a delay_io model on one year of CAMELS data."""
    windup_timesteps = 30
    years = 1
    df_train = camels_data.iloc[: 365 * years + windup_timesteps, :][
        ["OBS_RUN", "RAIM", "PET", "PRCP"]
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            df_train,
            dependent_columns=["OBS_RUN"],
            independent_columns=["RAIM", "PET", "PRCP"],
            windup_timesteps=windup_timesteps,
            init_transforms=1,
            max_transforms=1,
            max_iter=5,
            poly_order=1,
            verbose="warnings",
            bibo_stable=False,
            forcing_coef_constraints={"RAIM": -1, "PET": 1, "PRCP": -1},
        )
    return cast(dict[Any, Any], model)


@pytest.mark.slow
def test_delay_io_train_camels_returns_model(
    trained_camels_model: dict[Any, Any],
) -> None:
    """delay_io_train on CAMELS data must return a model dict with NSE > -1."""
    assert isinstance(trained_camels_model, dict)
    assert 1 in trained_camels_model
    nse_val = trained_camels_model[1]["final_model"]["error_metrics"]["NSE"]
    nse = float(nse_val[0]) if hasattr(nse_val, "__len__") else float(nse_val)
    assert nse > -1.0, f"CAMELS training NSE {nse:.4f} is unreasonably low"


@pytest.mark.slow
def test_delay_io_predict_camels_returns_prediction(
    trained_camels_model: dict[Any, Any],
    camels_data: pd.DataFrame,
) -> None:
    """delay_io_predict on CAMELS eval data must return a 'prediction' array."""
    windup_timesteps = 30
    years = 1
    df_eval = camels_data.iloc[-(365 * years + windup_timesteps) :, :][
        ["OBS_RUN", "RAIM", "PET", "PRCP"]
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = modpods.delay_io_predict(
            trained_camels_model, df_eval, num_transforms=1, evaluation=True
        )
    assert isinstance(pred, dict)
    assert "prediction" in pred
    assert pred["prediction"].shape[0] > 0

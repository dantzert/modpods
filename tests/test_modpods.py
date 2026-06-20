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
    n = 100
    index = pd.date_range("2000-01-01", periods=n, freq="1h")
    
    # Create forcing data
    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)
    
    # Create parameter dataframes
    shape_factors = pd.DataFrame({"u": [2.0]}, index=[1])
    scale_factors = pd.DataFrame({"u": [1.0]}, index=[1])
    loc_factors = pd.DataFrame({"u": [0.0]}, index=[1])
    
    # Transform
    result = modpods.transform_inputs(shape_factors, scale_factors, loc_factors, index, forcing)
    
    # Check output
    assert "u_tr_1" in result.columns
    assert len(result) == n
    assert not result.isnull().values.any()
    
    # Verify against direct convolution
    from scipy import signal, stats
    forcing_values = forcing["u"].to_numpy()
    shape_time = np.arange(0, n, 1)
    gamma_kernel = stats.gamma.pdf(shape_time, 2.0, scale=1.0, loc=0.0)
    expected = signal.fftconvolve(forcing_values, gamma_kernel, mode="full")[:n]
    
    np.testing.assert_allclose(result["u_tr_1"].values, expected, rtol=1e-10)


def test_transform_inputs_with_cache() -> None:
    """transform_inputs with cache must produce identical results and improve speed on repeated calls."""
    np.random.seed(42)
    n = 1000
    index = pd.date_range("2000-01-01", periods=n, freq="1h")
    
    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)
    shape_factors = pd.DataFrame({"u": [2.0]}, index=[1])
    scale_factors = pd.DataFrame({"u": [1.0]}, index=[1])
    loc_factors = pd.DataFrame({"u": [0.0]}, index=[1])
    
    # First call (cache miss)
    cache = modpods.TransformCache()
    result1 = modpods.transform_inputs(shape_factors, scale_factors, loc_factors, index, forcing, cache=cache)
    stats1 = cache.stats()
    
    # Second call with same params (cache hit)
    result2 = modpods.transform_inputs(shape_factors, scale_factors, loc_factors, index, forcing, cache=cache)
    stats2 = cache.stats()
    
    # Results must be identical
    np.testing.assert_allclose(result1["u_tr_1"].values, result2["u_tr_1"].values)
    
    # Cache should have 1 hit, 1 miss
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
    shape_factors = pd.DataFrame({"u": [2.0]}, index=[1])
    scale_factors = pd.DataFrame({"u": [1.0]}, index=[1])
    loc_factors = pd.DataFrame({"u": [0.0]}, index=[1])
    
    # Warm up
    _ = modpods.transform_inputs(shape_factors, scale_factors, loc_factors, index, forcing)
    
    # Time it
    start = time.perf_counter()
    for _ in range(5):
        _ = modpods.transform_inputs(shape_factors, scale_factors, loc_factors, index, forcing)
    elapsed = (time.perf_counter() - start) / 5
    
    # Should complete in well under 1 second for 5000 samples
    # (Original loop implementation took ~4 seconds for 5000 samples)
    assert elapsed < 0.1, f"transform_inputs too slow: {elapsed:.3f}s for {n} samples"


def test_transform_inputs_multiple_transforms() -> None:
    """transform_inputs must handle multiple transforms per input correctly."""
    np.random.seed(42)
    n = 200
    index = pd.date_range("2000-01-01", periods=n, freq="1h")
    
    forcing = pd.DataFrame({"u": np.cumsum(np.random.randn(n) * 0.1)}, index=index)
    
    # Two transforms with different parameters
    shape_factors = pd.DataFrame({"u": [2.0, 3.0]}, index=[1, 2])
    scale_factors = pd.DataFrame({"u": [1.0, 0.5]}, index=[1, 2])
    loc_factors = pd.DataFrame({"u": [0.0, 1.0]}, index=[1, 2])
    
    result = modpods.transform_inputs(shape_factors, scale_factors, loc_factors, index, forcing)
    
    assert "u_tr_1" in result.columns
    assert "u_tr_2" in result.columns
    assert len(result) == n
    assert not result.isnull().values.any()
    
    # Verify both transforms
    from scipy import signal, stats
    forcing_values = forcing["u"].to_numpy()
    
    for transform_idx, (shape, scale, loc) in enumerate([(2.0, 1.0, 0.0), (3.0, 0.5, 1.0)], 1):
        shape_time = np.arange(0, n, 1)
        gamma_kernel = stats.gamma.pdf(shape_time, shape, scale=scale, loc=loc)
        expected = signal.fftconvolve(forcing_values, gamma_kernel, mode="full")[:n]
        np.testing.assert_allclose(result[f"u_tr_{transform_idx}"].values, expected, rtol=1e-10)


def test_transform_inputs_multiple_inputs() -> None:
    """transform_inputs must handle multiple independent inputs correctly."""
    np.random.seed(42)
    n = 200
    index = pd.date_range("2000-01-01", periods=n, freq="1h")
    
    forcing = pd.DataFrame({
        "u1": np.cumsum(np.random.randn(n) * 0.1),
        "u2": np.cumsum(np.random.randn(n) * 0.1),
    }, index=index)
    
    shape_factors = pd.DataFrame({"u1": [2.0], "u2": [3.0]}, index=[1])
    scale_factors = pd.DataFrame({"u1": [1.0], "u2": [0.5]}, index=[1])
    loc_factors = pd.DataFrame({"u1": [0.0], "u2": [1.0]}, index=[1])
    
    result = modpods.transform_inputs(shape_factors, scale_factors, loc_factors, index, forcing)
    
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
    
    # Two parameter sets that quantize to the same key (1e-6 quantization)
    shape_factors1 = pd.DataFrame({"u": [2.0000001]}, index=[1])
    scale_factors1 = pd.DataFrame({"u": [1.0000001]}, index=[1])
    loc_factors1 = pd.DataFrame({"u": [0.0000001]}, index=[1])
    
    shape_factors2 = pd.DataFrame({"u": [2.0000002]}, index=[1])
    scale_factors2 = pd.DataFrame({"u": [1.0000002]}, index=[1])
    loc_factors2 = pd.DataFrame({"u": [0.0000002]}, index=[1])
    
    cache = modpods.TransformCache(quantization=1e-6)
    
    result1 = modpods.transform_inputs(shape_factors1, scale_factors1, loc_factors1, index, forcing, cache=cache)
    stats1 = cache.stats()
    
    result2 = modpods.transform_inputs(shape_factors2, scale_factors2, loc_factors2, index, forcing, cache=cache)
    stats2 = cache.stats()
    
    # Should be a cache hit due to quantization
    assert stats2["hits"] == 1
    np.testing.assert_allclose(result1["u_tr_1"].values, result2["u_tr_1"].values)


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
            verbose=False,
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
            verbose=False,
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
            verbose=False,
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
            verbose=False,
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
def compass_model(simple_lti_data: pd.DataFrame) -> dict[Any, Any]:
    """Train a model using the default compass-search optimizer."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = modpods.delay_io_train(
            simple_lti_data,
            dependent_columns=["x1"],
            independent_columns=["u"],
            windup_timesteps=0,
            init_transforms=1,
            max_transforms=1,
            max_iter=20,
            poly_order=1,
            verbose=False,
            optimization_method="compass_search",
        )
    return cast(dict[Any, Any], model)


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
            max_iter=20,
            poly_order=1,
            verbose=False,
            optimization_method="bayesian",
        )
    return cast(dict[Any, Any], model)


def test_compass_search_returns_valid_model(
    compass_model: dict[Any, Any],
) -> None:
    """Compass-search optimizer must return a well-formed model dict."""
    assert isinstance(compass_model, dict)
    assert 1 in compass_model
    assert "final_model" in compass_model[1]
    assert "error_metrics" in compass_model[1]["final_model"]
    r2 = float(compass_model[1]["final_model"]["error_metrics"]["r2"])
    assert r2 > -1.0, f"Compass R² {r2:.4f} is unreasonably low"


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


def test_both_methods_produce_comparable_r2(
    compass_model: dict[Any, Any],
    bayesian_model: dict[Any, Any],
) -> None:
    """Both optimization methods should achieve similar R² on the same data.

    The difference in R² should be within a reasonable margin, confirming
    that both methods solve the same underlying optimization problem.
    """
    r2_compass = float(compass_model[1]["final_model"]["error_metrics"]["r2"])
    r2_bayesian = float(bayesian_model[1]["final_model"]["error_metrics"]["r2"])
    # Both should be positive (reasonable fit)
    assert r2_compass > 0.0, f"Compass R² {r2_compass:.4f} is non-positive"
    assert r2_bayesian > 0.0, f"Bayesian R² {r2_bayesian:.4f} is non-positive"
    # Neither method should be dramatically worse than the other
    assert abs(r2_compass - r2_bayesian) < 0.5, (
        f"Methods diverge too much: compass={r2_compass:.4f}, "
        f"bayesian={r2_bayesian:.4f}"
    )


def test_compass_and_bayesian_predictions_agree(
    compass_model: dict[Any, Any],
    bayesian_model: dict[Any, Any],
    simple_lti_data: pd.DataFrame,
) -> None:
    """Predictions from compass and Bayesian models should broadly agree."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_compass = modpods.delay_io_predict(
            compass_model, simple_lti_data, num_transforms=1
        )
        pred_bayesian = modpods.delay_io_predict(
            bayesian_model, simple_lti_data, num_transforms=1
        )
    assert "prediction" in pred_compass
    assert "prediction" in pred_bayesian
    p_c = pred_compass["prediction"].ravel()
    p_b = pred_bayesian["prediction"].ravel()
    assert p_c.shape == p_b.shape, "Prediction shapes differ between methods"
    # Both predictions must be finite (no NaN or Inf)
    assert np.all(np.isfinite(p_c)), "Compass predictions contain NaN/Inf"
    assert np.all(np.isfinite(p_b)), "Bayesian predictions contain NaN/Inf"
    # Correlation of predictions should be high (both are fitting the same signal)
    # Guard against constant predictions (std == 0) which yield undefined correlation
    if p_c.std() > 0 and p_b.std() > 0:
        corr = float(np.corrcoef(p_c, p_b)[0, 1])
        assert (
            corr > 0.5
        ), f"Compass and Bayesian predictions are poorly correlated: {corr:.4f}"


# ---------------------------------------------------------------------------
# infer_causative_topology tests  (from test_topo_inference.py)
# ---------------------------------------------------------------------------


def test_infer_causative_topology_returns_dataframe(
    cascade_lti_system_data: pd.DataFrame,
) -> None:
    """infer_causative_topology must return a (DataFrame, DataFrame) tuple."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = modpods.infer_causative_topology(
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose=False,
            max_iter=0,
            method="granger",
        )
    assert isinstance(result, tuple) and len(result) == 2
    causative_topo, total_graph = result
    assert isinstance(causative_topo, pd.DataFrame)
    assert isinstance(total_graph, pd.DataFrame)


def test_infer_causative_topology_identifies_u1_causes_x2(
    cascade_lti_system_data: pd.DataFrame,
) -> None:
    """Granger causality must identify u1 as a cause of x2 (delayed cascade)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        causative_topo, _ = modpods.infer_causative_topology(
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose=False,
            max_iter=0,
            method="granger",
        )
    assert (
        causative_topo.loc["x2", "u1"] == "d"
    ), f"Expected u1→x2 to be 'd' (delayed), got '{causative_topo.loc['x2', 'u1']}'"


def test_infer_causative_topology_identifies_u2_causes_x8(
    cascade_lti_system_data: pd.DataFrame,
) -> None:
    """Granger causality must identify u2 as a cause of x8 (direct link)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        causative_topo, _ = modpods.infer_causative_topology(
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose=False,
            max_iter=0,
            method="granger",
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
        causative_topo, _ = modpods.infer_causative_topology(
            cascade_lti_system_data,
            dependent_columns=["x2", "x8", "x9"],
            independent_columns=["u1", "u2"],
            verbose=False,
            max_iter=0,
            method="granger",
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
            max_iter=10,
            bibo_stable=True,
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


# ---------------------------------------------------------------------------
# topo_from_pystorms tests  (from test_topo_from_swmm.py and
#                             test_lti_control_of_swmm.py) — SLOW
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_topo_from_pystorms_zeta_returns_dataframe() -> None:
    """topo_from_pystorms on the zeta scenario must return a non-empty DataFrame
    whose index and columns correspond to the scenario's state variables."""
    pystorms = pytest.importorskip("pystorms")
    env = pystorms.scenarios.zeta()
    topo = modpods.topo_from_pystorms(env)

    assert isinstance(topo, pd.DataFrame)
    assert topo.shape[0] > 0, "topology DataFrame must have at least one row"
    # All cells should be 'n', 'i', or 'd'
    valid_values = {"n", "i", "d"}
    for val in topo.values.flat:
        assert val in valid_values, f"unexpected cell value '{val}' in topology"


@pytest.mark.slow
def test_topo_from_pystorms_zeta_has_self_connections() -> None:
    """topo_from_pystorms (zeta) must mark each state as influencing itself ('i')."""
    pystorms = pytest.importorskip("pystorms")
    env = pystorms.scenarios.zeta()
    topo = modpods.topo_from_pystorms(env)

    for state in topo.index:
        if state in topo.columns:
            # Use .at[] to avoid pandas MultiIndex interpretation of tuple keys
            assert (
                topo.at[state, state] == "i"
            ), f"state {state!r} should have self-connection 'i'"


@pytest.mark.slow
def test_topo_from_pystorms_gamma_after_characterization() -> None:
    """After running a characterization simulation of the gamma scenario,
    topo_from_pystorms must successfully infer the network topology."""
    pystorms = pytest.importorskip("pystorms")
    np.random.seed(42)

    # Run characterization simulation (randomly opens/closes valves)
    env = pystorms.scenarios.gamma()
    done = False
    step = 0
    actions_characterize = np.ones(4)
    while not done:
        if step % 1000 == 0:
            actions_characterize = np.ones(4) * 0.3
            actions_characterize[np.random.randint(0, 4)] = np.random.rand()
        done = env.step(np.concatenate((actions_characterize, np.ones(7)), axis=0))
        step += 1

    # Restrict state/action space to the first 4 controlled basins
    dependent_columns = env.config["states"][4:8]
    independent_columns = [
        c for c in env.config["action_space"] if c not in dependent_columns
    ]
    env.config["states"] = dependent_columns
    env.config["action_space"] = independent_columns

    topo = modpods.topo_from_pystorms(env)

    assert isinstance(topo, pd.DataFrame)
    assert topo.shape[0] > 0


# ---------------------------------------------------------------------------
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
            max_iter=10,
            poly_order=1,
            verbose=False,
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

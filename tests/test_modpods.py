"""
Pytest tests for modpods core functions.

These tests cover the functions that are self-contained and do not require
external data files or long-running simulations.
"""

import warnings

import control as ct
import numpy as np
import pandas as pd
import pytest

import modpods

# ---------------------------------------------------------------------------
# lti_from_gamma
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
# infer_causative_topology (Granger causality)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cascade_lti_system_data() -> pd.DataFrame:
    """Generate response data from a known cascade LTI system.

    System topology (ground truth):
      u1 → x0 → x1 → x2  (u1 causes x2 via a long cascade, delayed)
      u2 → x8             (u2 causes x8 directly)
      x7 → x9, x8 → x9   (x9 driven by both chains; x7 is not observable)

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

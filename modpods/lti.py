import logging
from typing import Any

import control  # type: ignore
import numpy as np
import pandas as pd
import scipy.stats as stats

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from ._system_id import SystemIdModel, _n_polynomial_features
from ._validation import validate_columns, validate_system_data
from .kernels import get_kernel
from .model import _build_constraint_matrices
from .train import delay_io_train

logger = logging.getLogger(__name__)


def lti_from_gamma(
    shape,
    scale,
    location,
    dt=0,
    desired_NSE=0.999,
    verbose: Verbosity = "warnings",
    max_state_dim=50,
    max_iterations=200,
    max_pole_speed=5,
    min_pole_speed=0.01,
):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    t50 = shape * scale + location
    skewness = 2 / np.sqrt(shape)
    total_time_base = 2 * t50
    resolution = (t50) / (10 * (skewness + location))
    decay_rate = 1 / resolution
    decay_rate = np.clip(decay_rate, min_pole_speed, max_pole_speed)
    state_dim = max(1, min(int(np.ceil(shape * 2)), max_state_dim))
    decay_rate = state_dim / total_time_base
    resolution = 1 / decay_rate

    if _normalize_verbose(verbose) != "warnings":
        logger.info("state dimension is %s", state_dim)
        logger.info("decay rate is %s", decay_rate)
        logger.info("total time base is %s", total_time_base)
        logger.info("resolution is %s", resolution)

    t = np.linspace(0, 2 * total_time_base, num=200)
    gam = stats.gamma.pdf(t, shape, location, scale)

    A = decay_rate * np.diag(np.ones((state_dim - 1)), -1) - decay_rate * np.diag(
        np.ones((state_dim)), 0
    )
    B = np.concatenate((np.ones((1, 1)), np.zeros((state_dim - 1, 1))))
    C = np.ones((1, state_dim)) * max(gam)
    lti_sys = control.ss(A, B, C, 0)

    lti_approx = control.impulse_response(lti_sys, t)
    NSE = 1 - (
        np.sum(np.square(gam - lti_approx.y)) / np.sum(np.square(gam - np.mean(gam)))
    )
    if np.isnan(NSE):
        NSE = -10e6

    if _normalize_verbose(verbose) != "warnings":
        logger.info("initial NSE")
        logger.info("%s", NSE)
        logger.info("desired NSE")
        logger.info("%s", desired_NSE)

    iterations = 0

    speeds = [10, 5, 2, 1.1, 1.05, 1.01, 1.001]
    speed_idx = 0
    leap = speeds[speed_idx]

    while NSE < desired_NSE and iterations < max_iterations:

        og_was_best = True

        for i in range(C.shape[1] - 1, int(-1), int(-1)):

            og_approx = control.ss(A, B, C, 0)
            og_y = np.ndarray.flatten(control.impulse_response(og_approx, t).y)
            og_error = np.sum(np.abs(gam - og_y))
            og_NSE = 1 - (np.sum((gam - og_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2))

            Ctwice = np.array(C, copy=True)
            Ctwice[0, i] = leap * C[0, i]
            twice_approx = control.ss(A, B, Ctwice, 0)
            twice_y = np.ndarray.flatten(control.impulse_response(twice_approx, t).y)
            twice_NSE = 1 - (
                np.sum((gam - twice_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
            )

            Chalf = np.array(C, copy=True)
            Chalf[0, i] = (1 / leap) * C[0, i]
            half_approx = control.ss(A, B, Chalf, 0)
            half_y = np.ndarray.flatten(control.impulse_response(half_approx, t).y)
            half_NSE = 1 - (
                np.sum((gam - half_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
            )
            faster = np.array(A, copy=True)
            faster[i, i] = A[i, i] * leap
            if abs(faster[i, i]) < abs(max_pole_speed):
                if i > 0:
                    faster[i, i - 1] = A[i, i - 1] * leap
                faster_approx = control.ss(faster, B, C, 0)
                faster_y = np.ndarray.flatten(
                    control.impulse_response(faster_approx, t).y
                )
                faster_NSE = 1 - (
                    np.sum((gam - faster_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
                )
            else:
                faster_NSE = -10e6

            slower = np.array(A, copy=True)
            slower[i, i] = A[i, i] / leap
            if abs(slower[i, i]) > abs(min_pole_speed):
                if i > 0:
                    slower[i, i - 1] = A[i, i - 1] / leap
                slower_approx = control.ss(slower, B, C, 0)
                slower_y = np.ndarray.flatten(
                    control.impulse_response(slower_approx, t).y
                )
                slower_NSE = 1 - (
                    np.sum((gam - slower_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
                )
            else:
                slower_NSE = -10e6

            all_NSE = [
                og_NSE,
                twice_NSE,
                half_NSE,
                faster_NSE,
                slower_NSE,
            ]

            if twice_NSE >= max(all_NSE) and twice_NSE > og_NSE:
                C = Ctwice
                if twice_NSE > 1.001 * og_NSE:
                    og_was_best = False
            elif half_NSE >= max(all_NSE) and half_NSE > og_NSE:
                C = Chalf
                if half_NSE > 1.001 * og_NSE:
                    og_was_best = False

            elif slower_NSE >= max(all_NSE) and slower_NSE > og_NSE:
                A = slower
                if slower_NSE > 1.001 * og_NSE:
                    og_was_best = False
            elif faster_NSE >= max(all_NSE) and faster_NSE > og_NSE:
                A = faster
                if faster_NSE > 1.001 * og_NSE:
                    og_was_best = False

        NSE = og_NSE
        error = og_error
        iterations += 1
        if og_was_best:
            speed_idx += 1
            if speed_idx > len(speeds) - 1:
                break
            leap = speeds[speed_idx]
        if iterations % 2 == 0 and verbose != "warnings":
            logger.debug("iterations = %s", iterations)
            logger.debug("error = %s", error)
            logger.debug("NSE = %s", NSE)
            logger.debug("leap = %s", leap)

    lti_approx = control.ss(A, B, C, 0)
    y = np.ndarray.flatten(control.impulse_response(og_approx, t).y)
    error = np.sum(np.abs(gam - og_y))
    logger.info("LTI_from_gamma final NSE")
    logger.info("%s", NSE)
    if _normalize_verbose(verbose) != "warnings":
        logger.info("final system")
        logger.info("A")
        logger.info("%s", A)
        logger.info("B")
        logger.info("%s", B)
        logger.info("C")
        logger.info("%s", C)

        logger.info("final error")
        logger.info("%s", error)

    E = np.linalg.eigvals(A)
    if np.any(np.abs(E) > max_pole_speed) or np.any(np.abs(E) < min_pole_speed):
        logger.warning("final eigenvalues are outside the bounds specified")

    return {
        "lti_approx": lti_approx,
        "lti_approx_output": y,
        "error": error,
        "t": t,
        "gamma_pdf": gam,
    }


def lti_from_exponential_growth(rate, dt=0, desired_NSE=0.999, verbose="warnings"):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    A = np.array([[rate]])
    B = np.array([[1]])
    C = np.array([[1]])

    t = np.linspace(0, 10, num=200)
    target = np.exp(rate * t)
    target = target / np.sum(target)

    lti_sys = control.ss(A, B, C, 0)
    y = np.ndarray.flatten(control.impulse_response(lti_sys, t).y)
    y = y / np.sum(y)

    NSE = 1 - (
        np.sum(np.square(target - y)) / np.sum(np.square(target - np.mean(target)))
    )
    if np.isnan(NSE):
        NSE = -10e6

    error = np.sum(np.abs(target - y))

    if _normalize_verbose(verbose) != "warnings":
        logger.info("LTI_from_exponential_growth final NSE: %s", NSE)
        logger.info("A:\n%s", A)
        logger.info("B:\n%s", B)
        logger.info("C:\n%s", C)
        logger.info("final error: %s", error)

    return {
        "lti_approx": lti_sys,
        "lti_approx_output": y,
        "error": error,
        "t": t,
        "target": target,
    }


def lti_from_underdamped(zeta, omega_n, dt=0, desired_NSE=0.999, verbose="warnings"):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    omega_d = omega_n * np.sqrt(1.0 - zeta**2)

    A = np.array(
        [
            [0, 1],
            [-(omega_n**2), -2 * zeta * omega_n],
        ]
    )
    B = np.array([[0], [1]])
    C = np.array([[omega_n, 0]])

    if zeta < 0:
        t = np.linspace(0, 8 * np.pi / omega_d, num=200)
    else:
        t = np.linspace(0, 4 * np.pi / omega_d, num=200)
    target = (omega_n / omega_d) * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)
    if zeta >= 0:
        target = np.maximum(target, 0.0)

    lti_sys = control.ss(A, B, C, 0)
    y = np.ndarray.flatten(control.impulse_response(lti_sys, t).y)
    if zeta >= 0:
        y = np.maximum(y, 0.0)

    NSE = 1 - (
        np.sum(np.square(target - y)) / np.sum(np.square(target - np.mean(target)))
    )
    if np.isnan(NSE):
        NSE = -10e6

    error = np.sum(np.abs(target - y))

    if _normalize_verbose(verbose) != "warnings":
        logger.info("LTI_from_underdamped final NSE: %s", NSE)
        logger.info("A:\n%s", A)
        logger.info("B:\n%s", B)
        logger.info("C:\n%s", C)
        logger.info("final error: %s", error)

    return {
        "lti_approx": lti_sys,
        "lti_approx_output": y,
        "error": error,
        "t": t,
        "target": target,
    }


def lti_from_lognormal(mu, sigma, dt=0, desired_NSE=0.999, verbose="warnings"):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    t_end = 5 * np.exp(mu + 2 * sigma**2)
    t = np.linspace(0, t_end, num=200)
    target = stats.lognorm.pdf(t, sigma, scale=np.exp(mu))

    def _impulse_response(coeffs, t):
        a0, a1, a2, c0, c1, c2 = coeffs
        A = np.array([[0, 1, 0], [0, 0, 1], [-a0, -a1, -a2]])
        B = np.array([[0], [0], [1]])
        C = np.array([[c0, c1, c2]])
        sys = control.ss(A, B, C, 0)
        return np.ndarray.flatten(control.impulse_response(sys, t).y)

    omega_n = 1.0 / max(np.exp(mu), 1e-6)
    a0_init = omega_n**3
    a1_init = 3 * omega_n**2
    a2_init = 3 * omega_n
    target_max = np.max(target)
    c0_init = target_max * omega_n
    c1_init = 0.0
    c2_init = 0.0
    coeffs_init = np.array([a0_init, a1_init, a2_init, c0_init, c1_init, c2_init])

    def objective(coeffs):
        y = _impulse_response(coeffs, t)
        a0, a1, a2 = coeffs[:3]
        A = np.array([[0, 1, 0], [0, 0, 1], [-a0, -a1, -a2]])
        eigs = np.linalg.eigvals(A)
        stability_penalty = np.sum(np.maximum(np.real(eigs), 0.0) ** 2) * 1e6
        resid = target - y
        nse = 1.0 - np.sum(resid**2) / np.sum((target - np.mean(target)) ** 2)
        return -nse + stability_penalty

    from scipy.optimize import minimize

    bounds = [
        (1e-8, None),
        (1e-8, None),
        (1e-8, None),
        (1e-8, None),
        (None, None),
        (None, None),
    ]
    result = minimize(objective, coeffs_init, method="L-BFGS-B", bounds=bounds)
    a0, a1, a2, c0, c1, c2 = result.x
    A = np.array([[0, 1, 0], [0, 0, 1], [-a0, -a1, -a2]])
    B = np.array([[0], [0], [1]])
    C = np.array([[c0, c1, c2]])
    lti_sys = control.ss(A, B, C, 0)
    y = np.ndarray.flatten(control.impulse_response(lti_sys, t).y)
    y = np.maximum(y, 0.0)

    NSE = 1.0 - np.sum((target - y) ** 2) / np.sum((target - np.mean(target)) ** 2)
    if np.isnan(NSE):
        NSE = -10e6
    error = np.sum(np.abs(target - y))

    if _normalize_verbose(verbose) != "warnings":
        logger.info("LTI_from_lognormal final NSE: %s", NSE)
        logger.info("A:\n%s", A)
        logger.info("B:\n%s", B)
        logger.info("C:\n%s", C)
        logger.info("final error: %s", error)

    return {
        "lti_approx": lti_sys,
        "lti_approx_output": y,
        "error": error,
        "t": t,
        "target": target,
    }


def lti_from_bimodal_gamma(
    shape1,
    scale1,
    loc1,
    shape2,
    scale2,
    loc2,
    dt=0,
    desired_NSE=0.999,
    verbose="warnings",
):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    t_end = max(
        5 * (shape1 * scale1 + loc1 + 3 * scale1 * np.sqrt(shape1)),
        5 * (shape2 * scale2 + loc2 + 3 * scale2 * np.sqrt(shape2)),
    )
    t = np.linspace(0, t_end, num=300)
    target = 0.5 * stats.gamma.pdf(
        t, shape1, loc=loc1, scale=scale1
    ) + 0.5 * stats.gamma.pdf(t, shape2, loc=loc2, scale=scale2)

    result1 = lti_from_gamma(
        shape1,
        scale1,
        loc1,
        max_state_dim=max(3, int(np.ceil(shape1 * 2))),
        verbose=verbose,
    )
    result2 = lti_from_gamma(
        shape2,
        scale2,
        loc2,
        max_state_dim=max(3, int(np.ceil(shape2 * 2))),
        verbose=verbose,
    )

    sys1 = result1["lti_approx"]
    sys2 = result2["lti_approx"]
    n1 = sys1.A.shape[0]
    n2 = sys2.A.shape[0]
    A_combined = np.block([[sys1.A, np.zeros((n1, n2))], [np.zeros((n2, n1)), sys2.A]])
    B_combined = np.block([[sys1.B], [sys2.B]])
    C_combined = np.hstack([0.5 * sys1.C, 0.5 * sys2.C])
    lti_sys = control.ss(A_combined, B_combined, C_combined, 0)
    y = np.ndarray.flatten(control.impulse_response(lti_sys, t).y)
    y = np.maximum(y, 0.0)

    NSE = 1.0 - np.sum((target - y) ** 2) / np.sum((target - np.mean(target)) ** 2)
    if np.isnan(NSE):
        NSE = -10e6
    error = np.sum(np.abs(target - y))

    if _normalize_verbose(verbose) != "warnings":
        logger.info("LTI_from_bimodal_gamma final NSE: %s", NSE)
        logger.info("A:\n%s", A_combined)
        logger.info("B:\n%s", B_combined)
        logger.info("C:\n%s", C_combined)
        logger.info("final error: %s", error)
        logger.info("states from component 1: %s", n1)
        logger.info("states from component 2: %s", n2)

    return {
        "lti_approx": lti_sys,
        "lti_approx_output": y,
        "error": error,
        "t": t,
        "target": target,
    }


def lti_from_kernel(
    kernel,
    params,
    dt=0,
    desired_NSE=0.999,
    verbose="warnings",
    max_state_dim=50,
    max_iterations=200,
    max_pole_speed=5,
    min_pole_speed=0.01,
):
    if isinstance(kernel, str):
        kernel = get_kernel(kernel)

    if kernel.name == "gamma":
        shape = params["shape"]
        scale = params["scale"]
        loc = params["loc"]
        return lti_from_gamma(
            shape,
            scale,
            loc,
            dt=dt,
            desired_NSE=desired_NSE,
            verbose=verbose,
            max_state_dim=max_state_dim,
            max_iterations=max_iterations,
            max_pole_speed=max_pole_speed,
            min_pole_speed=min_pole_speed,
        )

    if kernel.name == "underdamped":
        zeta = params["zeta"]
        omega_n = params["omega_n"]
        return lti_from_underdamped(
            zeta,
            omega_n,
            dt=dt,
            desired_NSE=desired_NSE,
            verbose=verbose,
        )

    if kernel.name == "lognormal":
        mu = params["mu"]
        sigma = params["sigma"]
        return lti_from_lognormal(
            mu,
            sigma,
            dt=dt,
            desired_NSE=desired_NSE,
            verbose=verbose,
        )

    if kernel.name == "bimodal_gamma":
        shape1 = params["shape1"]
        scale1 = params["scale1"]
        loc1 = params["loc1"]
        shape2 = params["shape2"]
        scale2 = params["scale2"]
        loc2 = params["loc2"]
        return lti_from_bimodal_gamma(
            shape1,
            scale1,
            loc1,
            shape2,
            scale2,
            loc2,
            dt=dt,
            desired_NSE=desired_NSE,
            verbose=verbose,
        )

    if kernel.name == "exponential_growth":
        rate = params["rate"]
        return lti_from_exponential_growth(
            rate,
            dt=dt,
            desired_NSE=desired_NSE,
            verbose=verbose,
        )

    raise ValueError(f"Unsupported kernel: {kernel.name}")


# this function takes the system data and the causative topology and returns an LTI system
# if the causative topology isn't already defined, it needs to be created using infer_causative_topology
def lti_system_gen(
    causative_topology,
    system_data,
    independent_columns,
    dependent_columns,
    max_iter=250,
    swmm=False,
    bibo_stable=False,
    max_transition_state_dim=50,
    max_transforms=1,
    early_stopping_threshold=0.005,
    verbose: Verbosity = "warnings",
    forcing_coef_constraints=None,
    constraints=None,
    kernel="gamma",
):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    # cast the columns and indices of causative_topology to strings so the regression model can run properly
    if swmm:
        causative_topology.columns = causative_topology.columns.astype(str)
        causative_topology.index = causative_topology.index.astype(str)

        logger.info("causative topology")
        logger.info("%s", causative_topology.index)
        logger.info("%s", causative_topology.columns)

        dependent_columns = [str(col) for col in dependent_columns]
        independent_columns = [str(col) for col in independent_columns]
        logger.info("%s", dependent_columns)
        logger.info("%s", independent_columns)

        system_data.columns = system_data.columns.astype(str)
        logger.info("%s", system_data.columns)

    A = pd.DataFrame(index=dependent_columns, columns=dependent_columns)
    B = pd.DataFrame(index=dependent_columns, columns=independent_columns)
    C = pd.DataFrame(index=dependent_columns, columns=dependent_columns)
    C.loc[:, :] = np.diag(
        np.ones(len(dependent_columns))
    )  # these are the states which are observable

    # copy the corresponding entries from the causative topology into B
    for row in B.index:
        for col in B.columns:
            B.loc[row, col] = causative_topology.loc[row, col]
    # and into A
    for row in A.index:
        for col in A.columns:
            A.loc[row, col] = causative_topology.loc[row, col]

    logger.info("A")
    logger.info("%s", A)
    logger.info("B")
    logger.info("%s", B)
    logger.info("C")
    logger.info("%s", C)

    # use transform_only when calling delay_io_train to only train transfomrations for connections marked "d"
    # train a MISO model for each output
    delay_models: dict = {key: None for key in dependent_columns}

    for row in A.index:
        immediate_forcing = []
        delayed_forcing = []
        for col in A.columns:
            if col == row:
                continue
            if A[col][row] == "d":
                delayed_forcing.append(col)
            elif A[col][row] == "i":
                immediate_forcing.append(col)
        for col in B.columns:
            if B[col][row] == "d":
                delayed_forcing.append(col)
            elif B[col][row] == "i":
                immediate_forcing.append(col)
        # make total_forcing the union of immediate and delayed forcing
        total_forcing = immediate_forcing + delayed_forcing
        feature_names = [row] + total_forcing
        if delayed_forcing:
            logger.info(
                "training delayed model for %s with forcing %s",
                row,
                total_forcing,
            )
            delay_models[row] = delay_io_train(
                system_data,
                [row],
                total_forcing,
                transform_only=delayed_forcing,
                max_transforms=max_transforms,
                poly_order=1,
                max_iter=max_iter,
                verbose=verbose,
                bibo_stable=bibo_stable,
                forcing_coef_constraints=forcing_coef_constraints,
                constraints=constraints,
            )
        else:
            logger.info(
                "training immediate model for %s with forcing %s",
                row,
                total_forcing,
            )
            delay_models[row] = None

            if bibo_stable:
                n_features = _n_polynomial_features(len(feature_names), 1, False, False)

                constraint_lhs = np.zeros((1, n_features))
                constraint_rhs = np.zeros(1)

                for i, col in enumerate(feature_names):
                    if col == row:
                        constraint_lhs[0, i] = 1

                custom_lhs, custom_rhs, custom_inequality = _build_constraint_matrices(
                    feature_names, forcing_coef_constraints, constraints, n_targets=1
                )
                if custom_lhs.shape[0] > 0:
                    constraint_lhs = np.vstack([constraint_lhs, custom_lhs])
                    constraint_rhs = np.concatenate([constraint_rhs, custom_rhs])
                    all_inequality = custom_inequality
                else:
                    all_inequality = True

                model = SystemIdModel(
                    poly_degree=1,
                    include_bias=False,
                    include_interaction=False,
                    constraint_lhs=constraint_lhs,
                    constraint_rhs=constraint_rhs,
                    inequality_constraints=all_inequality,
                )

            else:
                model = SystemIdModel(
                    poly_degree=1,
                    include_bias=False,
                    include_interaction=False,
                    fd_order=10,
                    fd_drop_endpoints=True,
                )
            if system_data.loc[:, immediate_forcing].empty:
                instant_fit = model.fit(
                    x=system_data.loc[:, row],
                    t=np.arange(0, len(system_data.index), 1),
                    feature_names=feature_names,
                )
                instant_fit.print(precision=3)
                logger.info(
                    "Training r2 = %s",
                    instant_fit.score(
                        x=system_data.loc[:, row],
                        t=np.arange(0, len(system_data.index), 1),
                    ),
                )
                logger.info("%s", instant_fit.coefficients())
            else:
                instant_fit = model.fit(
                    x=system_data.loc[:, row],
                    t=np.arange(0, len(system_data.index), 1),
                    u=system_data.loc[:, immediate_forcing],
                    feature_names=feature_names,
                )
                instant_fit.print(precision=3)
                logger.info(
                    "Training r2 = %s",
                    instant_fit.score(
                        x=system_data.loc[:, row],
                        t=np.arange(0, len(system_data.index), 1),
                        u=system_data.loc[:, immediate_forcing],
                    ),
                )
                logger.info("%s", instant_fit.coefficients())
            for idx in range(len(feature_names)):
                if feature_names[idx] in A.columns:
                    A.loc[row, feature_names[idx]] = instant_fit.coefficients()[0][idx]
                elif feature_names[idx] in B.columns:
                    B.loc[row, feature_names[idx]] = instant_fit.coefficients()[0][idx]
                else:
                    logger.warning("couldn't find a column for %s", feature_names[idx])

    original_A = A.copy(deep=True)

    # Convert 'n', 'd', 'i' to numeric before parsing delay models
    A.replace({"n": 0.0, "d": 0.0, "i": 0.0}, inplace=True)
    B.replace({"n": 0.0, "d": 0.0, "i": 0.0}, inplace=True)
    C.replace({"n": 0.0, "d": 0.0, "i": 0.0}, inplace=True)
    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    B = B.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    C = C.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # now, parse the delay models into the A, B, and C matrices
    for row in original_A.index:
        if delay_models[row] is None:
            pass
        else:
            for num_transforms in range(1, max_transforms + 1):
                if num_transforms == 1:
                    optimal_number_transforms = num_transforms
                elif (
                    delay_models[row][num_transforms]["final_model"]["error_metrics"][
                        "r2"
                    ]
                    - delay_models[row][num_transforms - 1]["final_model"][
                        "error_metrics"
                    ]["r2"]
                    < early_stopping_threshold
                ):
                    optimal_number_transforms = num_transforms - 1
                    break
                else:
                    optimal_number_transforms = num_transforms

            transformation_approximations: dict[str, Any] = {
                transform_key: {}
                for transform_key in delay_models[row][optimal_number_transforms][
                    "kernel_params"
                ].columns
            }
            row_kernel_type = delay_models[row][optimal_number_transforms].get(
                "kernel_type", "gamma"
            )
            for transform_key in transformation_approximations.keys():
                for idx in range(1, optimal_number_transforms + 1):
                    logger.info(
                        "variable = %s, transformation = %s", transform_key, idx
                    )
                    delay_models[row][optimal_number_transforms]["final_model"][
                        "model"
                    ].print(precision=5)
                    kernel_params = delay_models[row][optimal_number_transforms][
                        "kernel_params"
                    ]
                    transformation_approximations[transform_key] = lti_from_kernel(
                        row_kernel_type,
                        kernel_params.loc[idx, transform_key].to_dict(),
                        max_state_dim=max_transition_state_dim,
                        verbose=verbose,
                    )

                    lti_result = transformation_approximations[transform_key]
                    Agam = lti_result["lti_approx"].A
                    Bgam = lti_result[
                        "lti_approx"
                    ].B  # only entry is unit impulse at top state
                    Cgam = lti_result["lti_approx"].C

                    tr_string = str("_tr_" + str(idx))

                    # Cgam needs to be scaled by the coefficient the forcing term had in the delay model
                    coefficients = {
                        coef_key: None
                        for coef_key in delay_models[row][optimal_number_transforms][
                            "final_model"
                        ]["model"].feature_names
                    }
                    for coef_key in coefficients.keys():
                        coef_index = delay_models[row][optimal_number_transforms][
                            "final_model"
                        ]["model"].feature_names.index(coef_key)
                        coefficients[coef_key] = delay_models[row][
                            optimal_number_transforms
                        ]["final_model"]["model"].coefficients()[0][coef_index]
                        if tr_string in coef_key and coef_key.replace(
                            tr_string, ""
                        ) == transform_key.replace(tr_string, ""):
                            Cgam = Cgam * coefficients[coef_key]  # scaling
                        else:  # these are the immediate effects, insert them now
                            if coef_key in A.columns:
                                A.loc[row, coef_key] = coefficients[coef_key]
                            elif coef_key in B.columns:
                                B.loc[row, coef_key] = coefficients[coef_key]

                    # Build Agam_index - use source->dest naming for unique state names
                    source_var = transform_key.replace(tr_string, "")
                    Agam_index = []
                    for agam_idx in range(Agam.shape[0]):
                        Agam_index.append(f"{source_var}->{row}{tr_string}_{agam_idx}")

                    Agam = pd.DataFrame(Agam, index=Agam_index, columns=Agam_index)
                    Bgam = pd.DataFrame(
                        Bgam,
                        index=Agam_index,
                        columns=[source_var],
                    )
                    Cgam = pd.DataFrame(Cgam, index=[row], columns=Agam_index)
                    # insert these into the A, B, and C matrices
                    # for Agam, the insertion row is immediately after the source (key)
                    # the insertion column is also immediately after the source (key)

                    # Build new state list by inserting Agam states after the source variable
                    if source_var in A.index:
                        # Source is a state variable - insert after it
                        source_loc = A.index.get_loc(source_var)
                        if isinstance(source_loc, slice):
                            source_loc = (
                                source_loc.stop - 1
                                if source_loc.stop
                                else len(A.index) - 1
                            )
                        before_states = list(A.index[: source_loc + 1])
                        after_states = list(A.index[source_loc + 1 :])
                        new_states = before_states + Agam_index + after_states
                    elif source_var in B.columns:
                        # Source is an input - insert at beginning of state vector
                        new_states = Agam_index + list(A.index)
                    else:
                        logger.warning(
                            "Source variable %s not found in A or B", source_var
                        )
                        new_states = list(A.index) + Agam_index

                    newA = pd.DataFrame(
                        0.0, index=new_states, columns=new_states, dtype=float
                    )
                    newB = pd.DataFrame(
                        0.0, index=new_states, columns=B.columns, dtype=float
                    )
                    newC = pd.DataFrame(
                        0.0, index=C.index, columns=new_states, dtype=float
                    )

                    # Copy existing A entries
                    for idx in newA.index:
                        for col in newA.columns:
                            if idx in A.index and col in A.columns:
                                newA.loc[idx, col] = A.loc[idx, col]

                    # Insert Agam block (state-to-state dynamics of the delay subsystem)
                    for idx in Agam.index:
                        for col in Agam.columns:
                            if idx in newA.index and col in newA.columns:
                                newA.loc[idx, col] = Agam.loc[idx, col]

                    # Insert Bgam - connects source input to first state of cascade
                    # Bgam has shape (n_states, 1) with 1 at top state
                    # If source_var is a state: goes in A at [delay_state, source_var]
                    # If source_var is an input: goes in B at [delay_state, source_var]
                    for idx in Bgam.index:
                        for col in Bgam.columns:
                            if idx in newA.index:
                                if col in newA.columns:
                                    # source_var is a state variable
                                    newA.loc[idx, col] = Bgam.loc[idx, col]
                                elif col in newB.columns:
                                    # source_var is an input
                                    newB.loc[idx, col] = Bgam.loc[idx, col]

                    # Copy existing B entries
                    for idx in newB.index:
                        for col in newB.columns:
                            if idx in B.index and col in B.columns:
                                newB.loc[idx, col] = B.loc[idx, col]

                    # Copy existing C entries
                    for idx in newC.index:
                        for col in newC.columns:
                            if idx in C.index and col in C.columns:
                                newC.loc[idx, col] = C.loc[idx, col]

                    # Insert Cgam - connects delay states to dependent variable (row)
                    # row is a state variable, so this goes in A at [row, delay_state]
                    for idx in Cgam.index:
                        for col in Cgam.columns:
                            if idx in newA.index and col in newA.columns:
                                newA.loc[idx, col] = Cgam.loc[idx, col]

                    A = newA.copy()
                    B = newB.copy()
                    C = newC.copy()

    if swmm:
        pass

    # if bibo_stable is specified and A not Hurwitz, make A Hurwitz by
    # subtracting I * shift from A so that max(real(eig(A))) < 0
    if bibo_stable:
        try:
            orig_eigs = np.linalg.eigvals(A.values)
            max_real_eig = float(np.max(np.real(orig_eigs)))
            if max_real_eig >= -1e-12:
                logger.warning(
                    "stabilizing unstable or marginally stable plant by shifting A (max real eig = %.6f)",
                    max_real_eig,
                )
                epsilon = 1e-3
                shift = max((1 + epsilon) * max_real_eig + epsilon, epsilon)
                A_stab = A - np.eye(len(A)) * shift
                A = A_stab.copy()
                # Verify
                new_eigs = np.linalg.eigvals(A.values)
                new_max_real = float(np.max(np.real(new_eigs)))
                logger.info("After stabilization: max real eig = %.6f", new_max_real)
        except Exception as e:
            logger.warning("Failed to stabilize A matrix: %s", e)

    # the regression model will scale the coefficients according to the timestep if the index is numeric
    try:
        pd.to_numeric(system_data.index, errors="raise")
        dt = system_data.index.values[1] - system_data.index.values[0]
        A = A / dt
        B = B / dt
        logger.info("system response data index converted to numeric type. dt = %s", dt)
    except Exception as e:
        logger.warning("%s", e)
        dt = None

    # cast all of A, B, and C to type float
    A = A.astype(float)
    B = B.astype(float)
    C = C.astype(float)

    lti_sys = control.ss(
        A, B, C, 0, inputs=B.columns, outputs=C.index, states=A.columns
    )

    return {"system": lti_sys, "A": A, "B": B, "C": C}


class LTISystem:
    """LTI system estimator following scikit-learn conventions."""

    def __init__(
        self,
        causative_topology: pd.DataFrame,
        independent_columns: list[str],
        dependent_columns: list[str],
        max_iter: int = 250,
        bibo_stable: bool = False,
        max_transition_state_dim: int = 50,
        max_transforms: int = 1,
        early_stopping_threshold: float = 0.005,
        verbose: Verbosity = "warnings",
        forcing_coef_constraints: Any = None,
        constraints: Any = None,
        kernel: str = "gamma",
    ) -> None:
        self.causative_topology = causative_topology
        self.independent_columns = independent_columns
        self.dependent_columns = dependent_columns
        self.max_iter = max_iter
        self.bibo_stable = bibo_stable
        self.max_transition_state_dim = max_transition_state_dim
        self.max_transforms = max_transforms
        self.early_stopping_threshold = early_stopping_threshold
        self.verbose = verbose
        self.forcing_coef_constraints = forcing_coef_constraints
        self.constraints = constraints
        self.kernel = kernel
        self.system_: Any = None
        self.A_: pd.DataFrame | None = None
        self.B_: pd.DataFrame | None = None
        self.C_: pd.DataFrame | None = None

    def fit(self, system_data: pd.DataFrame, **kwargs: Any) -> "LTISystem":
        validate_system_data(system_data)
        validate_columns(system_data, self.dependent_columns, "dependent_columns")
        validate_columns(system_data, self.independent_columns, "independent_columns")

        result = lti_system_gen(
            causative_topology=self.causative_topology,
            system_data=system_data,
            independent_columns=self.independent_columns,
            dependent_columns=self.dependent_columns,
            max_iter=self.max_iter,
            bibo_stable=self.bibo_stable,
            max_transition_state_dim=self.max_transition_state_dim,
            max_transforms=self.max_transforms,
            early_stopping_threshold=self.early_stopping_threshold,
            verbose=self.verbose,
            forcing_coef_constraints=self.forcing_coef_constraints,
            constraints=self.constraints,
            kernel=self.kernel,
            **kwargs,
        )
        self.system_ = result["system"]
        self.A_ = result["A"]
        self.B_ = result["B"]
        self.C_ = result["C"]
        return self

    def predict(
        self,
        system_data: pd.DataFrame,
        u_new: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> Any:
        import control as ct  # type: ignore

        if self.system_ is None:
            raise RuntimeError("Estimator has not fitted yet.")
        if u_new is None:
            return self.system_
        t = np.arange(len(u_new))
        u_array = u_new.values.T if u_new.ndim > 1 else u_new.values.flatten()
        yout, tout, xout = ct.forced_response(self.system_, T=t, U=u_array)
        return {"yout": yout, "tout": tout, "xout": xout}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "causative_topology": self.causative_topology,
            "independent_columns": self.independent_columns,
            "dependent_columns": self.dependent_columns,
            "max_iter": self.max_iter,
            "bibo_stable": self.bibo_stable,
            "max_transition_state_dim": self.max_transition_state_dim,
            "max_transforms": self.max_transforms,
            "early_stopping_threshold": self.early_stopping_threshold,
            "verbose": self.verbose,
            "forcing_coef_constraints": self.forcing_coef_constraints,
            "constraints": self.constraints,
            "kernel": self.kernel,
        }

    def set_params(self, **params: Any) -> "LTISystem":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"LTISystem(dependent_columns={self.dependent_columns}, "
            f"independent_columns={self.independent_columns}, "
            f"max_iter={self.max_iter}, bibo_stable={self.bibo_stable}, "
            f"kernel={self.kernel!r})"
        )

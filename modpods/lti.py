import logging
from typing import Any, cast

import control  # type: ignore
import numpy as np
import pandas as pd
import pysindy as ps  # type: ignore
import scipy.stats as stats
from pysindy.optimizers._constrained_sr3 import (  # type: ignore[import-untyped]
    ConstrainedSR3 as _ConstrainedSR3,
)

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from .kernels import ConvolutionKernel, get_kernel
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

    # a pole of speed -5 decays to less than 1% of it's value after one timestep
    # a pole of speed -0.01 decays to more than 99% of it's value after one timestep
    t50 = shape * scale + location  # center of mass
    skewness = 2 / np.sqrt(shape)
    total_time_base = (
        2 * t50
    )  # not that this contains the full shape, but if we fit this much of the curve perfectly we'll be close enough
    # resolution = (t50)/((skewness + location)) # make this coarser for faster debugging
    resolution = (t50) / (10 * (skewness + location))  # production version

    # resolution = 1/ skewness
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

    # make the timestep one so that the relative error is correct (dt too small makes error bigger than written)
    # t = np.linspace(0,3*total_time_base,1000)
    # desired_error = desired_error / dt
    t = np.linspace(0, 2 * total_time_base, num=200)

    # if verbose:
    #    print("dt is ",dt)
    #    print("scaled desired error is ",desired_error)

    gam = stats.gamma.pdf(t, shape, location, scale)

    # A is a cascade with the appropriate decay rate
    A = decay_rate * np.diag(np.ones((state_dim - 1)), -1) - decay_rate * np.diag(
        np.ones((state_dim)), 0
    )
    # influence enters at the top state only
    B = np.concatenate((np.ones((1, 1)), np.zeros((state_dim - 1, 1))))
    # contributions of states to the output will be scaled to match the gamma distribution
    C = np.ones((1, state_dim)) * max(gam)
    lti_sys = control.ss(A, B, C, 0)

    lti_approx = control.impulse_response(lti_sys, t)
    NSE = 1 - (
        np.sum(np.square(gam - lti_approx.y)) / np.sum(np.square(gam - np.mean(gam)))
    )
    # if NSE is nan, set to -10e6
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
    # the area under the curve is normalized to be one. so rather than basing our desired error off the
    # max of the distribution, it might be better to make it a percentage error, one percent or five percent
    while NSE < desired_NSE and iterations < max_iterations:

        og_was_best = (
            True  # start each iteration assuming that the original is the best
        )
        # search across the C vector
        for i in range(
            C.shape[1] - 1, int(-1), int(-1)
        ):  # across the columns # start at the end and come back
            # for i in range(int(0),C.shape[1],int(1)): # across the columns, start at the beginning and go forward

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
            faster[i, i] = A[i, i] * leap  # faster decay
            if abs(faster[i, i]) < abs(max_pole_speed):
                if (
                    i > 0
                ):  # first reservoir doesn't receive contribution from another reservoir. want to keep B at 1 for scaling
                    faster[i, i - 1] = A[i, i - 1] * leap  # faster rise
                faster_approx = control.ss(faster, B, C, 0)
                faster_y = np.ndarray.flatten(
                    control.impulse_response(faster_approx, t).y
                )
                faster_NSE = 1 - (
                    np.sum((gam - faster_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
                )
            else:
                faster_NSE = -10e6  # disallowed because the pole is too fast

            slower = np.array(A, copy=True)
            slower[i, i] = A[i, i] / leap  # slower decay
            if abs(slower[i, i]) > abs(min_pole_speed):
                if i > 0:
                    slower[i, i - 1] = A[i, i - 1] / leap  # slower rise
                slower_approx = control.ss(slower, B, C, 0)
                slower_y = np.ndarray.flatten(
                    control.impulse_response(slower_approx, t).y
                )
                slower_NSE = 1 - (
                    np.sum((gam - slower_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
                )
            else:
                slower_NSE = -10e6  # disallowed because the pole is too slow

            # all_errors = [og_error, twice_error, half_error, faster_error, slower_error]
            all_NSE = [
                og_NSE,
                twice_NSE,
                half_NSE,
                faster_NSE,
                slower_NSE,
            ]

            if twice_NSE >= max(all_NSE) and twice_NSE > og_NSE:
                C = Ctwice
                if twice_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?
            elif half_NSE >= max(all_NSE) and half_NSE > og_NSE:
                C = Chalf
                if half_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?

            elif slower_NSE >= max(all_NSE) and slower_NSE > og_NSE:
                A = slower
                if slower_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?
            elif faster_NSE >= max(all_NSE) and faster_NSE > og_NSE:
                A = faster
                if faster_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?

        NSE = og_NSE
        error = og_error
        iterations += 1  # this shouldn't be the termination condition unless the resolution is too coarse
        # normally the optimization should exit because the leap has become too small
        if (
            og_was_best
        ):  # the original was the best, so we're going to tighten up the optimization
            speed_idx += 1
            if speed_idx > len(speeds) - 1:
                break  # we're done
            leap = speeds[speed_idx]
        # print the iteration count every ten
        # comment out for production
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

    # are any of the final eigenvalues outside the bounds specified?
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


def lti_from_underdamped(zeta, omega_n, dt=0, desired_NSE=0.999, verbose="warnings"):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    omega_d = omega_n * np.sqrt(1.0 - zeta**2)

    state_dim = 2
    A = np.array(
        [
            [0, 1],
            [-(omega_n**2), -2 * zeta * omega_n],
        ]
    )
    B = np.array([[0], [1]])
    C = np.array([[omega_n, 0]])

    t = np.linspace(0, 4 * np.pi / omega_d, num=200)
    target = (omega_n / omega_d) * np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)
    target = np.maximum(target, 0.0)

    lti_sys = control.ss(A, B, C, 0)
    y = np.ndarray.flatten(control.impulse_response(lti_sys, t).y)
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

    max_state_dim = 50
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

    # cast the columns and indices of causative_topology to strings so sindy can run properly
    # We need the tuples to link the columns in system_data to the object names in the swmm model
    # so we'll cast these back to tuples once we're done
    if swmm:
        causative_topology.columns = causative_topology.columns.astype(str)
        causative_topology.index = causative_topology.index.astype(str)

        logger.info("causative topology")
        logger.info("%s", causative_topology.index)
        logger.info("%s", causative_topology.columns)

        # do the same for dependent_columns and independent_columns
        dependent_columns = [str(col) for col in dependent_columns]
        independent_columns = [str(col) for col in independent_columns]
        logger.info("%s", dependent_columns)
        logger.info("%s", independent_columns)

        # do the same for the columns of system_data
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
                continue  # don't need to include the output state as a forcing variable. it's already included by default
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
            # we'll parse this delayed causation into the matrices A, B, and C later
        else:
            logger.info(
                "training immediate model for %s with forcing %s",
                row,
                total_forcing,
            )
            delay_models[row] = None
            # we can put immediate causation into the matrices A, B, and C now

            if bibo_stable:  # negative autocorrelatoin
                library = ps.PolynomialLibrary(
                    degree=1, include_bias=False, include_interaction=False
                )
                library.fit(np.zeros((2, len(feature_names))))
                n_features = library.n_output_features_

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

                model = ps.SINDy(
                    differentiation_method=ps.FiniteDifference(),
                    feature_library=ps.PolynomialLibrary(
                        degree=1, include_bias=False, include_interaction=False
                    ),
                    optimizer=_ConstrainedSR3(
                        reg_weight_lam=0,
                        regularizer="l2",
                        constraint_lhs=constraint_lhs,
                        constraint_rhs=constraint_rhs,
                        inequality_constraints=all_inequality,
                    ),
                )

            else:  # unoconstrained
                model = ps.SINDy(
                    differentiation_method=ps.FiniteDifference(
                        order=10, drop_endpoints=True
                    ),
                    feature_library=ps.PolynomialLibrary(
                        degree=1, include_bias=False, include_interaction=False
                    ),
                    optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
                )
            if system_data.loc[
                :, immediate_forcing
            ].empty:  # the subsystem is autonomous
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
            else:  # there is some forcing
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
    # now, parse the delay models into the A, B, and C matrices
    for row in original_A.index:
        if delay_models[row] is None:
            pass
        else:  # we want the model with the most transformations where the last transformation added at least 0.5% to the R2 score
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
                    break  # improvement is too small to justify additional complexity
                else:
                    optimal_number_transforms = (
                        num_transforms  # the most recent one was worth it
                    )

            transformation_approximations: dict[str, Any] = {
                transform_key: {}
                for transform_key in delay_models[row][optimal_number_transforms][
                    "kernel_params"
                ].columns
            }
            row_kernel_type = delay_models[row][optimal_number_transforms].get(
                "kernel_type", "gamma"
            )
            for transform_key in transformation_approximations.keys():  # which input
                for idx in range(
                    1, optimal_number_transforms + 1
                ):  # which transformation
                    logger.info(
                        "variable = %s, transformation = %s", transform_key, idx
                    )
                    delay_models[row][optimal_number_transforms]["final_model"][
                        "model"
                    ].print(precision=5)
                    kernel_params = delay_models[row][optimal_number_transforms][
                        "kernel_params"
                    ]
                    shape = kernel_params.loc[(idx, "shape"), transform_key]
                    scale = kernel_params.loc[(idx, "scale"), transform_key]
                    loc = kernel_params.loc[(idx, "loc"), transform_key]
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

                    Agam_index = []
                    for agam_idx in range(Agam.shape[0]):
                        Agam_index.append(
                            transform_key.replace(tr_string, "")
                            + "->"
                            + row
                            + tr_string
                            + "_"
                            + str(agam_idx)
                        )
                    Agam = pd.DataFrame(Agam, index=Agam_index, columns=Agam_index)
                    Bgam = pd.DataFrame(
                        Bgam,
                        index=Agam_index,
                        columns=[transform_key.replace(tr_string, "")],
                    )
                    Cgam = pd.DataFrame(Cgam, index=[row], columns=Agam_index)
                    # insert these into the A, B, and C matrices
                    # for Agam, the insertion row is immediately after the source (key)
                    # the insertion column is also immediately after the source (key)

                    before_index = []
                    if (
                        transform_key.replace(tr_string, "") not in A.index
                    ):  # it's one of the forcing terms. put it in at the beginning
                        after_index = list(
                            A.index
                        )  # it's a forcing variable, so we don't want it in the newA index
                    else:  # it is a state variable
                        before_index = list(
                            A.index[
                                : A.index.get_loc(transform_key.replace(tr_string, ""))
                            ]
                        )

                        after_index = list(
                            A.index[
                                cast(
                                    int,
                                    A.index.get_loc(
                                        transform_key.replace(tr_string, "")
                                    ),
                                )
                                + 1 :
                            ]
                        )

                    # if transform_key.replace("_tr_1","") in A.index: # the transform key refers to a state (x)
                    if transform_key.replace(tr_string, "") in A.index:
                        # states = before_index + [transform_key.replace("_tr_1","")] + Agam_index + after_index # state dim expands by the number of rows in Agam
                        states = (
                            before_index
                            + [transform_key.replace(tr_string, "")]
                            + Agam_index
                            + after_index
                        )  # state dim expands by the number of rows in Agam
                        # include the current transform key in A because it's a state variable
                    # elif transform_key.replace("_tr_1","") in B.columns: # the transform key refers to a control input (u)
                    elif (
                        transform_key.replace(tr_string, "") in B.columns
                    ):  # the transform key refers to a control input (u)
                        states = (
                            before_index + Agam_index + after_index
                        )  # state dim expands by the number of rows in Agam
                        # don't include the current transform key in A because it's a control input, not a state variable

                    newA = pd.DataFrame(index=states, columns=states)
                    newB = pd.DataFrame(
                        index=states, columns=B.columns
                    )  # input dim remains consistent (columns of B)
                    newC = pd.DataFrame(
                        index=C.index, columns=states
                    )  # output dim remains consistent (rows of C)

                    # fill in newA with the corresponding entries from A
                    for idx in newA.index:
                        for col in newA.columns:
                            if (
                                idx in A.index and col in A.columns
                            ):  # if it's in the original A matrix, copy it over
                                newA.loc[idx, col] = A.loc[idx, col]
                            if (
                                idx in Agam.index and col in Agam.columns
                            ):  # if it's in Agam, copy it over
                                newA.loc[idx, col] = Agam.loc[idx, col]
                            if (
                                idx in Bgam.index and col in Bgam.columns
                            ):  # the input to the cascade is a state
                                newA.loc[idx, col] = Bgam.loc[idx, col]

                    for idx in newB.index:
                        for col in newB.columns:
                            if (
                                idx in B.index and col in B.columns
                            ):  # if it's in the original B matrix, copy it over
                                newB.loc[idx, col] = B.loc[idx, col]
                            if (
                                idx in Bgam.index and col in Bgam.columns
                            ):  # the input to the cascade is a forcing term
                                newB.loc[idx, col] = Bgam.loc[idx, col]

                    for idx in newC.index:
                        for col in newC.columns:
                            if (
                                idx in C.index and col in C.columns
                            ):  # if it's in the original C matrix, copy it over
                                newC.loc[idx, col] = C.loc[idx, col]
                            if (
                                idx in Cgam.index and col in Cgam.columns
                            ):  # outputs from the cascades
                                newA.loc[idx, col] = Cgam.loc[idx, col]

                    # copy over
                    A = newA.copy(deep=True)
                    B = newB.copy(deep=True)
                    C = newC.copy(deep=True)

    A.replace("n", 0.0, inplace=True)
    B.replace("n", 0.0, inplace=True)
    C.replace("n", 0.0, inplace=True)

    if swmm:
        pass
        #############
        # TODO: cast strings back to tuples in the indices and columns
        #############
        # cast the index and columns of causative_topology to tuples. they'll be of the form "(X,Y)"

        # do the same for dependent_columns and independent_columns

        # do the same for the columns of system_data

    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    B = B.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    C = C.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # if bibo_stable is specified and A not Hurwitz, make A Hurwitz by
    # subtracting I * shift from A so that max(real(eig(A))) < 0
    if bibo_stable:
        orig_eigs, _ = np.linalg.eig(A)
        max_real_eig = float(np.max(np.real(orig_eigs)))
        if max_real_eig >= -1e-12:
            logger.warning(
                "stabilizing unstable or marginally stable plant by shifting A"
            )
            epsilon = 10e-4
            shift = max((1 + epsilon) * max_real_eig, epsilon)
            A_stab = A - np.eye(len(A)) * shift
            A = A_stab.copy(deep=True)

    # sindy will scale the coefficients according to the timestep if the index is numeric
    # so the whole system needs to be scaled by the timestep if its numeric
    try:
        pd.to_numeric(
            system_data.index, errors="raise"
        )  # can the index be converted to a numeric type?
        dt = system_data.index.values[1] - system_data.index.values[0]
        A = A / dt
        B = B / dt
        C = C  # what we observe doesn't need to be adjusted, just the dynamics
        logger.info("system response data index converted to numeric type. dt = %s", dt)
    except Exception as e:
        logger.warning("%s", e)
        dt = None

    # cast all of A, B, and C to type float (integers cause issues with LQR / LQE calculations)
    A = A.astype(float)
    B = B.astype(float)
    C = C.astype(float)

    lti_sys = control.ss(
        A, B, C, 0, inputs=B.columns, outputs=C.index, states=A.columns
    )

    # returning the matrices too because control.ss strips the labels from the pandas dataframes and stores them as numpy matrices
    return {"system": lti_sys, "A": A, "B": B, "C": C}

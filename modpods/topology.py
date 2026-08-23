import logging
import warnings
from typing import Any, cast

import networkx as nx
import numpy as np
import pandas as pd
import pysindy as ps  # type: ignore
from scipy.optimize import minimize

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from ._validation import validate_columns, validate_system_data
from .kernels import get_kernel
from .transforms import transform_inputs

logger = logging.getLogger(__name__)


def find_topology_no_geo(
    system_data,
    dependent_columns,
    independent_columns,
    max_iterations=250,
    graph_type="Weak-Conn",
    verbose: Verbosity = "warnings",
    sensor_locations=None,
    init_neighbors=3,
    kernel="gamma",
):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)
    kernel = get_kernel(kernel)
    """
    Infer network topology from time series data using SINDy-based optimization.

    Args:
        system_data: pd.DataFrame with time series data, columns are variables
        dependent_columns: list of column names that are dependent variables
        independent_columns: list of column names that are independent/forcing variables
        max_iterations: maximum iterations for optimization
        graph_type: type of graph connectivity requirement ('Weak-Conn')
        verbose: whether to print detailed output
        sensor_locations: optional dict mapping column names to {"lat": float, "lon": float}.
            If provided, uses geographic filtering to reduce computation by only evaluating
            nearby sensors as potential forcings. Format: {"station_A": {"lat": 41.5, "lon": -74.5}, ...}
        init_neighbors: initial number of nearest neighbors to evaluate when sensor_locations
            is provided (default: 3). Ignored if sensor_locations is None.

    Returns:
        dict with keys: "edges", "best_params", "r2_values", "lead_lag"
    """

    # only print 3 places past the decimal for floats. don't use scientific notation. if less than 0.001, print as <0.001
    pd.options.display.float_format = "{:.3f}".format

    # Helper function to find the lag with strongest cross-correlation
    def cross_correlation_lag(x, y, max_lag):
        """Find the lag with strongest cross-correlation between x and y.

        Returns:
            best_lag: Positive lag means x leads y (x happens before y)
                      Negative lag means y leads x (y happens before x)
            best_corr: The correlation coefficient at best_lag
        """
        best_lag, best_corr = 0, -2
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                xs = x.iloc[-lag:]
                ys = y.iloc[: len(xs)]
            elif lag > 0:
                ys = y.iloc[lag:]
                xs = x.iloc[: len(ys)]
            else:
                xs, ys = x, y
            if len(xs) < 5 or xs.std() == 0 or ys.std() == 0:
                continue
            c = np.corrcoef(xs, ys)[0, 1]
            if np.isnan(c):
                continue
            if c > best_corr:
                best_corr, best_lag = c, lag
        return best_lag, best_corr

    # drop columns from system_data which aren't in dependent_columns or independent_columns
    # this ensures we only analyze the variables of interest
    system_data = pd.concat(
        (system_data[independent_columns], system_data[dependent_columns]),
        axis="columns",
    )

    # Store results for each column pair
    best_params = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=object
    )
    r2_values = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    lead_lag = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    edges = pd.DataFrame(
        index=system_data.columns, columns=system_data.columns, dtype=int, data=0
    )  # from column, to row. causation, not flow.

    for dep_col in dependent_columns:
        _ = np.array(system_data[dep_col].values)

        # First, compute autocorrelation-only R² (no external forcing)
        # This tells us how much of the dynamics can be explained by the state alone
        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(order=10, drop_endpoints=True),
            feature_library=ps.PolynomialLibrary(
                degree=2, include_bias=False, include_interaction=False
            ),
            optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
        )
        # Fit with no control input (u=None), just the state
        fit = model.fit(
            x=system_data.loc[:, dep_col],
            t=np.arange(0, len(system_data.index), 1),
            feature_names=[dep_col],
        )
        auto_r2 = fit.score(
            x=system_data.loc[:, dep_col], t=np.arange(0, len(system_data.index), 1)
        )
        r2_values.loc[dep_col, dep_col] = auto_r2

        for forcing_col in system_data.columns:
            if forcing_col == dep_col:
                continue  # already computed autocorrelation above

            # EXPERIMENTAL: Check lead/lag before expensive SISO optimization
            # Skip if forcing doesn't lead response (comment out to disable this check)
            max_lag_check = min(len(system_data) // 4, 100)
            early_lag, early_xcorr = cross_correlation_lag(
                system_data[forcing_col], system_data[dep_col], max_lag_check
            )
            if early_lag < -5:
                logger.info(
                    "Skipping %s -> %s: forcing lags response (lag=%s)",
                    forcing_col,
                    dep_col,
                    early_lag,
                )
                lead_lag.loc[dep_col, forcing_col] = early_lag
                r2_values.loc[dep_col, forcing_col] = 0.0
                best_params.loc[dep_col, forcing_col] = (
                    2.0,
                    2.0,
                    0.0,
                )  # default params
                continue
            # END EXPERIMENTAL

            logger.info("Optimizing transformation for %s -> %s", forcing_col, dep_col)
            forcing_orig = system_data[[forcing_col]].copy(deep=True)

            # Objective function to minimize (negative because we want to maximize correlation - p_value)
            def objective(params):
                # Create transformation parameter DataFrame
                kernel_params = pd.DataFrame(
                    index=pd.MultiIndex.from_tuples(
                        [(1, p) for p in kernel.param_names],
                        names=["transform", "param"],
                    ),
                    columns=[forcing_col],
                    dtype=float,
                )
                for i, p_name in enumerate(kernel.param_names):
                    kernel_params.loc[(1, p_name), forcing_col] = params[i]

                try:
                    transformed_inputs = pd.DataFrame(index=system_data.index)
                    # SINDY way
                    transformed = transform_inputs(
                        kernel,
                        kernel_params,
                        system_data.index,
                        forcing_orig,
                    )
                    transformed_inputs = pd.concat(
                        (transformed_inputs, transformed[[forcing_col + "_tr_1"]]),
                        axis="columns",
                    )
                    # build a sindy model with these inputs
                    feature_names = [dep_col, str(forcing_col + "_tr_1")]
                    model = ps.SINDy(
                        differentiation_method=ps.FiniteDifference(
                            order=10, drop_endpoints=True
                        ),
                        feature_library=ps.PolynomialLibrary(
                            degree=2, include_bias=False, include_interaction=False
                        ),
                        optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
                    )
                    fit = model.fit(
                        x=system_data.loc[:, dep_col],
                        u=transformed_inputs,
                        t=np.arange(0, len(system_data.index), 1),
                        feature_names=feature_names,
                    )
                    r2 = fit.score(
                        x=system_data.loc[:, dep_col],
                        u=transformed_inputs,
                        t=np.arange(0, len(system_data.index), 1),
                    )

                    return -r2  # Negative because minimize
                except Exception as e:
                    # if e contains any letters or numbers, print it for debugging
                    if any(c.isalnum() for c in str(e)):
                        if _normalize_verbose(verbose) != "warnings":
                            logger.debug("Exception in objective function: %s", e)

                    return 1e10  # Large penalty for invalid parameters

            # Initial guess and bounds
            x0 = kernel.default_init.tolist()
            bounds = [tuple(b) for b in kernel.default_bounds]

            # Optimize
            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": max_iterations,
                    "disp": verbose != "warnings",
                    "fatol": 1e-4,
                },
            )

            # Store best results
            best_params.loc[dep_col, forcing_col] = tuple(result.x.tolist())

            kernel_params = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [(1, p) for p in kernel.param_names],
                    names=["transform", "param"],
                ),
                columns=[forcing_col],
                dtype=float,
            )
            for i, p_name in enumerate(kernel.param_names):
                kernel_params.loc[(1, p_name), forcing_col] = result.x[i]

            transformed = transform_inputs(
                kernel,
                kernel_params,
                system_data.index,
                forcing_orig,
            )
            _ = np.array(transformed[forcing_col + "_tr_1"].values)
            feature_names = [dep_col, forcing_col]
            model = ps.SINDy(
                differentiation_method=ps.FiniteDifference(
                    order=10, drop_endpoints=True
                ),
                feature_library=ps.PolynomialLibrary(
                    degree=2, include_bias=False, include_interaction=False
                ),
                optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
            )
            fit = model.fit(
                x=system_data.loc[:, dep_col],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed,
                feature_names=feature_names,
            )
            # evaluate the r2 score
            r2 = fit.score(
                x=system_data.loc[:, dep_col],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed,
            )
            try:
                model.print()
            except Exception as e:
                logger.warning("%s", e)

            r2_values.loc[dep_col, forcing_col] = r2

            # Compute cross-correlation lag between forcing and response
            # Use max_lag of 1/4 of the data length, capped at 100
            max_lag = min(len(system_data) // 4, 100)
            best_lag, best_xcorr = cross_correlation_lag(
                system_data[forcing_col], system_data[dep_col], max_lag
            )
            lead_lag.loc[dep_col, forcing_col] = best_lag

            logger.info("Optimizing transformation for %s -> %s", forcing_col, dep_col)
            logger.info(
                "  BEST: %s",
                ", ".join(
                    f"{n}={v:.2f}"
                    for n, v in zip(kernel.param_names, result.x.tolist())
                ),
            )
            logger.info("  Cross-correlation: lag=%s, corr=%.4f", best_lag, best_xcorr)
            best_params.loc[dep_col, forcing_col] = tuple(result.x.tolist())

            logger.info("R2 Values:")
            logger.info("%s", r2_values)

    logger.info("Final SISO R2 Values:")
    logger.info("%s", r2_values)
    current_best_r2 = pd.Series(index=dependent_columns, dtype=float, data=0.0)
    logger.info("Lead/Lag Matrix: (positive lag means forcing leads response)")
    logger.info("%s", lead_lag)

    # OPTION A: Mask r2 values by nonnegative lead/lag (forcing must lead response)
    # This is applied AFTER SISO optimization - use this if not skipping early
    # r2_values = r2_values.mask(lead_lag < 0, 0)
    # print("Masked R2 Values (only forcing leads response):")
    # print(r2_values)

    # OPTION B: Early skip is done above in the SISO loop - r2_values already has 0s for skipped pairs

    # first identify the maximum r^2 value in each row. we know these will be included in the final topology
    # with an exception: if we form a cycle with these initial edges, remove the lowest r^2 edge in the cycle
    # for dep_col in dependent_columns:
    #    forcing_col = r2_values.loc[dep_col,:].idxmax()
    #    edges.loc[dep_col,forcing_col] = 1
    #    current_best_r2[dep_col] = r2_values.loc[dep_col,forcing_col]

    # try a different method of picking initial edges
    # find the n_columns edges in r2_values with the highest r^2 values
    # if they are the maximum in their row and column, include them
    sorted_r2 = r2_values.stack().sort_values(ascending=False)  # type: ignore[call-overload]
    for idx in sorted_r2.index:
        dep_col = idx[0]
        forcing_col = idx[1]
        r2 = r2_values.loc[dep_col, forcing_col]
        # is this the maximum in its row and column? (strongest connection for giver and receiver)
        if (
            r2 == r2_values.loc[dep_col, :].max()
            and r2 == r2_values.loc[:, forcing_col].max()
        ):
            edges.loc[dep_col, forcing_col] = 1
            current_best_r2[dep_col] = r2_values.loc[dep_col, forcing_col]
            logger.info(
                "Initial edge added: %s -> %s with r^2 = %.4f",
                forcing_col,
                dep_col,
                r2,
            )

    # check for cycles and remove them iteratively
    G = nx.from_pandas_adjacency(edges, create_using=nx.DiGraph)
    while True:
        try:
            # find_cycle returns a list of edges forming ONE cycle: [(u, v, dir), (v, w, dir), ...]
            cycle_edges = list(nx.find_cycle(G, orientation="original"))
            if len(cycle_edges) == 0:
                break

            logger.info(
                "Found cycle with %s edges. Removing lowest r^2 edge.",
                len(cycle_edges),
            )
            logger.info("Cycle edges: %s", [(e[0], e[1]) for e in cycle_edges])

            # find the edge with the lowest r^2 in the cycle
            min_r2 = float("inf")
            edge_to_remove = None
            for edge in cycle_edges:
                from_node = edge[0]  # source node
                to_node = edge[1]  # target node
                # In our adjacency matrix, edges.loc[row, col] = 1 means col -> row
                # So we need r2_values.loc[to_node, from_node] for edge from_node -> to_node
                r2 = r2_values.loc[to_node, from_node]
                logger.info("Edge %s -> %s: r^2 = %.4f", from_node, to_node, r2)
                if r2 < min_r2:
                    min_r2 = r2
                    edge_to_remove = (from_node, to_node)

            # remove this edge from our edges DataFrame
            # edges.loc[row, col] = 1 means col -> row, so to remove from_node -> to_node:
            edges.loc[edge_to_remove[1], edge_to_remove[0]] = 0
            logger.info(
                "Removed edge %s -> %s with r^2 = %.4f",
                edge_to_remove[0],
                edge_to_remove[1],
                min_r2,
            )

            # rebuild the graph for next iteration
            G = nx.from_pandas_adjacency(edges, create_using=nx.DiGraph)

        except nx.NetworkXNoCycle:
            # No cycle found, we're done
            logger.info("No cycles detected in initial edges.")
            break
        except Exception as e:
            logger.warning("Error during cycle detection: %s", e)
            break

    # Helper function to update correlation-weighted R² scores for a single output variable
    def update_corr_weighted_r2(dep_col):
        """Update corr_wted_r2 for all potential inputs to dep_col based on current edges."""
        selected_inputs = list(edges.loc[dep_col, edges.loc[dep_col, :] == 1].index)
        for forcing_col in system_data.columns:
            if forcing_col in selected_inputs or forcing_col == dep_col:
                continue  # skip already selected inputs / autocorrelation

            if len(selected_inputs) > 0:
                correlations = []
                for sel_input in selected_inputs:
                    # compute correlation between transformed versions of forcing_col and sel_input
                    params_1 = best_params.loc[dep_col, forcing_col]
                    kernel_params_1 = pd.DataFrame(
                        index=pd.MultiIndex.from_tuples(
                            [(1, p) for p in kernel.param_names],
                            names=["transform", "param"],
                        ),
                        columns=[forcing_col],
                        dtype=float,
                    )
                    for i, p_name in enumerate(kernel.param_names):
                        kernel_params_1.loc[(1, p_name), forcing_col] = params_1[i]
                    transformed_1 = transform_inputs(
                        kernel,
                        kernel_params_1,
                        system_data.index,
                        system_data[[forcing_col]],
                    )

                    params_2 = best_params.loc[dep_col, sel_input]
                    kernel_params_2 = pd.DataFrame(
                        index=pd.MultiIndex.from_tuples(
                            [(1, p) for p in kernel.param_names],
                            names=["transform", "param"],
                        ),
                        columns=[sel_input],
                        dtype=float,
                    )
                    for i, p_name in enumerate(kernel.param_names):
                        kernel_params_2.loc[(1, p_name), sel_input] = params_2[i]
                    transformed_2 = transform_inputs(
                        kernel,
                        kernel_params_2,
                        system_data.index,
                        system_data[[sel_input]],
                    )

                    together = pd.DataFrame(index=system_data.index)
                    together[forcing_col] = transformed_1[str(forcing_col + "_tr_1")]
                    together[sel_input] = transformed_2[str(sel_input + "_tr_1")]

                    # Check for zero variance before computing correlation
                    if (
                        together[forcing_col].std() == 0
                        or together[sel_input].std() == 0
                    ):
                        corr = 2.0  # constant variable, exclude it
                    else:
                        corr = np.corrcoef(together[forcing_col], together[sel_input])[
                            0, 1
                        ]
                        if np.isnan(corr):
                            corr = 0.0
                    correlations.append(abs(corr))
                _ = np.max(correlations)
            else:
                _ = 0.0

            corr_wted_r2.loc[dep_col, forcing_col] = (
                r2_values.loc[dep_col, forcing_col] * 1
            )  # ((1 - max_corr)) # was **10

    # Initialize correlation-weighted R² scores
    corr_wted_r2 = r2_values.copy(deep=True)
    for dep_col in dependent_columns:
        update_corr_weighted_r2(dep_col)

    sorted_r2 = r2_values.stack().sort_values(ascending=False)  # type: ignore[call-overload]
    if _normalize_verbose(verbose) != "warnings":
        logger.info("Sorted R2 values:")
        logger.info("%s", sorted_r2)

    # Use a while loop so we can re-sort after each edge addition
    # This ensures we always pick the best remaining candidate after correlation weights are updated
    evaluated_pairs = (
        set()
    )  # Track pairs we've already evaluated to avoid infinite loops

    while True:
        sorted_corr_wted_r2 = corr_wted_r2.stack().sort_values(ascending=False)  # type: ignore[call-overload]
        # Find the best candidate we haven't evaluated yet
        idx = None
        for candidate_idx in sorted_corr_wted_r2.index:
            if (
                candidate_idx not in evaluated_pairs
                and edges.loc[candidate_idx[0], candidate_idx[1]] != 1
            ):
                idx = candidate_idx
                break

        if idx is None:
            logger.info("No more candidate edges to evaluate.")
            break

        evaluated_pairs.add(idx)
        output_variable = idx[0]
        forcing_variable = idx[1]
        r2 = r2_values.loc[output_variable, forcing_variable]

        non_rain_edges = edges.loc[
            ~edges.index.str.contains("rain"), ~edges.columns.str.contains("rain")
        ]

        # would adding this edge reduce the number of components in the graph? (not considering rain)
        non_rain_edges_if_added = non_rain_edges.copy(deep=True)
        non_rain_edges_if_added.loc[output_variable, forcing_variable] = 1

        n_components_now = nx.number_weakly_connected_components(
            nx.from_pandas_adjacency(non_rain_edges, create_using=nx.DiGraph)
        )
        if n_components_now == 1:
            logger.info("graph is weakly connected.")
            # done
            break

        n_components = nx.number_weakly_connected_components(
            nx.from_pandas_adjacency(non_rain_edges_if_added, create_using=nx.DiGraph)
        )
        if "rain" not in forcing_variable.lower():  # always allow rain edges
            if n_components >= n_components_now:
                logger.info(
                    "Skipping addition of %s -> %s as it does not improve connectivity",
                    forcing_variable,
                    output_variable,
                )
                continue  # skip this addition as it doesn't improve connectivity

        logger.info(
            "Evaluating edge %s -> %s with r2 = %.4f",
            forcing_variable,
            output_variable,
            r2,
        )
        logger.info("current best r2 values:")
        logger.info("%s", current_best_r2)
        # build the candidate input set
        selected_inputs = list(
            edges.loc[output_variable, edges.loc[output_variable, :] == 1].index
        )
        candidate_inputs = selected_inputs + [forcing_variable]

        # optimize the transformations for all candidate inputs together, using siso best params as initial guesses
        def joint_objective(params, debug=False):
            # params is a flat list of shape, scale, loc for each candidate input
            transformed_inputs = pd.DataFrame(index=system_data.index)
            for i, input_var in enumerate(candidate_inputs):
                kernel_params = pd.DataFrame(
                    index=pd.MultiIndex.from_tuples(
                        [(1, p) for p in kernel.param_names],
                        names=["transform", "param"],
                    ),
                    columns=[input_var],
                    dtype=float,
                )
                for j, p_name in enumerate(kernel.param_names):
                    kernel_params.loc[(1, p_name), input_var] = params[
                        i * kernel.num_params + j
                    ]
                forcing_orig = system_data[[input_var]].copy()
                transformed = transform_inputs(
                    kernel,
                    kernel_params,
                    system_data.index,
                    forcing_orig,
                )
                # Include BOTH original and transformed columns, consistent with SISO phase
                transformed_inputs = pd.concat(
                    (transformed_inputs, transformed), axis="columns"
                )
            # build and fit the sindy model
            feature_names = [output_variable] + list(transformed_inputs.columns)
            model = ps.SINDy(
                differentiation_method=ps.FiniteDifference(
                    order=10, drop_endpoints=True
                ),
                feature_library=ps.PolynomialLibrary(
                    degree=2, include_bias=False, include_interaction=False
                ),
                optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
            )
            fit = model.fit(
                x=system_data.loc[:, output_variable],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed_inputs,
                feature_names=feature_names,
            )
            r2 = fit.score(
                x=system_data.loc[:, output_variable],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed_inputs,
            )
            if debug:
                logger.debug(
                    "DEBUG joint_objective: inputs=%s, r2=%.4f",
                    list(transformed_inputs.columns),
                    r2,
                )
                try:
                    model.print()
                except Exception:
                    pass
            return -r2  # Negative because minimize

        # initial guesses from SISO optimization
        x0 = []
        for input_var in candidate_inputs:
            shape, scale, loc = best_params.loc[output_variable, input_var]
            x0.extend([shape, scale, loc])
        bounds = []
        for input_var in candidate_inputs:
            bounds.extend(
                [(1.0, 300.0), (1e-5, 300.0), (0.0, 300.0)]
            )  # shape, scale, loc

        # First, compute baseline R² using SISO-optimized params (x0)
        # This ensures we never do worse than the initial guess
        baseline_r2 = -joint_objective(x0, debug=True)
        logger.info("Baseline R² with SISO params: %.4f", baseline_r2)

        # optimize
        multivariable_iterations = max_iterations * len(candidate_inputs)
        result = minimize(
            joint_objective,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": multivariable_iterations,
                "disp": verbose != "warnings",
            },
        )
        optimized_r2 = -result.fun

        # Use optimized params only if they improve on baseline, otherwise keep SISO params
        if optimized_r2 >= baseline_r2:
            optimized_params = result.x
            logger.info("Optimizer improved R² to %.4f", optimized_r2)
        else:
            optimized_params = cast(np.ndarray, np.asarray(x0, dtype=np.float64))
            logger.info(
                "Optimizer found worse R² (%.4f), keeping SISO params (R² = %.4f)",
                optimized_r2,
                baseline_r2,
            )

        # extract best params
        for i, input_var in enumerate(candidate_inputs):
            shape = optimized_params[i * 3]
            scale = optimized_params[i * 3 + 1]
            loc = optimized_params[i * 3 + 2]
            best_params.loc[output_variable, input_var] = (shape, scale, loc)
        # compute final r2 with optimized params
        transformed_inputs = pd.DataFrame(index=system_data.index)
        for i, input_var in enumerate(candidate_inputs):
            kernel_params = pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [(1, p) for p in kernel.param_names],
                    names=["transform", "param"],
                ),
                columns=[input_var],
                dtype=float,
            )
            for j, p_name in enumerate(kernel.param_names):
                kernel_params.loc[(1, p_name), input_var] = optimized_params[
                    i * kernel.num_params + j
                ]
            forcing_orig = system_data[[input_var]].copy()
            transformed = transform_inputs(
                kernel,
                kernel_params,
                system_data.index,
                forcing_orig,
            )
            # Include BOTH original and transformed columns, consistent with SISO phase
            transformed_inputs = pd.concat(
                (transformed_inputs, transformed), axis="columns"
            )
        feature_names = [output_variable] + list(transformed_inputs.columns)
        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(order=10, drop_endpoints=True),
            feature_library=ps.PolynomialLibrary(
                degree=2, include_bias=False, include_interaction=False
            ),
            optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
        )
        fit = model.fit(
            x=system_data.loc[:, output_variable],
            t=np.arange(0, len(system_data.index), 1),
            u=transformed_inputs,
            feature_names=feature_names,
        )
        r2 = fit.score(
            x=system_data.loc[:, output_variable],
            t=np.arange(0, len(system_data.index), 1),
            u=transformed_inputs,
        )

        logger.info(
            "Testing inputs %s for output %s -> r2 = %.4f",
            candidate_inputs,
            output_variable,
            r2,
        )
        if (
            r2 > current_best_r2[output_variable] + 0.01
        ):  # only keep it if it improves the r2 by at least 1%
            # add a conditional here for reducing the number of components in the graph. if it doesn't connect things that were previously unconnected, we don't want it.
            selected_inputs = candidate_inputs
            current_best_r2[output_variable] = r2
            logger.info(
                "Accepted new input %s, updated r2 = %.4f",
                forcing_variable,
                current_best_r2[output_variable],
            )
            edges.loc[output_variable, forcing_variable] = 1

            # Update correlation-weighted R² for this output since we added a new input
            # The while loop will re-sort at the next iteration
            update_corr_weighted_r2(output_variable)

        else:
            logger.info(
                "Rejected new input %s, r2 would be %.4f",
                forcing_variable,
                r2,
            )

    # transpose edges to have from -> to convention
    edges = edges.T
    # earlier in the code we have dependent variables on the rows and independent on columns.
    # that arrangement makes comparing the effect of potential inputs on each output easier.
    # but for output, it's more intuitive to have from -> to convention, so we transpose before returning.

    return {
        "edges": edges,
        "best_params": best_params,
        "r2_values": r2_values,
        "lead_lag": lead_lag,
    }


def infer_causative_topology(  # noqa: F811
    # type: ignore
    system_data,
    dependent_columns,
    independent_columns,
    graph_type="Weak-Conn",
    verbose: Verbosity = "warnings",
    max_iter=250,
    swmm=False,
    method="sindy",  # Changed default from "granger" to "sindy"
    derivative=False,
    sensor_locations=None,
    init_neighbors=3,
    kernel="gamma",
):
    """
    Infer causative topology from time series data using SINDy-based optimization.

    Args:
        system_data: pd.DataFrame with time series data
        dependent_columns: list of column names that are dependent variables
        independent_columns: list of column names that are independent/forcing variables
        graph_type: type of graph connectivity requirement ('Weak-Conn' or 'Strong-Conn')
        verbose: whether to print detailed output
        max_iter: maximum iterations for optimization
        swmm: whether this is for SWMM/pystorms data
        method: inference method ('sindy' is the only supported method now)
        derivative: whether to use derivative of response
        sensor_locations: optional dict mapping column names to {"lat": float, "lon": float}
        init_neighbors: initial number of nearest neighbors to evaluate when sensor_locations is provided (default: 3)

    Returns:
        dict with keys: "edges", "best_params", "r2_values", "lead_lag",
        "causative_topo", "total_graph".
        - edges: DataFrame adjacency matrix (from -> to convention)
        - best_params: DataFrame of transformation parameters (shape, scale, loc)
        - r2_values: DataFrame of R^2 values for each potential edge
        - lead_lag: DataFrame of lead/lag values (positive = forcing leads response)
        - causative_topo: DataFrame of "d"/"n" labels (dep row, forcing col)
        - total_graph: DataFrame of R^2 weights (dep row, forcing col)
    """

    # Handle deprecated methods
    if method in ("granger", "ccm", "transfer_entropy"):
        warnings.warn(
            f"Method '{method}' is deprecated. The Granger causality, CCM, and "
            "Transfer Entropy methods have been replaced by the improved SINDy-based "
            "topology inference (method='sindy'), which provides significantly better "
            "results. Please use method='sindy' (the new default).",
            DeprecationWarning,
            stacklevel=2,
        )
        # Fall back to new method
        method = "sindy"

    if swmm:
        # do the same for dependent_columns and independent_columns
        dependent_columns = [str(col) for col in dependent_columns]
        independent_columns = [str(col) for col in independent_columns]
        # do the same for the columns of system_data
        system_data.columns = system_data.columns.astype(str)

    # Import and use the new SINDy-based topology inference
    # (using our local implementation)
    result = find_topology_no_geo(
        system_data=system_data,
        dependent_columns=dependent_columns,
        independent_columns=independent_columns,
        sensor_locations=sensor_locations,
        max_iterations=max_iter,
        graph_type=graph_type,
        verbose=verbose,
        init_neighbors=init_neighbors,
        kernel=kernel,
    )
    # Convert result to match expected return format for backward compatibility
    # The new method returns edges in from->to convention (transposed from old)
    edges = result["edges"]
    _ = result["best_params"]
    r2_values = result["r2_values"]
    _ = result["lead_lag"]

    # For backward compatibility with code expecting (causative_topo, total_graph) tuple
    # causative_topo: 'd' for directed edge, 'n' for no edge
    # total_graph: numeric weights (R² values)
    causative_topo = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns
    ).fillna("n")
    total_graph = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    ).fillna(0.0)

    # Fill in the edges from the result
    # edges is in from->to convention (row=from, col=to)
    # causative_topo expects row=dependent (to), col=forcing (from)
    for dep_col in dependent_columns:
        for forcing_col in system_data.columns:
            if edges.loc[forcing_col, dep_col] == 1:  # from forcing_col -> to dep_col
                causative_topo.loc[dep_col, forcing_col] = "d"
                total_graph.loc[dep_col, forcing_col] = r2_values.loc[
                    dep_col, forcing_col
                ]

    return {
        "edges": edges,
        "best_params": result["best_params"],
        "r2_values": r2_values,
        "lead_lag": result["lead_lag"],
        "causative_topo": causative_topo,
        "total_graph": total_graph,
    }


class TopologyInference:
    """Topology inference estimator following scikit-learn conventions."""

    def __init__(
        self,
        dependent_columns: list[str],
        independent_columns: list[str],
        graph_type: str = "Weak-Conn",
        max_iter: int = 250,
        kernel: str = "gamma",
        verbose: Verbosity = "warnings",
        sensor_locations: dict[str, dict[str, float]] | None = None,
        init_neighbors: int = 3,
    ) -> None:
        self.dependent_columns = dependent_columns
        self.independent_columns = independent_columns
        self.graph_type = graph_type
        self.max_iter = max_iter
        self.kernel = kernel
        self.verbose = verbose
        self.sensor_locations = sensor_locations
        self.init_neighbors = init_neighbors
        self.causative_topo_: pd.DataFrame | None = None
        self.total_graph_: pd.DataFrame | None = None
        self.edges_: pd.DataFrame | None = None
        self.best_params_: pd.DataFrame | None = None
        self.r2_values_: pd.DataFrame | None = None
        self.lead_lag_: pd.DataFrame | None = None

    def fit(self, system_data: pd.DataFrame, **kwargs: Any) -> "TopologyInference":
        validate_system_data(system_data)
        validate_columns(system_data, self.dependent_columns, "dependent_columns")
        validate_columns(system_data, self.independent_columns, "independent_columns")

        result = infer_causative_topology(
            system_data=system_data,
            dependent_columns=self.dependent_columns,
            independent_columns=self.independent_columns,
            graph_type=self.graph_type,
            max_iter=self.max_iter,
            kernel=self.kernel,
            verbose=self.verbose,
            sensor_locations=self.sensor_locations,
            init_neighbors=self.init_neighbors,
            **kwargs,
        )
        self.causative_topo_ = result["causative_topo"]
        self.total_graph_ = result["total_graph"]
        self.edges_ = result["edges"]
        self.best_params_ = result["best_params"]
        self.r2_values_ = result["r2_values"]
        self.lead_lag_ = result["lead_lag"]
        return self

    def predict(self, system_data: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        if self.causative_topo_ is None:
            raise RuntimeError("Estimator has not been fitted yet.")
        return infer_causative_topology(
            system_data=system_data,
            dependent_columns=self.dependent_columns,
            independent_columns=self.independent_columns,
            graph_type=self.graph_type,
            max_iter=self.max_iter,
            kernel=self.kernel,
            verbose=self.verbose,
            sensor_locations=self.sensor_locations,
            init_neighbors=self.init_neighbors,
            **kwargs,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "dependent_columns": self.dependent_columns,
            "independent_columns": self.independent_columns,
            "graph_type": self.graph_type,
            "max_iter": self.max_iter,
            "kernel": self.kernel,
            "verbose": self.verbose,
            "sensor_locations": self.sensor_locations,
            "init_neighbors": self.init_neighbors,
        }

    def set_params(self, **params: Any) -> "TopologyInference":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"TopologyInference(dependent_columns={self.dependent_columns}, "
            f"independent_columns={self.independent_columns}, "
            f"graph_type={self.graph_type!r}, max_iter={self.max_iter}, "
            f"kernel={self.kernel!r})"
        )

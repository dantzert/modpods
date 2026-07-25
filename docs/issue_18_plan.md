# Issue #18 Plan: Make modpods as small as possible

## Goal
Reduce LOC, dependencies, and test runtime while preserving every public behavior and test that actually covers a path.

## Current State
| Metric | Value |
|--------|-------|
| `modpods.py` LOC | 3,369 |
| Total repo Python KB | ~325 KB |
| Test count | 26 (23 non-slow) |
| Non-slow test runtime | ~4.5 min |
| Dependencies | 12 direct |
| Dead files | `modpods_backup.py`, 16 loose PNG/SVG/TMP artifacts |

## Phase 1 — Delete dead weight (0 risk)
- Remove `modpods_backup.py`.
- Delete root-level test artifacts (`test_*.png`, `test_*.svg`, `*.TMP`, `*.h11~`, `*.qeb~`, `*.wa0~`).
- Remove commented-out code blocks in `modpods.py` (lines ~349-354, ~771-777, ~1071-1077, ~1318-1335, ~2152-2158, ~2220-2224, ~2529-2545, ~2955-2968, ~3009-3019).
- Remove deprecated `find_topology` wrapper.

## Phase 2 — Split monolith into modules
Split `modpods.py` into these files under `modpods/`:

```
modpods/
  __init__.py      # re-export public API
  transforms.py    # TransformCache, transform_inputs, _expected_improvement, _propose_location
  train.py         # delay_io_train, _run_scipy_optimizer
  model.py         # SINDY_delays_MI
  predict.py       # delay_io_predict
  lti.py            # lti_from_gamma, lti_system_gen
  topology.py       # find_topology_no_geo, infer_causative_topology
```

Public API stays `modpods.<func>`. No behavioral changes.

## Phase 3 — Replace custom code with stdlib / deps
| Custom code | Replacement | Savings |
|-------------|-------------|---------|
| `_expected_improvement`, `_propose_location` | Already used by `sklearn` GP; these ~50 lines only serve the inlined Bayesian loop. If Bayesian stays, keep; otherwise delete. | ~50 LOC |
| Hand-rolled metric computation (MAE, RMSE, NSE, alpha, beta, HFV, HFV10, LFV, FDC) duplicated in `SINDY_delays_MI` and `delay_io_predict` | Extract one `compute_metrics(y_true, y_pred)` function in `modpods/metrics.py`. | ~120 LOC duplicated |
| Nested loops in `lti_system_gen` copying A/B/C DataFrames entry-by-entry | `A = causative_topology.reindex(index=..., columns=...).fillna(0)` and same for B. | ~40 LOC |
| Repeated `pd.DataFrame` construction for shape/scale/loc inside objective closures | Build once outside, pass mutable views or slice/copy inside. | ~30 LOC |
| `cross_correlation_lag` and `update_corr_weighted_r2` closures inside `find_topology_no_geo` | Lift to module-level functions in `topology.py`. | ~10 LOC |

Dependency cuts:
- `cvxpy` is imported but never used directly (only via pysindy's `_ConstrainedSR3`). Keep as indirect dep; remove explicit install only if pysindy no longer needs it. **Verdict: keep for now, test first.**
- `statsmodels` is imported nowhere in `modpods.py`. **Remove from `pyproject.toml` and `requirements.txt`.**
- `dill` is imported nowhere. **Remove.**
- `pystorms` and `pyswmm` are used only in external scripts/notebooks. Move optional deps to an extras group `[swmm]`. Core install shrinks from 12 to ~9.

## Phase 4 — Deduplicate optimization plumbing
In `delay_io_train`, `SINDY_delays_MI` is called 10+ times per iteration with near-identical argument lists. Extract a `build_sindy_kwargs(...)` dict and a `run_sindy(**kwargs)` wrapper so the tuning loop builds one dict and mutates one param per candidate.

The Bayesian and scipy branches rebuild the same `bounds_list`, `transform_columns`, and `objective_function` closure. Factor into a shared `_setup_optimization(...)` that returns `bounds`, `objective`, and `transform_columns`.

## Phase 5 — Prune and speed up tests
Current: 23 non-slow tests take ~4.5 min (~11.7 s each). Target: <60 s total.

| Action | Detail |
|--------|--------|
| Convert slow tests to fixture-backed, timeboxed | `test_lti_system_gen_returns_state_space`: reduce `max_iter=10`, `max_transforms=1`, shrink cascade to 5 states. |
| Add `fast` marker and make it default | Run `pytest -m fast` in CI; keep `--runslow` for local. |
| Share fixtures more aggressively | `simple_lti_data` and `cascade_lti_system_data` are already module-scoped; ensure no hidden copies. |
| Remove tautological metric tests | `test_transform_inputs_correctness` recomputes the exact same FFT convolution the function already uses. Replace with a known-good literal array or a smaller `n`. |
| Drop redundant prediction tests | `test_all_methods_predictions_agree` asserts correlation > 0.5 between methods that all optimize the same objective. If `test_all_methods_produce_comparable_r2` passes, this is guaranteed. **Delete or merge.** |

Coverage target: keep branch coverage for invalid-input handling (NaN checks, empty DataFrames, negative shape factors). Add one explicit `pytest.raises` test for `transform_inputs` with NaN input if missing.

## Phase 6 — Verify
- `pytest -m fast` passes in <60 s.
- `pytest` (including slow) passes.
- `python -c "import modpods; print(dir(modpods))"` shows identical public names.
- `pip check` shows no broken deps.
- `modpods.py` LOC target: <1,500.

## Ponytail checks
- YAGNI: `modpods_backup.py` — delete. Test artifacts — delete.
- Stdlib: `scipy.stats`, `numpy`, `functools.lru_cache` (already used via `TransformCache`).
- One line before fifty: metric extraction, DataFrame reindexing, correlation lag.
- Deletion over addition: 3 deps cut, 1 deprecated function removed, commented code deleted.
- Fewest files: 7 module files + `__init__.py` is the minimum that stops the monolith from being one edit-grenade.

## Net target
- LOC: -40%+ (from 3,369 to <2,000)
- Test runtime: -75%+ (from 4.5 min to <60 s)
- Direct deps: -25% (from 12 to 9)

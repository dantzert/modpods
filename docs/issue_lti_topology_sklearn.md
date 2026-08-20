# Refactor `lti.py` and `topology.py` to follow scikit-learn conventions

## Goal

Bring the `lti` and `topology` modules in line with the estimator-class pattern
introduced in issue #17 for the core modeling API.

## Proposed Changes

### `TopologyInference` estimator class

- **Constructor**: `TopologyInference(dependent_columns, independent_columns, graph_type="Weak-Conn", max_iter=250, kernel="gamma", ...)`
- **Methods**:
  - `fit(system_data)` → returns `self`, sets `self.causative_topo_` and `self.total_graph_`
  - `predict(system_data)` → re-runs inference (or returns stored result)
  - `get_params()` / `set_params()` for sklearn compatibility
- **Return type fix**: `infer_causative_topology` already returns a dict in this
  branch; the legacy tuple return is removed.

### `LTISystem` estimator class

- **Constructor**: `LTISystem(causative_topology, independent_columns, dependent_columns, max_iter=250, bibo_stable=False, ...)`
- **Methods**:
  - `fit(system_data)` → returns `self`, sets `self.system_` (`control.StateSpace`),
    `self.A_`, `self.B_`, `self.C_`
  - `predict(system_data, u_new)` → simulate using fitted state-space
  - `get_params()` / `set_params()`

### Aliases

Preserve old function names as module-level aliases:
- `modpods.infer_causative_topology` → kept for backward compat
- `modpods.lti_system_gen` → kept for backward compat
- `modpods.find_topology_no_geo` → kept as low-level function

### Validation

Add `validate_system_data` and `validate_columns` checks to all new `fit()` methods.

### Input validation

- `system_data` must be `pd.DataFrame` with `DatetimeIndex`
- `dependent_columns` / `independent_columns` must exist in `system_data`
- Numeric data check

## Open Questions

- Should `TopologyInference.fit()` return `self` (sklearn convention) or a
  dedicated `TopologyResult` object?
- Should `LTISystem` wrap `control.StateSpace` directly or return a dict?
- Does `find_topology_no_geo` need an estimator wrapper, or is it fine as a
  low-level function?

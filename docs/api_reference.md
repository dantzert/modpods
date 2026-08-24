# API / Parameter Reference

This page documents every public symbol exported by `modpods`.  Parameter
types, defaults, and semantics are taken directly from the implementation in
`modpods/`.

---

## `modpods.DelayIO`

Scikit-learn-style estimator for delay-IO model discovery.  Instantiate with
your column names and call `fit()` to train, then `predict()` to simulate.

```python
model = modpods.DelayIO(
    dependent_columns=["Q"],
    independent_columns=["P", "PET"],
    windup_timesteps=30,
    init_transforms=1,
    max_transforms=4,
    poly_order=3,
    optimization_method="bayesian",
    kernel="gamma",
    random_state=42,
)
estimators = model.fit(system_data)
pred = model.predict(system_data, n_transforms=1)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dependent_columns` | `list[str]` | required | Output / state columns to model. |
| `independent_columns` | `list[str]` | required | Input / forcing columns. |
| `windup_timesteps` | `int` | `0` | Number of initial rows to discard as spin-up. |
| `init_transforms` | `int` | `1` | Minimum number of transforms to try. |
| `max_transforms` | `int` | `4` | Maximum number of transforms to try. |
| `poly_order` | `int` | `3` | Polynomial degree for the feature library. |
| `transform_dependent` | `bool` | `False` | If `True`, transform **all** columns (outputs + inputs) instead of only inputs. |
| `verbose` | `str` | `"warnings"` | Logging level: `"warnings"`, `"info"`, or `"debug"`. |
| `include_bias` | `bool` | `False` | Include a constant term in the feature library. |
| `include_interaction` | `bool` | `False` | Include interaction terms in the feature library. |
| `bibo_stable` | `bool` | `False` | Constrain the highest-order output autocorrelation to be negative. |
| `forcing_coef_constraints` | `dict` or `None` | `None` | Dict mapping input column names to constraint strengths. |
| `constraints` | `list[dict]` or `None` | `None` | Custom coefficient constraints. |
| `early_stopping_threshold` | `float` | `0.005` | Minimum R² improvement required to add another transform. |
| `optimization_method` | `str` | `"bayesian"` | Optimizer backend. |
| `kernel` | `str` or `ConvolutionKernel` | `"gamma"` | Kernel name (`"gamma"`, `"lognormal"`, `"bimodal_gamma"`, `"underdamped"`, `"try-all"`, `"run-all"`) or instance. |
| `random_state` | `int` or `None` | `None` | Seed for reproducibility. |

### Methods

- `fit(system_data, **optimizer_kwargs)` → `list[DelayIOModel]`
  Train models for each transform count from `init_transforms` to `max_transforms`.
  Returns a list of fitted `DelayIOModel` objects.  Also sets `self.estimators_`
  and `self.best_estimator_`.

- `predict(system_data, n_transforms=None, evaluation=False, windup_timesteps=None)` → `dict`
  Simulate on new data.  If `n_transforms` is `None`, uses `self.best_estimator_`.

- `get_params(deep=True)` → `dict`
  Return constructor parameters (scikit-learn interface).

- `set_params(**params)` → `DelayIO`
  Update constructor parameters and return `self`.

### Attributes (after `fit`)

- `estimators_` — `list[DelayIOModel]`, one per transform count.
- `best_estimator_` — `DelayIOModel` with the highest training R².

---

## `modpods.DelayIOModel`

A single fitted delay-io model returned by `DelayIO.fit()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_transforms_` | `int` | Number of transforms used. |
| `kernel_type_` | `str` | Kernel name (`"gamma"`, etc.). |
| `kernel_params_` | `pd.DataFrame` | Optimized kernel parameters. |
| `windup_timesteps_` | `int` | Spin-up length. |
| `dependent_columns_` | `list[str]` | Output columns. |
| `independent_columns_` | `list[str]` | Input columns. |
| `error_metrics_` | `dict` | Error metrics from training. |
| `r2_` | `float` | Training R². |

### Methods

- `predict(system_data, evaluation=False, windup_timesteps=None)` → `dict`
  Simulate and optionally evaluate.  Returns dict with `"prediction"`,
  `"error_metrics"`, and `"diverged"`.

---

## `modpods.delay_io_train`

Train a dynamical model from time-series data using polynomial regression with
gamma-distribution convolution transforms.

```python
modpods.delay_io_train(
    system_data,
    dependent_columns,
    independent_columns,
    windup_timesteps=0,
    init_transforms=1,
    max_transforms=4,
    max_iter=250,
    poly_order=3,
    transform_dependent=False,
    verbose="warnings",
    include_bias=False,
    include_interaction=False,
    bibo_stable=False,
    transform_only=None,
    forcing_coef_constraints=None,
    early_stopping_threshold=0.005,
    optimization_method="bayesian",
    **optimizer_kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_data` | `pd.DataFrame` | required | Time-series data. Index is time; columns are variable names. |
| `dependent_columns` | `list[str]` | required | Output / state columns to model. |
| `independent_columns` | `list[str]` | required | Input / forcing columns. |
| `windup_timesteps` | `int` | `0` | Number of initial rows to discard as spin-up. |
| `init_transforms` | `int` | `1` | Minimum number of gamma transforms to try. |
| `max_transforms` | `int` | `4` | Maximum number of gamma transforms to try. |
| `max_iter` | `int` | `250` | Iteration budget for the optimizer. Meaning varies by `optimization_method`. |
| `poly_order` | `int` | `3` | Polynomial degree for the feature library. |
| `transform_dependent` | `bool` | `False` | If `True`, transform **all** columns (outputs + inputs) instead of only inputs. |
| `verbose` | `str` or `bool` | `"warnings"` | Logging level: `"warnings"`, `"info"`, or `"debug"`. |
| `include_bias` | `bool` | `False` | Include a constant term in the feature library. |
| `include_interaction` | `bool` | `False` | Include interaction terms in the feature library. |
| `bibo_stable` | `bool` | `False` | Constrain the highest-order output autocorrelation to be negative. |
| `transform_only` | `list[str]` or `None` | `None` | Subset of `independent_columns` to transform; the rest pass through untransformed. |
| `forcing_coef_constraints` | `dict[str, float]` or `None` | `None` | Dict mapping input column names to constraint strengths. **Sign convention:** a positive value constrains the coefficient to be `<= -value`. |
| `early_stopping_threshold` | `float` | `0.005` | Minimum R² improvement required to add another transform. If the gain is smaller, training stops early. |
| `optimization_method` | `str` | `"bayesian"` | Optimizer backend. See table below. |
| `**optimizer_kwargs` | `dict` | — | Extra keyword arguments forwarded to the underlying optimizer (e.g., `scipy.optimize.differential_evolution`). |

### `optimization_method` options

| Value | Backend | Notes |
|-------|---------|-------|
| `"bayesian"` | Built-in Gaussian-process Expected Improvement | Default. Good default for small budgets. |
| `"differential_evolution"` | `scipy.optimize.differential_evolution` | Population-based global search. |
| `"dual_annealing"` | `scipy.optimize.dual_annealing` | Stochastic; often needs ~4× `max_iter`. |
| `"simulated_annealing"` | `scipy.optimize.simulated_annealing` | Stochastic; often needs ~4× `max_iter`. |
| `"direct"` | `scipy.optimize.direct` | Deterministic partition-based. |
| `"brute"` | `scipy.optimize.brute` | Grid search (20×20 by default). |

### Returns

`dict` keyed by `num_transforms` (int).  Each value is a dict:

```python
{
    "final_model": {
        "model": SystemIdModel,
        "error_metrics": {
            "MAE", "RMSE", "NSE", "alpha", "beta",
            "HFV", "HFV10", "LFV", "FDC", "r2",
        },
        "simulated": bool,
        "response": pd.DataFrame,
        "forcing": pd.DataFrame,
        "index": pd.Index,
        "diverged": bool,
    },
    "shape_factors": pd.DataFrame,
    "scale_factors": pd.DataFrame,
    "loc_factors": pd.DataFrame,
    "windup_timesteps": int,
    "dependent_columns": list[str],
    "independent_columns": list[str],
    "transform_cache": TransformCache,
}
```

---

## `modpods.delay_io_predict`

Simulate a trained model on new data and optionally compute error metrics.

```python
modpods.delay_io_predict(
    delay_io_model,
    system_data,
    num_transforms=1,
    evaluation=False,
    windup_timesteps=None,
    verbose="warnings",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delay_io_model` | `dict` | required | Model dict returned by `delay_io_train`. |
| `system_data` | `pd.DataFrame` | required | New time-series data with the same columns as training data. |
| `num_transforms` | `int` | `1` | Which transform-count entry in the model dict to use. |
| `evaluation` | `bool` | `False` | If `True`, compute error metrics against `system_data`. |
| `windup_timesteps` | `int` or `None` | `None` | Override spin-up length; defaults to the value stored in the model. |
| `verbose` | `str` or `bool` | `"warnings"` | Logging level: `"warnings"`, `"info"`, or `"debug"`. |

### Returns

`dict` with keys:

- `"prediction"` — `np.ndarray` of simulated values.
- `"error_metrics"` — dict of metric name → list of values (only if `evaluation=True`).
- `"diverged"` — `bool`, `True` if simulation raised an exception.

---

## `modpods.transform_inputs`

Apply gamma-distribution convolution transforms to forcing inputs using
FFT-based convolution.

```python
modpods.transform_inputs(
    shape_factors,
    scale_factors,
    loc_factors,
    index,
    forcing,
    *,
    cache=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape_factors` | `pd.DataFrame` | Rows are transform indices (`1..N`), columns are variable names. Values are gamma shape parameters. |
| `scale_factors` | `pd.DataFrame` | Same layout; gamma scale parameters. |
| `loc_factors` | `pd.DataFrame` | Same layout; gamma location parameters. |
| `index` | `pd.Index` | Time index (used for output length). |
| `forcing` | `pd.DataFrame` | Raw forcing inputs. |
| `cache` | `TransformCache` or `None` | Optional memoization cache. Near-identical parameter sets reuse cached results. |

### Returns

`pd.DataFrame` — original columns plus `{col}_tr_{i}` columns for each transform.

---

## `modpods.TransformCache`

LRU cache for gamma-transformed time series.  Keys are quantized so
near-identical parameter sets hit the same entry.

```python
cache = modpods.TransformCache(max_entries=2000, quantization=1e-6)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_entries` | `int` | `2000` | Maximum cached transforms before oldest entries are evicted. |
| `quantization` | `float` | `1e-6` | Rounding step for cache keys. Set to `0` to disable. |

### Methods

- `get(input_name, forcing_values, shape, scale, loc)` → `np.ndarray`
- `clear()` — reset cache and counters.
- `stats()` → `dict` with keys `hits`, `misses`, `total`, `hit_rate`, `size`, `max_entries`.

---

## `modpods.SINDY_delays_MI`

Build and fit a system-identification model with optional coefficient constraints
and gamma-transformed inputs.  This is the low-level function called by
`delay_io_train`; most users should call `delay_io_train` instead.

```python
modpods.SINDY_delays_MI(
    shape_factors,
    scale_factors,
    loc_factors,
    index,
    forcing,
    response,
    final_run,
    poly_degree,
    include_bias,
    include_interaction,
    windup_timesteps,
    bibo_stable=False,
    transform_dependent=False,
    transform_only=None,
    forcing_coef_constraints=None,
    transform_cache=None,
    verbose="warnings",
)
```

### Returns

`dict` with keys:

- `"model"` — fitted `SystemIdModel` object (or `None` on failure).
- `"error_metrics"` — dict of metric name → list of values (or `[False]` on failure).
- `"r2"` — training R² (float).
- `"simulated"`, `"response"`, `"forcing"`, `"index"`, `"diverged"`.

---

## `modpods.infer_causative_topology`

Discover which input variables causally influence which output variables
using polynomial regression with gamma transforms.

```python
causative_topo, total_graph = modpods.infer_causative_topology(
    system_data,
    dependent_columns,
    independent_columns,
    graph_type="Weak-Conn",
    verbose="warnings",
    max_iter=250,
    swmm=False,
    method="polynomial_regression",
    derivative=False,
    sensor_locations=None,
    init_neighbors=3,
)
```

### Returns

- `causative_topo` — `pd.DataFrame` indexed by `dependent_columns`, columns are all variables in `system_data`.  Values are `"d"` (delayed/directed), `"i"` (immediate), or `"n"` (no edge).
- `total_graph` — `pd.DataFrame` of the same shape with numeric R² weights.

---

## `modpods.find_topology_no_geo`

Lower-level topology inference (no geographic filtering).  Returns a dict
instead of the legacy tuple.

```python
result = modpods.find_topology_no_geo(
    system_data,
    dependent_columns,
    independent_columns,
    max_iterations=250,
    graph_type="Weak-Conn",
    verbose="warnings",
    sensor_locations=None,
    init_neighbors=3,
)
```

### Returns

`dict` with keys:

- `"edges"` — `pd.DataFrame` adjacency matrix (`1` = edge, `0` = no edge).
- `"best_params"` — `pd.DataFrame` of `(shape, scale, loc)` tuples.
- `"r2_values"` — `pd.DataFrame` of R² for each candidate edge.
- `"lead_lag"` — `pd.DataFrame` of cross-correlation lags.

---

## `modpods.lti_from_gamma`

Fit an LTI state-space system whose impulse response approximates a gamma
PDF.

```python
result = modpods.lti_from_gamma(
    shape,
    scale,
    location,
    dt=0,
    desired_NSE=0.999,
    verbose="warnings",
    max_state_dim=50,
    max_iterations=200,
    max_pole_speed=5,
    min_pole_speed=0.01,
)
```

### Returns

`dict` with keys:

- `"lti_approx"` — `control.StateSpace` system.
- `"lti_approx_output"` — `np.ndarray` impulse response values.
- `"error"` — sum of absolute errors.
- `"t"` — time vector.
- `"gamma_pdf"` — target gamma PDF values.

---

## `modpods.lti_system_gen`

Convert a causative topology and time-series data into an LTI state-space
model.

```python
result = modpods.lti_system_gen(
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
    verbose="warnings",
)
```

### Returns

`dict` with keys:

- `"system"` — `control.StateSpace`.
- `"A"`, `"B"`, `"C"` — `pd.DataFrame` state-space matrices.
- `"topology"` — `pd.DataFrame` adjacency matrix used.
- `"models"` — dict of trained `delay_io_train` results per output.

---

## `modpods.Verbosity` and `modpods.configure_verbosity`

Logging helpers.  `Verbosity` is the literal type `"warnings" | "info" | "debug"`.

```python
modpods.configure_verbosity("info")
```

---

## Error Metrics

When `evaluation=True`, `delay_io_predict` returns the following metrics
for each output column:

| Metric | Full name | Description |
|--------|-----------|-------------|
| `MAE` | Mean Absolute Error | Average absolute deviation. |
| `RMSE` | Root Mean Squared Error | Square root of mean squared error. |
| `NSE` | Nash-Sutcliffe Efficiency | 1 is perfect, 0 is mean model, negative is worse than mean. |
| `alpha` | | Bias ratio (mean simulated / mean observed). |
| `beta` | | Variance ratio. |
| `HFV` | High Flow Volume | Accuracy during high-flow periods. |
| `HFV10` | High Flow Volume top 10% | Accuracy during the top 10% flows. |
| `LFV` | Low Flow Volume | Accuracy during low-flow periods. |
| `FDC` | Flow Duration Curve | Overall FDC skill. |

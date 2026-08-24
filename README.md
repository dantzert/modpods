# modpods

Model Discovery in Partially Observable Dynamical Systems

modpods discovers governing equations from time-series data using sparse regression with gamma-distribution convolution kernels. It is designed for
practitioners who want to fit interpretable dynamical models to their data with
minimal configuration.

## Installation

```bash
pip install modpods
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add modpods
```

## Quick Start

```python
import numpy as np
import pandas as pd
import modpods

# Load or create your time-series data as a DataFrame
# Columns are variable names; the index is time
data = pd.read_csv("my_data.csv", parse_dates=True, index_col="time")

# Separate dependent (outputs) and independent (inputs/forcing) columns
dependent_columns = ["y1", "y2"]
independent_columns = ["u1", "u2"]

# Train a model: discover equations that explain y1, y2 from u1, u2
model = modpods.delay_io_train(
    system_data=data,
    dependent_columns=dependent_columns,
    independent_columns=independent_columns,
    windup_timesteps=10,
    init_transforms=1,
    max_transforms=2,
    max_iter=250,
    poly_order=2,
    verbose=False,
)

# Predict on new data
prediction = modpods.delay_io_predict(
    model, data, num_transforms=1, evaluation=True
)

# Inspect error metrics
print(prediction["error_metrics"])
```

## Functionality Overview

### `delay_io_train`

Train a dynamical model from time-series data. The function:

1. Applies gamma-distribution convolution transforms to input channels to capture
   delayed causation.
2. Uses sparse regression to discover
   governing equations in the form `ẋ = f(x, u)`.
3. Supports constrained optimization (e.g., enforcing that certain coefficients
   are negative or positive).
4. Returns a dictionary of trained models keyed by the number of transforms.

### `delay_io_predict`

Simulate a trained model on new data and compute error metrics (MAE, RMSE, NSE,
alpha, beta, HFV, HFV10, LFV, FDC).

### `transform_inputs`

Apply gamma-distribution convolution to forcing inputs. Useful as a standalone
preprocessing step.

### `infer_causative_topology`

Discover which input variables causally influence which output variables from
data alone. Returns an adjacency matrix and transformation parameters.

### `lti_system_gen`

Convert a causative topology and time-series data into a linear time-invariant
(LTI) state-space model suitable for control design.

### `lti_from_gamma`

Generate an LTI system whose impulse response matches a given gamma distribution.

## Citation

Original paper is https://doi.org/10.1016/j.advwatres.2024.104796

"""End-to-end quickstart example for modpods.

Generates a small synthetic LTI dataset, trains a delay-IO model, runs
prediction, and prints error metrics. Designed to run in CI without
external data files.
"""

import control as ct  # type: ignore
import numpy as np
import pandas as pd

import modpods


def main() -> None:
    np.random.seed(0)
    dt = 0.05
    n = 400
    t = np.arange(0, n * dt, dt)

    A = np.array([[-1.0, 0.0], [1.0, -1.0]])
    B = np.array([[1.0], [0.0]])
    sys = ct.ss(A, B, np.eye(2), 0)
    u = np.zeros((n, 1))
    u[50:80, 0] = np.random.rand(30)
    response = ct.forced_response(sys, t, np.transpose(u))

    data = pd.DataFrame(
        index=t,
        data={
            "u": response.inputs[0],
            "x0": response.states[0],
            "x1": response.states[1],
        },
    )

    model = modpods.delay_io_train(
        data,
        dependent_columns=["x1"],
        independent_columns=["u"],
        windup_timesteps=0,
        init_transforms=1,
        max_transforms=1,
        max_iter=5,
        poly_order=1,
        verbose="warnings",
    )

    pred = modpods.delay_io_predict(
        model, data, num_transforms=1, evaluation=True
    )

    print("Prediction shape:", pred["prediction"].shape)
    print("Error metrics:", pred["error_metrics"])


if __name__ == "__main__":
    main()

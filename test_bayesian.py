import numpy as np
import pandas as pd

import modpods

# Create a simple test case
np.random.seed(42)

# Simulate some simple time series data
n_samples = 200
t = np.arange(n_samples)

# Simple system: output depends on delayed and transformed input
input_signal = np.random.randn(n_samples) * 0.5 + np.sin(t * 0.1)
delayed_input = np.concatenate([np.zeros(5), input_signal[:-5]])  # 5-step delay
output_signal = (
    0.7 * delayed_input
    + 0.3 * np.roll(delayed_input, 3)
    + 0.1 * np.random.randn(n_samples)
)

# Create DataFrame
test_data = pd.DataFrame({"input": input_signal, "output": output_signal})

# Test with minimal parameters
print("Testing all optimization methods with minimal example...")
try:
    methods = [
        ("Bayesian", "bayesian", 15),
        ("Differential Evolution", "differential_evolution", 20),
        ("Dual Annealing", "dual_annealing", 20),
    ]

    models = {}
    for name, method, max_iter in methods:
        print(f"\n=== Testing {name} ===")
        model = modpods.delay_io_train(
            test_data,
            ["output"],
            ["input"],
            windup_timesteps=10,
            init_transforms=1,
            max_transforms=1,
            max_iter=max_iter,
            verbose=True,
            poly_order=1,
            optimization_method=method,
        )
        models[name] = model
        print(f"{name} completed successfully!")
        print(f"R² = {model[1]['final_model']['error_metrics']['r2']:.6f}")

    print("\n=== Comparison ===")
    for name, model in models.items():
        r2 = model[1]['final_model']['error_metrics']['r2']
        print(f"{name:25s} R²: {r2:.6f}")

    # Compare best vs worst
    r2_values = {name: model[1]["final_model"]["error_metrics"]["r2"] for name, model in models.items()}
    best_name = max(r2_values, key=r2_values.get)
    worst_name = min(r2_values, key=r2_values.get)
    improvement = r2_values[best_name] - r2_values[worst_name]
    print(f"Best method: {best_name} (R² = {r2_values[best_name]:.6f})")
    print(f"Worst method: {worst_name} (R² = {r2_values[worst_name]:.6f})")
    print(f"Absolute improvement: {improvement:.6f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

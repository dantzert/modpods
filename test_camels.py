import pandas as pd

import modpods

# Test with the original CAMELS dataset
print("Testing Bayesian optimization with CAMELS dataset...")

# Load the original dataset
filepath = "./tests/data/03439000_05_model_output.txt"
df = pd.read_csv(filepath, sep=r"\s+")
print("Data loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Prepare data as in original test
df.rename(
    {"YR": "year", "MNTH": "month", "DY": "day", "HR": "hour"}, axis=1, inplace=True
)
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df.set_index("datetime", inplace=True)

# Shift forcing to make system causal
df.RAIM = df.RAIM.shift(-1)
df.dropna(inplace=True)

# Use subset for testing
windup_timesteps = 30
years = 1
df_train = df.iloc[: 365 * years + windup_timesteps, :]

# Test all methods on real data
forcing_coef_constraints = {"RAIM": -1, "PET": 1, "PRCP": -1}
df_train = df_train[["OBS_RUN", "RAIM", "PET", "PRCP"]]

print(f"\nTraining data shape: {df_train.shape}")
print("Training all optimization methods...")

try:
    methods = [
        ("Bayesian", "bayesian", 25),
        ("Differential Evolution", "differential_evolution", 30),
        ("Dual Annealing", "dual_annealing", 30),
    ]

    models = {}
    for name, method, max_iter in methods:
        print(f"\n=== {name} on CAMELS Data ===")
        model = modpods.delay_io_train(
            df_train,
            ["OBS_RUN"],
            ["RAIM", "PET", "PRCP"],
            windup_timesteps=windup_timesteps,
            init_transforms=1,
            max_transforms=1,
            max_iter=max_iter,
            verbose=False,
            forcing_coef_constraints=forcing_coef_constraints,
            poly_order=1,
            bibo_stable=False,
            optimization_method=method,
        )
        models[name] = model
        r2 = model[1]["final_model"]["error_metrics"]["r2"]
        print(f"{name} R² = {r2:.6f}")

    # Results
    print("\n=== CAMELS Dataset Results ===")
    for name, model in models.items():
        r2 = model[1]["final_model"]["error_metrics"]["r2"]
        print(f"{name:25s} R²: {r2:.6f}")

    # Compare best vs worst
    r2_values = {name: model[1]["final_model"]["error_metrics"]["r2"] for name, model in models.items()}
    best_name = max(r2_values, key=r2_values.get)
    worst_name = min(r2_values, key=r2_values.get)
    improvement = r2_values[best_name] - r2_values[worst_name]
    pct_improvement = (improvement / r2_values[worst_name]) * 100 if r2_values[worst_name] > 0 else 0
    print(f"Best method: {best_name} (R² = {r2_values[best_name]:.6f})")
    print(f"Worst method: {worst_name} (R² = {r2_values[worst_name]:.6f})")
    print(f"Absolute improvement: {improvement:.6f}")
    print(f"Percent improvement:  {pct_improvement:.1f}%")

    if improvement > 0:
        print(f"✓ {best_name} found a better solution!")
    else:
        print(f"→ {worst_name} performed better on this dataset")

    print("\n=== Parameter Comparison ===")
    for name, model in models.items():
        print(f"{name} factors:")
        print(f"  Shape: {model[1]['shape_factors'].iloc[0,0]:.3f}")
        print(f"  Scale: {model[1]['scale_factors'].iloc[0,0]:.3f}")
        print(f"  Location: {model[1]['loc_factors'].iloc[0,0]:.3f}")

    print("\n=== SUCCESS: All methods completed successfully! ===")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()

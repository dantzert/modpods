import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modpods

# Create a simple test case
np.random.seed(42)

# Simulate some simple time series data
n_samples = 200
t = np.arange(n_samples)

# Simple system: output depends on delayed and transformed input
input_signal = np.random.randn(n_samples) * 0.5 + np.sin(t * 0.1)
delayed_input = np.concatenate([np.zeros(5), input_signal[:-5]])  # 5-step delay
output_signal = 0.7 * delayed_input + 0.3 * np.roll(delayed_input, 3) + 0.1 * np.random.randn(n_samples)

# Create DataFrame
test_data = pd.DataFrame({
    'input': input_signal,
    'output': output_signal
})

# Test integrated Bayesian optimization
print("Testing integrated Bayesian optimization in delay_io_train...")
try:
    # Test compass search first (default)
    print("\n=== Testing Compass Search ===")
    model_compass = modpods.delay_io_train(
        test_data, ['output'], ['input'],
        windup_timesteps=10, init_transforms=1, max_transforms=1,
        max_iter=5, verbose=True, poly_order=1,
        optimization_method="compass_search"
    )
    print("Compass search completed successfully!")
    print(f"R² = {model_compass[1]['final_model']['error_metrics']['r2']:.6f}")
    
    # Test integrated Bayesian optimization
    print("\n=== Testing Integrated Bayesian Optimization ===")
    model_bayesian = modpods.delay_io_train(
        test_data, ['output'], ['input'],
        windup_timesteps=10, init_transforms=1, max_transforms=1,
        max_iter=15, verbose=True, poly_order=1,
        optimization_method="bayesian"
    )
    print("Bayesian optimization completed successfully!")
    print(f"R² = {model_bayesian[1]['final_model']['error_metrics']['r2']:.6f}")
    
    print("\n=== Comparison ===")
    print(f"Compass search R²: {model_compass[1]['final_model']['error_metrics']['r2']:.6f}")
    print(f"Bayesian opt R²:   {model_bayesian[1]['final_model']['error_metrics']['r2']:.6f}")
    
    improvement = model_bayesian[1]['final_model']['error_metrics']['r2'] - model_compass[1]['final_model']['error_metrics']['r2']
    print(f"Improvement:       {improvement:.6f}")
    
    print("\n=== SUCCESS: Bayesian optimization integrated successfully! ===")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
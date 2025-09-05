import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modpods

# Test with the original CAMELS dataset
print("Testing Bayesian optimization with CAMELS dataset...")

# Load the original dataset
filepath = "./03439000_05_model_output.txt"
df = pd.read_csv(filepath, sep=r'\s+')
print("Data loaded successfully!")
print(f"Dataset shape: {df.shape}")

# Prepare data as in original test
df.rename({'YR':'year','MNTH':'month','DY':'day','HR':'hour'},axis=1,inplace=True)
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
df.set_index('datetime',inplace=True)

# Shift forcing to make system causal
df.RAIM = df.RAIM.shift(-1)
df.dropna(inplace=True)

# Use subset for testing
windup_timesteps = 30
years = 1
df_train = df.iloc[:365*years + windup_timesteps,:]

# Test both methods on real data
forcing_coef_constraints = {'RAIM':-1, 'PET':1,'PRCP':-1}
df_train = df_train[['OBS_RUN','RAIM','PET','PRCP']]

print(f"\nTraining data shape: {df_train.shape}")
print("Training both optimization methods...")

try:
    # Compass search
    print("\n=== Compass Search on CAMELS Data ===")
    model_compass = modpods.delay_io_train(
        df_train, ['OBS_RUN'], ['RAIM','PET','PRCP'],
        windup_timesteps=windup_timesteps,
        init_transforms=1, max_transforms=1, max_iter=20,
        verbose=False, forcing_coef_constraints=forcing_coef_constraints,
        poly_order=1, bibo_stable=False,
        optimization_method="compass_search"
    )
    compass_r2 = model_compass[1]['final_model']['error_metrics']['r2']
    print(f"Compass search R² = {compass_r2:.6f}")
    
    # Bayesian optimization
    print("\n=== Bayesian Optimization on CAMELS Data ===")
    model_bayesian = modpods.delay_io_train(
        df_train, ['OBS_RUN'], ['RAIM','PET','PRCP'],
        windup_timesteps=windup_timesteps,
        init_transforms=1, max_transforms=1, max_iter=25,
        verbose=False, forcing_coef_constraints=forcing_coef_constraints,
        poly_order=1, bibo_stable=False,
        optimization_method="bayesian"
    )
    bayesian_r2 = model_bayesian[1]['final_model']['error_metrics']['r2']
    print(f"Bayesian optimization R² = {bayesian_r2:.6f}")
    
    # Results
    improvement = bayesian_r2 - compass_r2
    pct_improvement = (improvement / compass_r2) * 100 if compass_r2 > 0 else 0
    
    print(f"\n=== CAMELS Dataset Results ===")
    print(f"Compass search R²:    {compass_r2:.6f}")
    print(f"Bayesian opt R²:      {bayesian_r2:.6f}")
    print(f"Absolute improvement: {improvement:.6f}")
    print(f"Percent improvement:  {pct_improvement:.1f}%")
    
    if improvement > 0:
        print("✓ Bayesian optimization found a better solution!")
    else:
        print("→ Compass search performed better on this dataset")
        
    print("\n=== Parameter Comparison ===")
    print("Compass search factors:")
    print(f"  Shape: {model_compass[1]['shape_factors'].iloc[0,0]:.3f}")
    print(f"  Scale: {model_compass[1]['scale_factors'].iloc[0,0]:.3f}")
    print(f"  Location: {model_compass[1]['loc_factors'].iloc[0,0]:.3f}")
    
    print("Bayesian optimization factors:")
    print(f"  Shape: {model_bayesian[1]['shape_factors'].iloc[0,0]:.3f}")
    print(f"  Scale: {model_bayesian[1]['scale_factors'].iloc[0,0]:.3f}")
    print(f"  Location: {model_bayesian[1]['loc_factors'].iloc[0,0]:.3f}")
    
    print("\n=== SUCCESS: Both methods completed successfully! ===")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
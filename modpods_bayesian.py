import pandas as pd
import numpy as np
import pysindy as ps
import scipy.stats as stats
from scipy import signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import control as control
import networkx as nx
import sys
try:
    import pyswmm # not a requirement for any other function
except ImportError:
    pyswmm = None
import re
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings

# Import original modpods functions
from modpods import *

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Computes the Expected Improvement at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)
    
    mu_sample_opt = np.max(Y_sample)
    
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    """
    Proposes the next sampling point by optimizing the acquisition function.
    """
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()
    
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
            
    return min_x.reshape(-1, 1)

def delay_io_train_bayesian(system_data, dependent_columns, independent_columns, 
                           windup_timesteps=0, init_transforms=1, max_transforms=4, 
                           max_iter=50, poly_order=3, transform_dependent=False, 
                           verbose=False, extra_verbose=False, include_bias=False, 
                           include_interaction=False, bibo_stable=False,
                           transform_only=None, forcing_coef_constraints=None,
                           early_stopping_threshold=0.005):
    """
    Bayesian optimization version of delay_io_train function.
    """
    forcing = system_data[independent_columns].copy(deep=True)
    orig_forcing_columns = forcing.columns
    response = system_data[dependent_columns].copy(deep=True)
    results = dict()

    # Determine which columns to transform
    if transform_dependent:
        transform_columns = system_data.columns.tolist()
    elif transform_only is not None:
        transform_columns = transform_only
    else:
        transform_columns = independent_columns
    
    for num_transforms in range(init_transforms, max_transforms + 1):
        print(f"num_transforms: {num_transforms}")
        
        # Define parameter bounds for this number of transforms
        # Parameters: [shape1, scale1, loc1, shape2, scale2, loc2, ...]
        n_params = len(transform_columns) * num_transforms * 3
        bounds = []
        for transform in range(1, num_transforms + 1):
            for col in transform_columns:
                bounds.append([1.0, 50.0])    # shape_factors bounds
                bounds.append([0.1, 5.0])     # scale_factors bounds  
                bounds.append([0.0, 20.0])    # loc_factors bounds
        bounds = np.array(bounds)
        
        def objective_function(params_vector):
            """Objective function that takes parameter vector and returns R²"""
            try:
                # Convert vector to DataFrames
                shape_factors = pd.DataFrame(columns=transform_columns, index=range(1, num_transforms + 1))
                scale_factors = pd.DataFrame(columns=transform_columns, index=range(1, num_transforms + 1))
                loc_factors = pd.DataFrame(columns=transform_columns, index=range(1, num_transforms + 1))
                
                idx = 0
                for transform in range(1, num_transforms + 1):
                    for col in transform_columns:
                        shape_factors.loc[transform, col] = params_vector[idx]
                        scale_factors.loc[transform, col] = params_vector[idx + 1]
                        loc_factors.loc[transform, col] = params_vector[idx + 2]
                        idx += 3
                
                # Evaluate using SINDY_delays_MI
                result = SINDY_delays_MI(shape_factors, scale_factors, loc_factors,
                                       system_data.index, forcing, response, False, 
                                       poly_order, include_bias, include_interaction,
                                       windup_timesteps, bibo_stable, transform_dependent,
                                       transform_only, forcing_coef_constraints)
                
                r2 = result['error_metrics']['r2']
                if verbose:
                    print(f"  R² = {r2:.6f}")
                return r2
            except Exception as e:
                if verbose:
                    print(f"  Evaluation failed: {e}")
                return -1.0  # Poor score for failed evaluations
        
        # Bayesian optimization
        n_initial = min(10, max(5, max_iter // 4))
        X_sample = []
        Y_sample = []
        
        if verbose:
            print(f"Starting Bayesian optimization with {n_initial} initial samples...")
        
        # Generate initial random samples
        for i in range(n_initial):
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            y = objective_function(x)
            X_sample.append(x)
            Y_sample.append(y)
            if verbose:
                print(f"  Initial sample {i+1}/{n_initial}: R² = {y:.6f}")
        
        X_sample = np.array(X_sample)
        Y_sample = np.array(Y_sample).reshape(-1, 1)
        
        # Main Bayesian optimization loop
        best_r2 = np.max(Y_sample)
        best_params = X_sample[np.argmax(Y_sample)]
        
        # Gaussian Process setup
        kernel = Matern(length_scale=1.0, nu=2.5)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, 
                                     n_restarts_optimizer=5, random_state=42)
        
        for iteration in range(max_iter - n_initial):
            # Fit GP and find next point
            gpr.fit(X_sample, Y_sample.ravel())
            next_x = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)
            next_x = next_x.flatten()
            
            # Evaluate objective
            next_y = objective_function(next_x)
            
            if verbose:
                print(f"  BO iteration {iteration+1}/{max_iter-n_initial}: R² = {next_y:.6f}")
            
            # Update samples
            X_sample = np.append(X_sample, [next_x], axis=0)
            Y_sample = np.append(Y_sample, next_y)
            
            # Update best
            if next_y > best_r2:
                best_r2 = next_y
                best_params = next_x
                if verbose:
                    print(f"    New best R² = {best_r2:.6f}")
        
        # Convert best parameters back to DataFrames
        shape_factors = pd.DataFrame(columns=transform_columns, index=range(1, num_transforms + 1))
        scale_factors = pd.DataFrame(columns=transform_columns, index=range(1, num_transforms + 1))
        loc_factors = pd.DataFrame(columns=transform_columns, index=range(1, num_transforms + 1))
        
        idx = 0
        for transform in range(1, num_transforms + 1):
            for col in transform_columns:
                shape_factors.loc[transform, col] = best_params[idx]
                scale_factors.loc[transform, col] = best_params[idx + 1]
                loc_factors.loc[transform, col] = best_params[idx + 2]
                idx += 3
        
        # Final evaluation
        final_model = SINDY_delays_MI(shape_factors, scale_factors, loc_factors,
                                     system_data.index, forcing, response, True, 
                                     poly_order, include_bias, include_interaction,
                                     windup_timesteps, bibo_stable, transform_dependent,
                                     transform_only, forcing_coef_constraints)
        
        print(f"\nFinal model for {num_transforms} transforms:")
        try:
            print(final_model['model'].print(precision=5))
        except Exception as e:
            print(e)
        print(f"R² = {final_model['error_metrics']['r2']:.6f}")
        print("Shape factors:")
        print(shape_factors)
        print("Scale factors:")
        print(scale_factors)
        print("Location factors:")
        print(loc_factors)
        print()
        
        # Store results
        results[num_transforms] = {
            'final_model': final_model.copy(),
            'shape_factors': shape_factors.copy(deep=True),
            'scale_factors': scale_factors.copy(deep=True),
            'loc_factors': loc_factors.copy(deep=True),
            'windup_timesteps': windup_timesteps,
            'dependent_columns': dependent_columns,
            'independent_columns': independent_columns
        }
        
        # Early stopping check
        if (num_transforms > init_transforms and 
            results[num_transforms]['final_model']['error_metrics']['r2'] - 
            results[num_transforms-1]['final_model']['error_metrics']['r2'] < early_stopping_threshold):
            print(f"Last transformation added less than {early_stopping_threshold*100}% to R² score. Terminating early.")
            break
    
    return results
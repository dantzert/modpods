import pandas as pd
import numpy as np
import pysindy as ps
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt
import control as control
import networkx as nx
import sys
import pyswmm # not a requirement for any other function
import re

# delay model builds differential equations relating the dependent variables to transformations of all the variables
# if there are no independent variables, then dependent_columns should be a list of all the columns in the dataframe
# and independent_columns should be an empty list
# by default, only the independent variables are transformed, but if transform_dependent is set to True, then the dependent variables are also transformed
# REQUIRES: 
# a pandas dataframe, 
# the column names of the dependent and indepdent variables, 
# the number of timesteps to "wind up" the latent states,
# the initial number of transformations to use in the optimization,
# the maximum number of transformations to use in the optimization,
# the maximum number of iterations to use in the optimization
# and the order of the polynomial to use in the optimization
# bibo_stable: if true, the highest order output autocorrelation term is constrained to be negative
# RETURNS:
# models for each number of transformations from min to max
# NOTE: this code works for MIMO models, however, if output variables are dependent on each other
# poor simulation fidelity is likely due to their errors contributing to each other
# if the learned dynamics are highly accurate such that errors do not grow too large in any dependent variable, a MIMO model should work fine
# if you anticipate significant errors in the simulation of any dependent variable, you should use multiple MISO models instead
# as the model predicts derivatives, system_data must represent a *causal* system
# that is, forcing and the response to that forcing cannot occur at the same timestep
# it may be necessary for the user to shift the forcing data back to make the system causal (especially for time aggregated data like daily rainfall-runoff)
# forcing_coef_constraints is a dictionary of column name and then a 1, 0, or -1 depending on whether the coefficients of that variable should be positive, unconstrained, or negative
def delay_io_train(system_data, dependent_columns, independent_columns, 
                   windup_timesteps=0,init_transforms=1, max_transforms=4, 
                   max_iter=250, poly_order=3, transform_dependent=False, 
                   verbose=False, extra_verbose=False, include_bias=False, 
                   include_interaction=False, bibo_stable = False,
                   transform_only = None, forcing_coef_constraints=None):
    forcing = system_data[independent_columns].copy(deep=True)

    orig_forcing_columns = forcing.columns
    response = system_data[dependent_columns].copy(deep=True)

    results = dict() # to store the optimized models for each number of transformations

    if transform_dependent:
        shape_factors = pd.DataFrame(columns = system_data.columns, index = range(init_transforms, max_transforms+1))
        shape_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
        scale_factors = pd.DataFrame(columns = system_data.columns, index = range(init_transforms, max_transforms+1))
        scale_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
        loc_factors = pd.DataFrame(columns = system_data.columns, index = range(init_transforms, max_transforms+1))
        loc_factors.iloc[0,:] = 0 # first transformation is [1,1,0] for each input
    elif transform_only is not None: # the user provided a list of columns to transform
        shape_factors = pd.DataFrame(columns = transform_only, index = range(init_transforms, max_transforms+1))
        shape_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
        scale_factors = pd.DataFrame(columns = transform_only, index = range(init_transforms, max_transforms+1))
        scale_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
        loc_factors = pd.DataFrame(columns = transform_only, index = range(init_transforms, max_transforms+1))
        loc_factors.iloc[0,:] = 0 # first transformation is [1,1,0] for each input
    else:
        # the transformation factors should be pandas dataframes where the index is which transformation it is and the columns are the variables
        shape_factors = pd.DataFrame(columns = forcing.columns, index = range(init_transforms, max_transforms+1))
        shape_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
        scale_factors = pd.DataFrame(columns = forcing.columns, index = range(init_transforms, max_transforms+1))
        scale_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
        loc_factors = pd.DataFrame(columns = forcing.columns, index = range(init_transforms, max_transforms+1))
        loc_factors.iloc[0,:] = 0 # first transformation is [1,1,0] for each input
    #print(shape_factors)
    #print(scale_factors)
    #print(loc_factors)
    # first transformation is [1,1,0] for each input
    '''
    shape_factors = np.ones(shape=(forcing.shape[1] , init_transforms)   )
    scale_factors = np.ones(shape=(forcing.shape[1] , init_transforms)   )
    loc_factors = np.zeros(shape=(forcing.shape[1] , init_transforms)   )
    '''
    #speeds =  list([500,200,50,10, 5,2, 1.1, 1.05,1.01]) 
    speeds = list([100,50,20,10,5,2,1.1,1.05,1.01]) # I don't have a great idea of what good values for these are yet
    if transform_dependent: # just trying something
        improvement_threshold = 1.001 # when improvements are tiny, tighten up the jumps
    else:
        improvement_threshold = 1.0

    for num_transforms in range(init_transforms,max_transforms + 1):
        print("num_transforms")
        print(num_transforms)
        speed_idx = 0
        speed = speeds[speed_idx]

        if (not num_transforms == init_transforms):  # if we're not starting right now
            # start dull
            shape_factors.iloc[num_transforms-1,:] = 10*(num_transforms-1) # start with a broad peak centered at ten timesteps
            scale_factors.iloc[num_transforms-1,:] = 1
            loc_factors.iloc[num_transforms-1,:] = 0
            if verbose:
                print("starting factors for additional transformation\nshape\nscale\nlocation")
                print(shape_factors)
                print(scale_factors)
                print(loc_factors)



        prev_model = SINDY_delays_MI(shape_factors, scale_factors, loc_factors, system_data.index, 
                                     forcing, response,extra_verbose, poly_order , include_bias, 
                                     include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)

        print("\nInitial model:\n")
        try:
            print(prev_model['model'].print(precision=5))
            print("R^2")
            print(prev_model['error_metrics']['r2'])
        except Exception as e: # and print the exception:
            print(e)
            pass
        print("shape factors")
        print(shape_factors)
        print("scale factors")
        print(scale_factors)
        print("location factors")
        print(loc_factors)
        print("\n")

        if not verbose:
            print("training ", end='')

        #no_improvement_last_time = False
        for iterations in range(0,max_iter ):
            if not verbose and iterations % 5 == 0:
                print(str(iterations)+".", end='')

            if transform_dependent:
                tuning_input = system_data.columns[(iterations // num_transforms) %  len(system_data.columns)] # row =  iter // width % height]
            elif transform_only is not None:
                tuning_input = transform_only[(iterations // num_transforms) %  len(transform_only)]
            else:
                tuning_input = orig_forcing_columns[(iterations // num_transforms) %  len(orig_forcing_columns)] # row =  iter // width % height
            tuning_line = iterations % num_transforms + 1 # col =  % width (plus one because there's no zeroth transformation)
            if verbose:
                print(str("tuning input: {i} | tuning transformation: {l:g}".format(i=tuning_input,l=tuning_line)))


            sooner_locs = loc_factors.copy(deep=True)
            sooner_locs[tuning_input][tuning_line] = float(loc_factors[tuning_input][tuning_line] - speed/10  )
            if ( sooner_locs[tuning_input][tuning_line] < 0):
                sooner = {'error_metrics':{'r2':-1}}
            else:
                sooner = SINDY_delays_MI(shape_factors ,scale_factors ,sooner_locs, 
                system_data.index, forcing, response, extra_verbose, poly_order , 
                include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)
      
      
            later_locs = loc_factors.copy(deep=True)
            later_locs[tuning_input][tuning_line] = float ( loc_factors[tuning_input][tuning_line]  +   1.01*speed/10 )
            later = SINDY_delays_MI(shape_factors , scale_factors,later_locs, 
                system_data.index, forcing, response, extra_verbose, poly_order , 
                include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)
      

            shape_up = shape_factors.copy(deep=True)
            shape_up[tuning_input][tuning_line] = float ( shape_factors[tuning_input][tuning_line]*speed*1.01 )
            shape_upped = SINDY_delays_MI(shape_up , scale_factors, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , 
                                    include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)
      
            shape_down = shape_factors.copy(deep=True)
            shape_down[tuning_input][tuning_line] = float ( shape_factors[tuning_input][tuning_line]/speed )
            if (shape_down[tuning_input][tuning_line] < 1):
                shape_downed = {'error_metrics':{'r2':-1}} # return a score of negative one as this is illegal
            else:
                shape_downed = SINDY_delays_MI(shape_down , scale_factors, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , 
                                    include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)

            scale_up = scale_factors.copy(deep=True)
            scale_up[tuning_input][tuning_line] = float(  scale_factors[tuning_input][tuning_line]*speed*1.01 )
            scaled_up = SINDY_delays_MI(shape_factors , scale_up, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , 
                                    include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)


            scale_down = scale_factors.copy(deep=True)
            scale_down[tuning_input][tuning_line] = float ( scale_factors[tuning_input][tuning_line]/speed )
            scaled_down = SINDY_delays_MI(shape_factors , scale_down, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , 
                                    include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)
      
            # rounder
            rounder_shape = shape_factors.copy(deep=True)
            rounder_shape[tuning_input][tuning_line] = shape_factors[tuning_input][tuning_line]*(speed*1.01)
            rounder_scale = scale_factors.copy(deep=True)
            rounder_scale[tuning_input][tuning_line] = scale_factors[tuning_input][tuning_line]/(speed*1.01)
            rounder = SINDY_delays_MI(rounder_shape , rounder_scale, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , 
                                    include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)

            # sharper
            sharper_shape = shape_factors.copy(deep=True)
            sharper_shape[tuning_input][tuning_line] = shape_factors[tuning_input][tuning_line]/speed
            if (sharper_shape[tuning_input][tuning_line] < 1):
                sharper = {'error_metrics':{'r2':-1}} # lower bound on shape to avoid inf
            else:
                sharper_scale = scale_factors.copy(deep=True)
                sharper_scale[tuning_input][tuning_line] = scale_factors[tuning_input][tuning_line]*speed
                sharper = SINDY_delays_MI(sharper_shape ,sharper_scale,loc_factors, 
                                            system_data.index, forcing, response, extra_verbose, poly_order , 
                                            include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)


    

            scores = [prev_model['error_metrics']['r2'], shape_upped['error_metrics']['r2'], shape_downed['error_metrics']['r2'], 
                      scaled_up['error_metrics']['r2'], scaled_down['error_metrics']['r2'], sooner['error_metrics']['r2'], 
                      later['error_metrics']['r2'], rounder['error_metrics']['r2'], sharper['error_metrics']['r2'] ]
            #print(scores)

            if (sooner['error_metrics']['r2'] >= max(scores) and sooner['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = sooner.copy()
                loc_factors = sooner_locs.copy(deep=True)
            elif (later['error_metrics']['r2'] >= max(scores) and later['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = later.copy()
                loc_factors = later_locs.copy(deep=True)
            elif(shape_upped['error_metrics']['r2'] >= max(scores) and shape_upped['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = shape_upped.copy()
                shape_factors = shape_up.copy(deep=True)
            elif(shape_downed['error_metrics']['r2'] >=max(scores) and shape_downed['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = shape_downed.copy()
                shape_factors = shape_down.copy(deep=True)
            elif(scaled_up['error_metrics']['r2'] >= max(scores) and scaled_up['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = scaled_up.copy()
                scale_factors = scale_up.copy(deep=True)
            elif(scaled_down['error_metrics']['r2'] >= max(scores) and scaled_down['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = scaled_down.copy()
                scale_factors = scale_down.copy(deep=True)
            elif (rounder['error_metrics']['r2'] >= max(scores) and rounder['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = rounder.copy()
                shape_factors = rounder_shape.copy(deep=True)
                scale_factors = rounder_scale.copy(deep=True)
            elif (sharper['error_metrics']['r2'] >= max(scores) and sharper['error_metrics']['r2'] > improvement_threshold*prev_model['error_metrics']['r2']):
                prev_model = sharper.copy()
                shape_factors = sharper_shape.copy(deep=True)
                scale_factors = sharper_scale.copy(deep=True)
            # the middle was best, but it's bad, tighten up the bounds (if we're at the last tuning line of the last input)
            
            elif( num_transforms == tuning_line and tuning_input == shape_factors.columns[-1]): # no improvement transforming last column
                #no_improvement_last_time=True
                speed_idx = speed_idx + 1
                if verbose:
                    print("\n\ntightening bounds\n\n")
                '''
            elif (num_transforms == tuning_line and tuning_input == orig_forcing_columns[0] and no_improvement_last_time): # no improvement next iteration (first column)
                speed_idx = speed_idx + 1
                no_improvement_last_time=False
                if verbose:
                    print("\n\ntightening bounds\n\n")
                    '''

            if (speed_idx >= len(speeds)):
                print("\n\noptimization complete\n\n")
                break
            speed = speeds[speed_idx]
            if (verbose):
                print("\nprevious, shape up, shape down, scale up, scale down, sooner, later, rounder, sharper")
                print(scores)
                print("speed")
                print(speed)
                print("shape factors")
                print(shape_factors)
                print("scale factors")
                print(scale_factors)
                print("location factors")
                print(loc_factors)
                print("iteration no:")
                print(iterations)
                print("model")
                try:
                    prev_model['model'].print(precision=5)
                except Exception as e:
                    print(e)
                print("\n")

    
    
        final_model = SINDY_delays_MI(shape_factors, scale_factors ,loc_factors,system_data.index, forcing, response, True, poly_order , 
                                      include_bias, include_interaction,windup_timesteps,bibo_stable,transform_dependent=transform_dependent,
                                     transform_only=transform_only,forcing_coef_constraints=forcing_coef_constraints)
        print("\nFinal model:\n")
        try:
            print(final_model['model'].print(precision=5))
        except Exception as e:
            print(e)
        print("R^2")
        print(prev_model['error_metrics']['r2'])
        print("shape factors")
        print(shape_factors)
        print("scale factors")
        print(scale_factors)
        print("location factors")
        print(loc_factors)
        print("\n")
        results[num_transforms] = {'final_model':final_model.copy(), 
                                   'shape_factors':shape_factors.copy(deep=True), 
                                   'scale_factors':scale_factors.copy(deep=True), 
                                   'loc_factors':loc_factors.copy(deep=True),
                                   'windup_timesteps':windup_timesteps,
                                   'dependent_columns':dependent_columns,
                                   'independent_columns':independent_columns}
    return results


def SINDY_delays_MI(shape_factors, scale_factors, loc_factors, index, forcing, response, final_run, 
                    poly_degree, include_bias, include_interaction,windup_timesteps,bibo_stable=False,
                    transform_dependent=False,transform_only=None, forcing_coef_constraints=None):
    if transform_only is not None:
        transformed_forcing = transform_inputs(shape_factors, scale_factors,loc_factors, index, forcing.loc[:,transform_only])
        untransformed_forcing = forcing.drop(columns=transform_only)
        # combine forcing and transformed forcing column-wise
        forcing = pd.concat((untransformed_forcing,transformed_forcing),axis='columns')
    else:
        forcing = transform_inputs(shape_factors, scale_factors,loc_factors, index, forcing)

    feature_names =  response.columns.tolist() +  forcing.columns.tolist()

    # SINDy
    if (not bibo_stable): # no constraints, normal mode
        model = ps.SINDy(
            differentiation_method= ps.FiniteDifference(),
            feature_library=ps.PolynomialLibrary(degree=poly_degree,include_bias = include_bias, include_interaction=include_interaction), 
            optimizer = ps.STLSQ(threshold=0), 
            feature_names = feature_names
        )
    if (bibo_stable): # highest order output autocorrelation is constrained to be negative
        #import cvxpy
        #run_cvxpy= True
        # Figure out how many library features there will be
        library = ps.PolynomialLibrary(degree=poly_degree,include_bias = include_bias, include_interaction=include_interaction)
        total_train = pd.concat((response,forcing), axis='columns')
        library.fit([ps.AxesArray(total_train,{"ax_sample":0,"ax_coord":1})])
        n_features = library.n_output_features_
        #print(f"Features ({n_features}):", library.get_feature_names())
        # Set constraints
        n_targets = total_train.shape[1] # not sure what targets means after reading through the pysindy docs
        #print("n_targets")
        #print(n_targets)
        constraint_rhs = np.zeros((len(response.columns),1))
        # one row per constraint, one column per coefficient
        constraint_lhs = np.zeros((len(response.columns) , n_features ))

        #print(constraint_rhs)
        #print(constraint_lhs)
        # constrain the highest order output autocorrelation to be negative
        # this indexing is only right for include_interaction=False, include_bias=False, and pure polynomial library
        # for more complex libraries, some conditional logic will be needed to grab the right column
        constraint_lhs[:,-len(forcing.columns)-len(response.columns):-len(forcing.columns)] = 1
        # leq 0
        #print("constraint lhs")
        #print(constraint_lhs)

        # forcing_coef_constraints not actually implemented yet
        #if forcing_coef_constraints is not None:
        if False:
            constraint_rhs = np.zeros((n_features,)) # every feature is constrained
            # one row per constraint, one column per coefficient
            constraint_lhs = np.zeros((n_features , n_targets*n_features ) )
            # bibo stability, set the highest order output autocorrelation to be negative
            constraint_lhs[:n_targets,-len(forcing.columns)-len(response.columns):-len(forcing.columns)] = 1

            print(forcing.columns)
            forcing_constraints_array = np.ndarray(shape=(1,len(forcing.columns)))
            for i, col in enumerate(forcing.columns):
                if col in forcing_coef_constraints.keys(): # invert the sign because the eqn is written as "leq 0"
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[col]
                elif str(col).replace('_tr_1','') in forcing_coef_constraints.keys():
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[str(col).replace('_tr_1','')]
                elif str(col).replace('_tr_2','') in forcing_coef_constraints.keys():
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[str(col).replace('_tr_2','')]
                elif str(col).replace('_tr_3','') in forcing_coef_constraints.keys():
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[str(col).replace('_tr_3','')]
                else:
                    forcing_constraints_array[0,i] = 0

            for row in range(n_targets, n_features):
                constraint_lhs[row, row*n_features] = forcing_constraints_array[0,row - n_targets]


            # constrain the highest order output autocorrelation to be negative
            # this indexing is only right for include_interaction=False, include_bias=False, and pure polynomial library
            # for more complex libraries, some conditional logic will be needed to grab the right column
            constraint_lhs[:n_targets,-len(forcing.columns)-len(response.columns):-len(forcing.columns)] = 1

            print(forcing_constraints_array)

            print('constraint lhs')
            print(constraint_lhs)
            print('constraint rhs')
            print(constraint_rhs)


        model = ps.SINDy(
            differentiation_method= ps.FiniteDifference(),
            feature_library=ps.PolynomialLibrary(degree=poly_degree,include_bias = include_bias, include_interaction=include_interaction),
            optimizer = ps.ConstrainedSR3(threshold=0, thresholder = "l2",constraint_lhs=constraint_lhs, constraint_rhs = constraint_rhs, inequality_constraints=True),
            feature_names = feature_names
        )
    if transform_dependent:
        # combine response and forcing into one dataframe
        total_train = pd.concat((response,forcing), axis='columns')
        total_train = transform_inputs(shape_factors, scale_factors,loc_factors, index, total_train)
        # remove the columns in total_train that are already in response (just want to keep the transformed forcing)
        total_train = total_train.drop(columns = response.columns)
        feature_names =  response.columns.tolist() +  total_train.columns.tolist()
        
        # need to add constraints such that variables don't depend on their own past values (but they can have autocorrelations)


        library = ps.PolynomialLibrary(degree=poly_degree,include_bias = include_bias, include_interaction=include_interaction)
        library_terms = pd.concat((total_train,response), axis='columns')
        library.fit([ps.AxesArray(library_terms,{"ax_sample":0,"ax_coord":1})])
        n_features = library.n_output_features_
        #print(f"Features ({n_features}):", library.get_feature_names())
        # Set constraints
        n_targets = response.shape[1] # not sure what targets means after reading through the pysindy docs

        constraint_rhs = np.zeros((n_targets,))
        # one row per constraint, one column per coefficient
        constraint_lhs = np.zeros((n_targets , n_features*n_targets))
        # for bibo stability, starting guess is that each dependent variable is negatively autocorrelated and depends on no other variable
        if bibo_stable:
            initial_guess = np.zeros((n_targets,n_features))
            for idx in range(0,n_targets):
                initial_guess[idx,idx] = -1
        else:
            initial_guess = None
        #print(constraint_rhs)
        #print(constraint_lhs)
        # set the coefficient on a variable's own transformed value to 0
        for idx in range(0,n_targets):
            constraint_lhs[idx,(idx+1)*n_features - n_targets + idx] = 1

        #print("constraint lhs")
        #print(constraint_lhs)

        model = ps.SINDy(
            differentiation_method= ps.FiniteDifference(),
            feature_library=library,
            optimizer = ps.ConstrainedSR3(threshold=0, thresholder = "l0",
                                          nu = 10e9, initial_guess = initial_guess,
                                          constraint_lhs=constraint_lhs, 
                                          constraint_rhs = constraint_rhs, 
                                          inequality_constraints=False,
                                          max_iter=10000),
            feature_names = feature_names
        )

        try:
            # windup latent states (if your windup is too long, this will error)
            model.fit(response.values[windup_timesteps:,:], t = np.arange(0,len(index),1)[windup_timesteps:], u = total_train.values[windup_timesteps:,:])
            r2 = model.score(response.values[windup_timesteps:,:],t=np.arange(0,len(index),1)[windup_timesteps:],u=total_train.values[windup_timesteps:,:]) # training data score
        except Exception as e: # and print the exception
            print("Exception in model fitting, returning r2=-1\n")
            print(e)
            error_metrics = {"MAE":[False],"RMSE":[False],"NSE":[False],"alpha":[False],"beta":[False],"HFV":[False],"HFV10":[False],"LFV":[False],"FDC":[False],"r2":-1}
            return {"error_metrics": error_metrics, "model": None, "simulated": False, "response": response, "forcing": forcing, "index": index,"diverged":False}
        

    else:
        try:
            # windup latent states (if your windup is too long, this will error)
            model.fit(response.values[windup_timesteps:,:], t = np.arange(0,len(index),1)[windup_timesteps:], u = forcing.values[windup_timesteps:,:])
            r2 = model.score(response.values[windup_timesteps:,:],t=np.arange(0,len(index),1)[windup_timesteps:],u=forcing.values[windup_timesteps:,:]) # training data score
        except Exception as e: # and print the exception
            print("Exception in model fitting, returning r2=-1\n")
            print(e)
            error_metrics = {"MAE":[False],"RMSE":[False],"NSE":[False],"alpha":[False],"beta":[False],"HFV":[False],"HFV10":[False],"LFV":[False],"FDC":[False],"r2":-1}
            return {"error_metrics": error_metrics, "model": None, "simulated": False, "response": response, "forcing": forcing, "index": index,"diverged":False}
        # r2 is how well we're doing across all the outputs. that's actually good to keep model accuracy lumped because that's what makes most sense to drive the optimization
    # even though the metrics we'll want to evaluate models on are individual output accuracy
    #print("training R^2", r2)
    #model.print(precision=5)

    # return false for things not evaluated / don't exist
    error_metrics = {"MAE":[False],"RMSE":[False],"NSE":[False],"alpha":[False],"beta":[False],"HFV":[False],"HFV10":[False],"LFV":[False],"FDC":[False],"r2":r2}
    simulated = False
    if (final_run): # only simulate final runs because it's slow
        try: #once in high volume training put this back in, but want to see the errors during development
            if transform_dependent:
                simulated = model.simulate(response.values[windup_timesteps,:],t=np.arange(0,len(index),1)[windup_timesteps:],u=total_train.values[windup_timesteps:,:])
            else:
                simulated = model.simulate(response.values[windup_timesteps,:],t=np.arange(0,len(index),1)[windup_timesteps:],u=forcing.values[windup_timesteps:,:])
            mae = list()
            rmse = list()
            nse = list()
            alpha = list()
            beta = list()
            hfv = list()
            hfv10 = list()
            lfv = list()
            fdc = list()
            for col_idx in range(0,len(response.columns)): # univariate performance metrics
                error = response.values[windup_timesteps+1:,col_idx]-simulated[:,col_idx]

                #print("error")
                #print(error)
                # nash sutcliffe efficiency between response and simulated
                mae.append(np.mean(np.abs(error)))
                rmse.append(np.sqrt(np.mean(error**2 ) ))
                #print("mean measured = ", np.mean(response.values[windup_timesteps+1:,col_idx]  ))
                #print("sum of squared error between measured and model = ", np.sum((error)**2 ))
                #print("sum of squared error between measured and mean of measured = ", np.sum((response.values[windup_timesteps+1:,col_idx]-np.mean(response.values[windup_timesteps+1:,col_idx]  ) )**2 ))
                nse.append(1 - np.sum((error)**2 )  /  np.sum((response.values[windup_timesteps+1:,col_idx]-np.mean(response.values[windup_timesteps+1:,col_idx]  ) )**2 ) )
                alpha.append(np.std(simulated[:,col_idx])/np.std(response.values[windup_timesteps+1:,col_idx]))
                beta.append(np.mean(simulated[:,col_idx])/np.mean(response.values[windup_timesteps+1:,col_idx]))
                hfv.append(100*np.sum(np.sort(simulated[:,col_idx])[-int(0.02*len(index)):]-np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.02*len(index)):])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.02*len(index)):]))
                hfv10.append(100*np.sum(np.sort(simulated[:,col_idx])[-int(0.1*len(index)):]-np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.1*len(index)):])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.1*len(index)):]))
                lfv.append(100*np.sum(np.sort(simulated[:,col_idx])[-int(0.3*len(index)):]-np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.3*len(index)):])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.3*len(index)):]))
                fdc.append(100*(np.log10(np.sort(simulated[:,col_idx])[int(0.2*len(index))]) 
                                -  np.log10(np.sort(simulated[:,col_idx])[int(0.7*len(index))])  
                           - np.log10(np.sort(response.values[windup_timesteps+1:,col_idx])[int(0.2*len(index))])
                                      + np.log10(np.sort(response.values[windup_timesteps+1:,col_idx])[int(0.7*len(index))]) ) 
                           / np.log10(np.sort(response.values[windup_timesteps+1:,col_idx])[int(0.2*len(index))])
                                      - np.log10(np.sort(response.values[windup_timesteps+1:,col_idx])[int(0.7*len(index))])) 

            print("MAE = ", mae)
            print("RMSE = ", rmse)
            print("NSE = ", nse)
            # alpha nse decomposition due to gupta et al 2009
            print("alpha = ", alpha)
            print("beta = ", beta)
            # top 2% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV = ", hfv)
            # top 10% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV10 = ", hfv10)
            # 30% low flow bias (LFV) due to yilmaz et al 2008
            print("LFV = ", lfv)
            # bias of FDC midsegment slope due to yilmaz et al 2008
            print("FDC = ", fdc)
            # compile all the error metrics into a dictionary
            error_metrics = {"MAE":mae,"RMSE":rmse,"NSE":nse,"alpha":alpha,"beta":beta,"HFV":hfv,"HFV10":hfv10,"LFV":lfv,"FDC":fdc,"r2":r2}
        
        except Exception as e: # and print the exception:
            print("Exception in simulation\n")
            print(e)
            error_metrics = {"MAE":[np.NAN],"RMSE":[np.NAN],"NSE":[np.NAN],"alpha":[np.NAN],"beta":[np.NAN],"HFV":[np.NAN],"HFV10":[np.NAN],"LFV":[np.NAN],"FDC":[np.NAN],"r2":r2}

            return {"error_metrics": error_metrics, "model": model, "simulated": response[1:], "response": response, "forcing": forcing, "index": index,"diverged":True}

        
          
    return {"error_metrics": error_metrics, "model": model, "simulated": simulated, "response": response, "forcing": forcing, "index": index,"diverged":False}
    #return [r2, model, mae, rmse, index, simulated , response , forcing]



def transform_inputs(shape_factors, scale_factors, loc_factors,index, forcing):
    # original forcing columns -> columns of forcing that don't have _tr_ in their name
    orig_forcing_columns = [col for col in forcing.columns if "_tr_" not in col]
    #print("original forcing columns = ", orig_forcing_columns)
    # how many rows of shape_factors do not contain NaNs?
    num_transforms = shape_factors.count().iloc[0]
    #print("num_transforms = ", num_transforms)
    #print("forcing at beginning of transform inputs")
    #print(forcing)
    shape_time = np.arange(0,len(index),1) 
    for input in orig_forcing_columns: # which input are we talking about?
        for transform_idx in range(1,num_transforms + 1): # which transformation of that input are we talking about?
            # if the column doesn't exist, create it
            if (str(str(input) + "_tr_" + str(transform_idx)) not in forcing.columns):
                forcing.loc[:,str(str(input) + "_tr_" + str(transform_idx))] = 0.0
            # now, fill it with zeros (need to reset between different transformation shape factors)
            forcing[str(str(input) + "_tr_" + str(transform_idx))].values[:] = float(0.0)
            #print(forcing)
            for idx in range(0,len(index)): # timestep
                if (abs(forcing[input].iloc[idx]) > 10**-6): # when nonzero forcing occurs
                    if (idx == int(0)):
                        forcing[str(str(input) + "_tr_" + str(transform_idx))].values[idx:] += forcing[input].values[idx]*stats.gamma.pdf(shape_time, shape_factors[input][transform_idx], scale=scale_factors[input][transform_idx], loc = loc_factors[input][transform_idx]) 
                    else:
                        forcing[str(str(input) + "_tr_" + str(transform_idx))].values[idx:] += forcing[input].values[idx]*stats.gamma.pdf(shape_time[:-idx], shape_factors[input][transform_idx], scale=scale_factors[input][transform_idx], loc = loc_factors[input][transform_idx]) 

    #print("forcing at end of transform inputs")
    #print(forcing)
    # assert there are no NaNs in the forcing
    assert(forcing.isnull().values.any() == False)
    return forcing

# REQUIRES: the output of delay_io_train, starting value of otuput, forcing timeseries
# EFFECTS: returns a simulated response given forcing and a model
# REQUIRED EDITS: not written to accomodate transform_dependent yet
def delay_io_predict(delay_io_model, system_data, num_transforms=1,evaluation=False , windup_timesteps=None):
    if windup_timesteps is None: # user didn't specify windup timesteps, use what the model trained with.
        windup_timesteps = delay_io_model[num_transforms]['windup_timesteps']
    forcing = system_data[delay_io_model[num_transforms]['independent_columns']].copy(deep=True)
    response = system_data[delay_io_model[num_transforms]['dependent_columns']].copy(deep=True)
            
    transformed_forcing = transform_inputs(shape_factors=delay_io_model[num_transforms]['shape_factors'], 
                                           scale_factors=delay_io_model[num_transforms]['scale_factors'], 
                                           loc_factors=delay_io_model[num_transforms]['loc_factors'], 
                                           index=system_data.index,forcing=forcing)
    try:
        prediction = delay_io_model[num_transforms]['final_model']['model'].simulate(system_data[delay_io_model[num_transforms]['dependent_columns']].iloc[windup_timesteps,:], 
                                                                         t=np.arange(0,len(system_data.index),1)[windup_timesteps:], 
                                                                         u=transformed_forcing[windup_timesteps:])
    except Exception as e: # and print the exception:
        print("Exception in simulation\n")
        print(e)
        print("diverged.")
        error_metrics = {"MAE":[np.NAN],"RMSE":[np.NAN],"NSE":[np.NAN],"alpha":[np.NAN],"beta":[np.NAN],"HFV":[np.NAN],"HFV10":[np.NAN],"LFV":[np.NAN],"FDC":[np.NAN]}
        return {'prediction':np.NAN*np.ones(shape=response[windup_timesteps+1:].shape), 'error_metrics':error_metrics,"diverged":True}

    # return all the error metrics if the prediction is being evaluated against known measurements
    if (evaluation):
        try:
            mae = list()
            rmse = list()
            nse = list()
            alpha = list()
            beta = list()
            hfv = list()
            hfv10 = list()
            lfv = list()
            fdc = list()
            for col_idx in range(0,len(response.columns)): # univariate performance metrics
                error = response.values[windup_timesteps+1:,col_idx]-prediction[:,col_idx]

                initial_error_length = len(error)
                error = error[~np.isnan(error)]
                if (len(error) < 0.75*initial_error_length):
                    print("WARNING: More than 25% of the entries in error were NaN")

                #print("error")
                #print(error)
                # nash sutcliffe efficiency between response and prediction
                mae.append(np.mean(np.abs(error)))
                rmse.append(np.sqrt(np.mean(error**2 ) ))
                #print("mean measured = ", np.mean(response.values[windup_timesteps+1:,col_idx]  ))
                #print("sum of squared error between measured and model = ", np.sum((error)**2 ))
                #print("sum of squared error between measured and mean of measured = ", np.sum((response.values[windup_timesteps+1:,col_idx]-np.mean(response.values[windup_timesteps+1:,col_idx]  ) )**2 ))
                nse.append(1 - np.sum((error)**2 )  /  np.sum((response.values[windup_timesteps+1:,col_idx]-np.mean(response.values[windup_timesteps+1:,col_idx]  ) )**2 ) )
                alpha.append(np.std(prediction[:,col_idx])/np.std(response.values[windup_timesteps+1:,col_idx]))
                beta.append(np.mean(prediction[:,col_idx])/np.mean(response.values[windup_timesteps+1:,col_idx]))
                hfv.append(np.sum(np.sort(prediction[:,col_idx])[-int(0.02*len(system_data.index)):])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.02*len(system_data.index)):]))
                hfv10.append(np.sum(np.sort(prediction[:,col_idx])[-int(0.1*len(system_data.index)):])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.1*len(system_data.index)):]))
                lfv.append(np.sum(np.sort(prediction[:,col_idx])[:int(0.3*len(system_data.index))])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[:int(0.3*len(system_data.index))]))
                fdc.append(np.mean(np.sort(prediction[:,col_idx])[-int(0.6*len(system_data.index)):-int(0.4*len(system_data.index))])/np.mean(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.6*len(system_data.index)):-int(0.4*len(system_data.index))]))
            

            print("MAE = ", mae)
            print("RMSE = ", rmse)
            
            print("NSE = ", nse)
            # alpha nse decomposition due to gupta et al 2009
            print("alpha = ", alpha)
            print("beta = ", beta)
            # top 2% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV = ", hfv)
            # top 10% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV10 = ", hfv10)
            # 30% low flow bias (LFV) due to yilmaz et al 2008
            print("LFV = ", lfv)
            # bias of FDC midsegment slope due to yilmaz et al 2008
            print("FDC = ", fdc)
            # compile all the error metrics into a dictionary
            error_metrics = {"MAE":mae,"RMSE":rmse,"NSE":nse,"alpha":alpha,"beta":beta,"HFV":hfv,"HFV10":hfv10,"LFV":lfv,"FDC":fdc}
            # omit r2 here because it doesn't mean the same thing as it does for training, would be misleading.
            # r2 in training expresses how much of the derivative is predicted by the model, whereas in evaluation it expresses how much of the response is predicted by the model

            return {'prediction':prediction, 'error_metrics':error_metrics,"diverged":False}
        except Exception as e: # and print the exception:
            print(e)
            print("Simulation diverged.")
            error_metrics = {"MAE":[np.NAN],"RMSE":[np.NAN],"NSE":[np.NAN],"alpha":[np.NAN],"beta":[np.NAN],"HFV":[np.NAN],"HFV10":[np.NAN],"LFV":[np.NAN],"FDC":[np.NAN],"diverged":True}

            return {'prediction':prediction, 'error_metrics':error_metrics}
    else:
        error_metrics = {"MAE":[np.NAN],"RMSE":[np.NAN],"NSE":[np.NAN],"alpha":[np.NAN],"beta":[np.NAN],"HFV":[np.NAN],"HFV10":[np.NAN],"LFV":[np.NAN],"FDC":[np.NAN]}
        return {'prediction':prediction, 'error_metrics':error_metrics,"diverged":False}




### the functions below are for generating LTI systems directly from data


# the function below returns an LTI system (in the matrices A, B, and C) that mimic the shape of a given gamma distribution
# scaling should be correct, but need to verify that
# max state dim, resolution, and max iterations could be icnrased to improve accuracy
def lti_from_gamma(shape, scale, location,dt=0,desired_NSE = 0.999,verbose=False,
                   max_state_dim=50,max_iterations=200, max_pole_speed = 5, min_pole_speed = 0.01):
    # a pole of speed -5 decays to less than 1% of it's value after one timestep
    # a pole of speed -0.01 decays to more than 99% of it's value after one timestep

    # i've assumed here that gamma pdf is defined the same as in matlab
    # if that's not true testing will show it soon enough
    t50 = shape*scale + location # center of mass
    skewness = 2 / np.sqrt(shape)
    total_time_base = 2*t50 # not that this contains the full shape, but if we fit this much of the curve perfectly we'll be close enough
    #resolution = (t50)/((skewness + location)) # make this coarser for faster debugging
    resolution = (t50)/(10*(skewness + location)) # production version

    #resolution = 1/ skewness
    decay_rate = 1 / resolution
    decay_rate = np.clip(decay_rate ,min_pole_speed, max_pole_speed)
    state_dim = int(np.floor(total_time_base*decay_rate)) # this keeps the time base fixed for a given decay rate
    if state_dim > max_state_dim:
        state_dim = max_state_dim
        decay_rate = state_dim / total_time_base
        resolution = 1 / decay_rate
    if state_dim < 1:
        state_dim = 1
        decay_rate = state_dim / total_time_base
        resolution = 1 / decay_rate
        
    decay_rate = np.clip(decay_rate ,min_pole_speed, max_pole_speed)

    if verbose:
        print("state dimension is ",state_dim)
        print("decay rate is ",decay_rate)
        print("total time base is ",total_time_base)
        print("resolution is", resolution)


    # make the timestep one so that the relative error is correct (dt too small makes error bigger than written)
    #t = np.linspace(0,3*total_time_base,1000)
    #desired_error = desired_error / dt
    '''
    if dt > 0: # true if numeric
        t = np.arange(0,2*total_time_base,dt)
    else:
        t= np.linspace(0,2*total_time_base,num=200) 
    '''
    t = np.linspace(0,2*total_time_base,num=200)
    
    #if verbose:
    #    print("dt is ",dt)
    #    print("scaled desired error is ",desired_error)

    gam = stats.gamma.pdf(t,shape,location,scale)

    # A is a cascade with the appropriate decay rate
    A = decay_rate*np.diag(np.ones((state_dim-1)) , -1) - decay_rate*np.diag(np.ones((state_dim)),0)
    # influence enters at the top state only
    B = np.concatenate((np.ones((1,1)),np.zeros((state_dim-1,1))))
    # contributions of states to the output will be scaled to match the gamma distribution
    C = np.ones((1,state_dim))*max(gam)
    lti_sys = control.ss(A,B,C,0)

    lti_approx = control.impulse_response(lti_sys,t)
    '''
    error = np.sum(np.abs(gam - lti_approx.y))
    if(verbose):
        print("initial error")
        print(error)
        #print("desired error")
        #print(max(gam))
        #print(desired_error)
        '''
    NSE = 1 - (np.sum(np.square(gam - lti_approx.y)) / np.sum(np.square(gam - np.mean(gam)) ))
    # if NSE is nan, set to -10e6
    if np.isnan(NSE):
        NSE = -10e6


    if verbose:
        print("initial NSE")
        print(NSE)
        print("desired NSE")
        print(desired_NSE)
        
    iterations = 0

    speeds = [10,5,2,1.1,1.05,1.01,1.001]
    speed_idx = 0
    leap = speeds[speed_idx]
    # the area under the curve is normalized to be one. so rather than basing our desired error off the 
    # max of the distribution, it might be better to make it a percentage error, one percent or five percent
    while (NSE < desired_NSE and iterations < max_iterations):
        
        og_was_best = True # start each iteration assuming that the original is the best
        # search across the C vector
        for i in range(C.shape[1]-1,int(-1),int(-1)): # accross the columns # start at the end and come back
        #for i in range(int(0),C.shape[1],int(1)): # accross the columns, start at the beginning and go forward
            
            og_approx = control.ss(A,B,C,0)
            og_y = np.ndarray.flatten(control.impulse_response(og_approx,t).y)
            og_error = np.sum(np.abs(gam - og_y))
            og_NSE = 1 - (np.sum((gam - og_y)**2) / np.sum((gam - np.mean(gam))**2))

            Ctwice = np.array(C, copy=True)
            Ctwice[0,i] = leap*C[0,i]
            twice_approx = control.ss(A,B,Ctwice,0)
            twice_y = np.ndarray.flatten(control.impulse_response(twice_approx,t).y)
            twice_error = np.sum(np.abs(gam - twice_y))
            twice_NSE = 1 - (np.sum((gam - twice_y)**2) / np.sum((gam - np.mean(gam))**2))

            Chalf = np.array(C,copy=True)
            Chalf[0,i] = (1/leap)*C[0,i]
            half_approx = control.ss(A,B,Chalf,0)
            half_y = np.ndarray.flatten(control.impulse_response(half_approx,t).y)
            half_error = np.sum(np.abs(gam - half_y))
            half_NSE = 1 - (np.sum((gam - half_y)**2) / np.sum((gam - np.mean(gam))**2))
            '''
            Cneg = np.array(C,copy=True)
            Cneg[0,i] = -C[0,i]
            neg_approx = control.ss(A,B,Cneg,0)
            neg_y = np.ndarray.flatten(control.impulse_response(neg_approx,t).y)
            neg_error = np.sum(np.abs(gam - neg_y))
            neg_NSE = 1 - (np.sum((gam - neg_y)**2) / np.sum((gam - np.mean(gam))**2))
            '''
            faster = np.array(A,copy=True)
            faster[i,i] = A[i,i]*leap # faster decay
            if abs(faster[i,i]) < abs(max_pole_speed):
                if i > 0: # first reservoir doesn't receive contribution from another reservoir. want to keep B at 1 for scaling
                    faster[i,i-1] = A[i,i-1]*leap # faster rise
                faster_approx = control.ss(faster,B,C,0)
                faster_y = np.ndarray.flatten(control.impulse_response(faster_approx,t).y)
                faster_error = np.sum(np.abs(gam - faster_y))
                faster_NSE = 1 - (np.sum((gam - faster_y)**2) / np.sum((gam - np.mean(gam))**2))
            else:
                faster_NSE = -10e6 # disallowed because the pole is too fast

            slower = np.array(A,copy=True)
            slower[i,i] = A[i,i]/leap # slower decay
            if abs(slower[i,i]) > abs(min_pole_speed):
                if i > 0:
                    slower[i,i-1] = A[i,i-1]/leap # slower rise
                slower_approx = control.ss(slower,B,C,0)
                slower_y = np.ndarray.flatten(control.impulse_response(slower_approx,t).y)
                slower_error = np.sum(np.abs(gam - slower_y))
                slower_NSE = 1 - (np.sum((gam - slower_y)**2) / np.sum((gam - np.mean(gam))**2))
            else:
                slower_NSE = -10e6 # disallowed because the pole is too slow

            #all_errors = [og_error, twice_error, half_error, faster_error, slower_error]
            all_NSE  = [og_NSE, twice_NSE, half_NSE, faster_NSE, slower_NSE]# , neg_NSE]

            if (twice_NSE >= max(all_NSE) and twice_NSE > og_NSE):
                C = Ctwice
                if twice_NSE > 1.001*og_NSE: # an appreciable difference
                    og_was_best = False # did we change something this iteration?
            elif (half_NSE >= max(all_NSE) and half_NSE > og_NSE):
                C = Chalf
                if half_NSE > 1.001*og_NSE: # an appreciable difference
                    og_was_best = False # did we change something this iteration?
                
            elif (slower_NSE >= max(all_NSE) and slower_NSE > og_NSE):
                A = slower
                if slower_NSE > 1.001*og_NSE: # an appreciable difference
                    og_was_best = False # did we change something this iteration?
            elif (faster_NSE >= max(all_NSE) and faster_NSE > og_NSE):
                A = faster
                if faster_NSE > 1.001*og_NSE: # an appreciable difference
                    og_was_best = False # did we change something this iteration?
                    '''
            elif (neg_NSE >= max(all_NSE) and neg_NSE > og_NSE):
                C = Cneg
                if neg_NSE > 1.001*og_NSE:
                    og_was_best = False
                    '''
                    


        NSE = og_NSE
        error = og_error
        iterations += 1 # this shouldn't be the termination condition unless the resolution is too coarse
        # normally the optimization should exit because the leap has become too small
        if og_was_best: # the original was the best, so we're going to tighten up the optimization
            speed_idx += 1
            if speed_idx > len(speeds)-1:
                break # we're done
            leap = speeds[speed_idx]
        # print the iteration count every ten
        # comment out for production
        if (iterations % 2 == 0 and verbose):
            print("iterations = ", iterations)
            print("error = ", error)
            print("NSE = ", NSE)
            print("leap = ", leap)

    lti_approx = control.ss(A,B,C,0)
    y = np.ndarray.flatten(control.impulse_response(og_approx,t).y)
    error = np.sum(np.abs(gam - og_y))
    print("LTI_from_gamma final NSE")
    print(NSE)
    if (verbose):
        print("final system\n")
        print("A")
        print(A)
        print("B")
        print(B)
        print("C")
        print(C)

        print("\nfinal error")
        print(error)

    # are any of the final eigenvalues outside the bounds specified?
    E = np.linalg.eigvals(A)
    if (np.any(np.abs(E) > max_pole_speed) or np.any(np.abs(E) < min_pole_speed)):
        print("WARNING: final eigenvalues are outside the bounds specified")


    return {"lti_approx":lti_approx, "lti_approx_output":y, "error":error, "t":t, "gamma_pdf":gam}



# this function takes the system data and the causative topology and returns an LTI system
# if the causative topology isn't already defined, it needs to be created using infer_causative_topology
def lti_system_gen(causative_topology, system_data,independent_columns,dependent_columns,max_iter=250,
                   swmm=False,bibo_stable = False,max_transition_state_dim=50):

    # cast the columns and indices of causative_topology to strings so sindy can run properly
    # We need the tuples to link the columns in system_data to the object names in the swmm model  
    # so we'll cast these back to tuples once we're done
    if swmm:
        causative_topology.columns = causative_topology.columns.astype(str)
        causative_topology.index = causative_topology.index.astype(str)

        print("causative topology \n")
        print(causative_topology.index)
        print(causative_topology.columns)

        # do the same for dependent_columns and independent_columns
        dependent_columns = [str(col) for col in dependent_columns]
        independent_columns = [str(col) for col in independent_columns]
        print(dependent_columns)
        print(independent_columns)
    
    
        # do the same for the columns of system_data
        system_data.columns = system_data.columns.astype(str)
        print(system_data.columns)


    A = pd.DataFrame(index=dependent_columns, columns=dependent_columns)
    B = pd.DataFrame(index=dependent_columns, columns=independent_columns)
    C = pd.DataFrame(index=dependent_columns,columns=dependent_columns)
    C.loc[:,:] = np.diag(np.ones(len(dependent_columns))) # these are the states which are observable
   
    # copy the corresponding entries from the causative topology into B
    for row in B.index:
        for col in B.columns:
            B[col][row] = causative_topology[col][row]
    # and into A
    for row in A.index:
        for col in A.columns:
            A[col][row] = causative_topology[col][row]

    print("A")
    print(A)
    print("B")
    print(B)
    print("C")
    print(C)
    # use transform_only when calling delay_io_train to only train transfomrations for connections marked "d"
    # train a MISO model for each output
    delay_models = {key: None for key in dependent_columns}
    
    for row in A.index:
        immediate_forcing = []
        delayed_forcing = []
        for col in A.columns:
            if col == row:
                continue # don't need to include the output state as a forcing variable. it's already included by default
            if A[col][row] == "d":
                delayed_forcing.append(col)
            elif A[col][row] == "i":
                immediate_forcing.append(col)
        for col in B.columns:
            if B[col][row] == "d":
                delayed_forcing.append(col)
            elif B[col][row] == "i":
                immediate_forcing.append(col)
        # make total_forcing the union of immediate and delayed forcing
        total_forcing = immediate_forcing + delayed_forcing
        feature_names = [row] + total_forcing
        if (delayed_forcing):
            print("training delayed model for ", row, " with forcing ", total_forcing, "\n")
            delay_models[row] = delay_io_train(system_data,[row],total_forcing,
                                     transform_only=delayed_forcing,max_transforms=1,
                                     poly_order=1, max_iter=max_iter,verbose=False,bibo_stable=bibo_stable)
            # we'll parse this delayed causation into the matrices A, B, and C later
        else:
            ####### TODO: incorporate bibo stability constraint into instantaneous fits ########
            print("training immediate model for ", row, " with forcing ", total_forcing, "\n")
            delay_models[row] = None
            # we can put immediate causation into the matrices A, B, and C now

            if (bibo_stable): # negative autocorrelatoin
                # Figure out how many library features there will be
                library = ps.PolynomialLibrary(degree=1,include_bias = False, include_interaction=False)
                #total_train = pd.concat((response,forcing), axis='columns')
                library.fit([ps.AxesArray(feature_names,{"ax_sample":0,"ax_coord":1})])
                n_features = library.n_output_features_
                #print(f"Features ({n_features}):", library.get_feature_names())
                # Set constraints
                #n_targets = total_train.shape[1] # not sure what targets means after reading through the pysindy docs
                #print("n_targets")
                #print(n_targets)
                constraint_rhs = 0
                # one row per constraint, one column per coefficient
                constraint_lhs = np.zeros((1 , n_features ))

                #print(constraint_rhs)
                #print(constraint_lhs)
                # constrain the highest order output autocorrelation to be negative
                # this indexing is only right for include_interaction=False, include_bias=False, and pure polynomial library
                # for more complex libraries, some conditional logic will be needed to grab the right column
                constraint_lhs[:,0] = 1
                
                model = ps.SINDy(
                            differentiation_method= ps.FiniteDifference(),
                            feature_library=ps.PolynomialLibrary(degree=1,include_bias = False, include_interaction=False),
                            optimizer = ps.ConstrainedSR3(threshold=0, thresholder = "l2",constraint_lhs=constraint_lhs, constraint_rhs = constraint_rhs, inequality_constraints=True),
                            feature_names = feature_names
                        )

            else: # unoconstrained
                model = ps.SINDy(
                    differentiation_method= ps.FiniteDifference(order=10,drop_endpoints=True),
                    feature_library=ps.PolynomialLibrary(degree=1,include_bias = False, include_interaction=False), 
                    optimizer=ps.optimizers.STLSQ(threshold=0,alpha=0),
                    feature_names = feature_names
                    ) 
            if system_data.loc[:,immediate_forcing].empty: # the subsystem is autonomous
                instant_fit = model.fit(x = system_data.loc[:,row] ,t = np.arange(0,len(system_data.index),1))
                instant_fit.print(precision=3)
                print("Training r2 = ", instant_fit.score(x = system_data.loc[:,row] ,t = np.arange(0,len(system_data.index),1)))
                print(instant_fit.coefficients())
            else: # there is some forcing
                #instant_fit = model.fit(x = system_data.loc[:,row] ,t = system_data.index.values, u = system_data.loc[:,immediate_forcing]) # sindy can't take datetime indices
                instant_fit = model.fit(x = system_data.loc[:,row] ,t = np.arange(0,len(system_data.index),1) , u = system_data.loc[:,immediate_forcing])
                instant_fit.print(precision=3)
                print("Training r2 = ", instant_fit.score(x = system_data.loc[:,row] ,t = np.arange(0,len(system_data.index),1), u = system_data.loc[:,immediate_forcing]))
                print(instant_fit.coefficients())
            for idx in range(len(feature_names)):
                if feature_names[idx] in A.columns:
                    A.loc[row,feature_names[idx]] = instant_fit.coefficients()[0][idx]
                elif feature_names[idx] in B.columns:
                    B.loc[row,feature_names[idx]] = instant_fit.coefficients()[0][idx]
                else:
                    print("couldn't find a column for ", feature_names[idx])
            #print("updated A")
            #print(A)
            #print("updated B")
            #print(B)

    original_A = A.copy(deep=True)
    # now, parse the delay models into the A, B, and C matrices
    # the changes will be as follows:
    # the A matrix will have matrices of the form [B_gam, A_gam; 0 , C_gam] inserted into it
    # where X_gam are the matrices generated by the lti_from_gamma function to represent the delayed causation shape
    # the B and C matrices will just have zeros inserted into them to maintain compatible dimensions
    # none of these cascades are observable or directly receive input.
    for row in original_A.index:
        if delay_models[row] is None:
            pass
        else:
            transformation_approximations = {transform_key: None for transform_key in delay_models[row][1]['shape_factors'].columns}
            for transform_key in transformation_approximations.keys():
                delay_models[row][1]['final_model']['model'].print(precision=5)
                shape = delay_models[row][1]['shape_factors'].loc[1,transform_key]
                scale = delay_models[row][1]['scale_factors'].loc[1,transform_key]
                loc = delay_models[row][1]['loc_factors'].loc[1,transform_key]
                '''
                # infer the timestep of system_data from the index
                timestep = system_data.index[1] - system_data.index[0]
                try: # if the timestep is numeric
                    pd.to_numeric(timestep)
                    transformation_approximations[transform_key] = lti_from_gamma(shape,scale,loc,dt=timestep)
                    
                    Agam = transformation_approximations[transform_key]['lti_approx'].A / timestep
                    Bgam = transformation_approximations[transform_key]['lti_approx'].B / timestep 
                    Cgam = transformation_approximations[transform_key]['lti_approx'].C / timestep
                except Exception as e: # if the timestep is something like a datetime
                    print(e)'''
                transformation_approximations[transform_key] = lti_from_gamma(shape,scale,loc,max_state_dim = max_transition_state_dim)
                    
                Agam = transformation_approximations[transform_key]['lti_approx'].A 
                Bgam = transformation_approximations[transform_key]['lti_approx'].B # only entry is unit impulse at top state
                Cgam = transformation_approximations[transform_key]['lti_approx'].C 

                # Cgam needs to be scaled by the coefficient the forcing term had in the delay model
                coefficients = {coef_key: None for coef_key in delay_models[row][1]['final_model']['model'].feature_names}
                for coef_key in coefficients.keys():
                    coef_index = delay_models[row][1]['final_model']['model'].feature_names.index(coef_key)
                    coefficients[coef_key] = delay_models[row][1]['final_model']['model'].coefficients()[0][coef_index]
                    if "_tr_1" in coef_key and coef_key.replace("_tr_1","") == transform_key.replace("_tr_1",""):
                        '''
                        try: 
                            pd.to_numeric(timestep,errors='raise')
                            Cgam = Cgam * coefficients[coef_key] / timestep
                        except Exception as e:
                            print(e)
                            Cgam = Cgam * coefficients[coef_key]
                        '''
                        
                        Cgam = Cgam * coefficients[coef_key] # scaling
                    else: # these are the immediate effects, insert them now
                        if coef_key in A.columns:
                            A.loc[row,coef_key] = coefficients[coef_key]
                        elif coef_key in B.columns:
                            B.loc[row,coef_key] = coefficients[coef_key]

                
                Agam_index = []
                for idx in range(Agam.shape[0]):
                    Agam_index.append(transform_key.replace("_tr_1","") + "->" + row + "_" + str(idx))
                Agam = pd.DataFrame(Agam, index = Agam_index, columns = Agam_index)
                Bgam = pd.DataFrame(Bgam, index = Agam_index, columns = [transform_key.replace("_tr_1","")])
                Cgam = pd.DataFrame(Cgam, index = [row], columns = Agam_index)
                #print("Agam")
                #print(Agam)
                #print("Bgam")
                #print(Bgam)
                #print("Cgam")
                #print(Cgam)
                # insert these into the A, B, and C matrices
                # for Agam, the insertion row is immediately after the source (key)
                # the insertion column is also immediately after the source (key)
                
                ### everything below this point is garbage. not performing at all as desired at the moment


                # first need to create space for the new rows and columns
                # create before_index and after_index variables, which record the parts of the index of A that occur before and after row
                before_index = []
                #after_index = []
                if transform_key.replace("_tr_1","") not in A.index: # it's one of the forcing terms. put it in at the beginning
                    after_index = list(A.index) # it's a forcing variable, so we don't want it in the newA index
                else: # it is a state variable
                    before_index = list(A.index[:A.index.get_loc(transform_key.replace("_tr_1",""))])
                    after_index = list(A.index[A.index.get_loc(transform_key.replace("_tr_1",""))+1:])
                    '''
                    for idx in A.index:
                        if idx == key.replace("_tr_1",""):
                            before_index.append(idx) # if it's a state variable, we want it in the newA index
                            break
                        else:
                            before_index.append(idx)
                    for idx in range(A.index.get_loc(key.replace("_tr_1",""))+1,len(A.index)):
                        after_index.append(A.index[idx])
                        '''
                if transform_key.replace("_tr_1","") in A.index: # the transform key refers to a state (x)
                    states = before_index + [transform_key.replace("_tr_1","")] + Agam_index + after_index # state dim expands by the number of rows in Agam
                    # include the current transform key in A because it's a state variable
                elif transform_key.replace("_tr_1","") in B.columns: # the transform key refers to a control input (u)
                    states = before_index + Agam_index + after_index # state dim expands by the number of rows in Agam    
                    # don't include the current transform key in A because it's a control input, not a state variable
                    
                newA = pd.DataFrame(index=states, columns = states) 
                newB = pd.DataFrame(index = states, columns = B.columns) # input dim remains consistent (columns of B)
                newC = pd.DataFrame(index = C.index, columns = states) # output dim remains consistent (rows of C)

                # fill in newA with the corresponding entries from A
                for idx in newA.index:
                    for col in newA.columns:
                        if idx in A.index and col in A.columns: # if it's in the original A matrix, copy it over
                            newA.loc[idx,col] = A.loc[idx,col]
                        if idx in Agam.index and col in Agam.columns: # if it's in Agam, copy it over
                            newA.loc[idx,col] = Agam.loc[idx,col]
                        if idx in Bgam.index and col in Bgam.columns: # the input to the cascade is a state
                            newA.loc[idx,col] = Bgam.loc[idx,col]


                for idx in newB.index:
                    for col in newB.columns:
                        if idx in B.index and col in B.columns: # if it's in the original B matrix, copy it over
                            newB.loc[idx,col] = B.loc[idx,col]
                        if idx in Bgam.index and col in Bgam.columns: # the input to the cascade is a forcing term
                            newB.loc[idx,col] = Bgam.loc[idx,col]

                for idx in newC.index:
                    for col in newC.columns:
                        if idx in C.index and col in C.columns: # if it's in the original C matrix, copy it over
                            newC.loc[idx,col] = C.loc[idx,col]
                        if idx in Cgam.index and col in Cgam.columns: # outputs from the cascades
                            newA.loc[idx,col] = Cgam.loc[idx,col]

                #print("newA")
                #print(newA.to_string())
                #print("newB")
                #print(newB.to_string())
                #print("newC")
                #print(newC.to_string())

                # copy over
                A = newA.copy(deep=True)
                B = newB.copy(deep=True)
                C = newC.copy(deep=True)


    A.replace("n",0.0,inplace=True)
    B.replace("n",0.0,inplace=True)
    C.replace("n",0.0,inplace=True)
    
    if swmm:
        pass
        #############
        # TODO: cast strings back to tuples in the indices and columns
        #############
        # cast the index and columns of causative_topology to tuples. they'll be of the form "(X,Y)"

        # do the same for dependent_columns and independent_columns

        # do the same for the columns of system_data
    



    A.fillna(0.0,inplace=True)
    B.fillna(0.0,inplace=True)
    C.fillna(0.0,inplace=True)
    
    # if bibo_stable is specified and A not hurwitz, make A hurwitz by defining A' = A - I*max(real(eig(A)))
    # this will gaurantee stability (max eigenvalue will have real part < 0)
    if bibo_stable:
        orig_eigs, _ = np.linalg.eig(A)
        if any(np.real(orig_eigs) > 0):
            print("stabilizing unstable plant by subtracting I*max(real(eig)) from A")
            epsilon = 10e-4
            A_stab = A - np.eye(len(A))*(1+epsilon)*max(np.real(orig_eigs)) # add factor of (1+epsilon) for stability, not marginal stabilty
            stab_eigs, _ = np.linalg.eig(A_stab)
            A = A_stab.copy(deep=True)

    # sindy will scale the coefficients according to the timestep if the index is numeric 
    # so the whole system needs to be scaled by the timestep if its numeric
    try:
        pd.to_numeric(system_data.index,errors='raise') # can the index be converted to a numeric type?
        dt = system_data.index.values[1] - system_data.index.values[0]
        A = A / dt
        B = B / dt 
        C = C # what we observe doesn't need to be adjusted, just the dynamics
        print("system response data index converted to numeric type. dt = ")
        print(dt)
    except Exception as e:
        print(e)
        dt = None
    
    # cast all of A, B, and C to type float (integers cause issues with LQR / LQE calculations)
    A = A.astype(float)
    B = B.astype(float)
    C = C.astype(float)

    lti_sys = control.ss(A,B,C,0,inputs=B.columns,outputs=C.index,states=A.columns)


    # returning the matrices too because control.ss strips the labels from the pandas dataframes and stores them as numpy matrices
    return {"system":lti_sys,"A":A,"B":B,"C":C}





# this function takes in the system data, 
# which columns are dependent and which are independent, 
# as well as an optional constraint on the topology of the digraph
# we will return a digraph (not multidigraph as there are no parallel edges) as defined in https://networkx.org/documentation/stable/reference/classes/digraph.html 
# we'll assume there are always self-loops (the derivative always depends on the current value of the variable)
# this will also be returned as an adjacency matrix
# this doesn't go all the way to turning the data into an LTI system. that will be another function that uses this one
def infer_causative_topology(system_data, dependent_columns, independent_columns, 
                             graph_type='Weak-Conn',verbose=False,max_iter = 250,swmm=False,
                             method='granger', derivative=False):

    if swmm:
        # do the same for dependent_columns and independent_columns
        dependent_columns = [str(col) for col in dependent_columns]
        independent_columns = [str(col) for col in independent_columns]
        print(dependent_columns)
        print(independent_columns)
    
    
        # do the same for the columns of system_data
        system_data.columns = system_data.columns.astype(str)
        print(system_data.columns)

    if method == 'granger': # granger causality
        from statsmodels.tsa.stattools import grangercausalitytests
        causative_topo = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna('n')
        total_graph = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(1.0)

        print(causative_topo)

        max_p = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1.0)        
        min_p = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(2.0)
        median_p = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(2.0)
        three_quarters_p = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(2.0)
        one_quarter_p = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(2.0)
        min_p_lag = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1)
        max_p_lag = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1)
        max_p_f = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1.0)
        min_p_f = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1.0)
        median_f = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1.0)
        three_quarters_f = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1.0)
        one_quarter_f = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(-1.0)


        # first column in df is the output (granger caused by other)
        # second column is the proposed forcer
        for dep_col in dependent_columns: # for each column which is out
            for other_col in system_data.columns: # for every other variable (input)
                if other_col == dep_col:
                    continue # we're already accounting for autocorrelatoin in every fit
                print("check if ", other_col, " granger causes ", dep_col)
                #print(system_data[[dep_col,other_col]])
                try:
                    gc_res = grangercausalitytests(system_data[[dep_col,other_col]],maxlag=25,verbose=False)
                except Exception as e:
                    print(e)    
                    continue
                # iterate through the dictionary and compute the maximum and minimum p values for the F test
                p_values = []
                f_values = []
                for key in gc_res.keys():
                    f_test_p_value = gc_res[key][0]['ssr_ftest'][1]
                    p_values.append(f_test_p_value)
                    f_values.append(gc_res[key][0]['ssr_ftest'][0])
                    if f_test_p_value > max_p.loc[dep_col,other_col]:
                        max_p.loc[dep_col,other_col] = f_test_p_value
                        max_p_f.loc[dep_col,other_col] = gc_res[key][0]['ssr_ftest'][0]
                        max_p_lag.loc[dep_col,other_col] = key
                        
                    if f_test_p_value < min_p.loc[dep_col,other_col]:
                        min_p.loc[dep_col,other_col] = f_test_p_value
                        min_p_f.loc[dep_col,other_col] = gc_res[key][0]['ssr_ftest'][0]
                        min_p_lag.loc[dep_col,other_col] = key
                
                median_p.loc[dep_col,other_col] = np.median(p_values)
                median_f.loc[dep_col,other_col] = np.median(f_values)
                three_quarters_p.loc[dep_col,other_col] = np.quantile(p_values,0.75)
                three_quarters_f.loc[dep_col,other_col] = np.quantile(f_values,0.75)
                one_quarter_p.loc[dep_col,other_col] = np.quantile(p_values,0.25)
                one_quarter_f.loc[dep_col,other_col] = np.quantile(f_values,0.25)
        
        print("max p values")  
        print(max_p)
        print("f values corresponding to max p")
        print(max_p_f)
        print("max p lag")
        print(max_p_lag)
        print("min p values")
        print(min_p)
        print("f values corresponding to min p")
        print(min_p_f)
        print("min p lag")
        print(min_p_lag)
        print("median p values")
        print(median_p)
        print("median f values")
        print(median_f)
        
        print("now determine causative topology based on connectivity constraint")
        # start with the maximum p values, taking the significant links, then move down through the quantiles
        # if the graph is not connected, we'll move down to the next quantile
        # keep going until you satisfy the connectivity criteria
        if graph_type == 'Weak-Conn':
            # locate the smallest value of p in max_p which corresponds to an "n" in causative topo
            # this will be the first link we add
            '''
            i = 0
            while(i < 10e3):
                i += 1
                min_p_value = 2.0
                min_p_row = None
                min_p_col = None
                for row in causative_topo.index:
                    for col in causative_topo.columns:
                        if max_p.loc[row,col] < 0: 
                            continue # not valid
                        if max_p.loc[row,col] < min_p_value and causative_topo.loc[row,col] == 'n':
                            min_p_value = max_p.loc[row,col]
                            min_p_row = row
                            min_p_col = col
                            # if equal
                        elif max_p.loc[row,col] == min_p_value and causative_topo.loc[row,col] == 'n':
                            if min_p_value < 0.05:
                                print("tie in significant p")
                                # take the one with the higher f value
                                if max_p_f.loc[row,col] > max_p_f.loc[min_p_row,min_p_col]:
                                    min_p_value = max_p.loc[row,col]
                                    min_p_row = row
                                    min_p_col = col
    
                if min_p_value < 0.05:
                    causative_topo.loc[min_p_row,min_p_col] = 'd'
                    total_graph.loc[min_p_row,min_p_col] = min_p_value
                    print("added link from ", min_p_col, " to ", min_p_row, " with p = ", min_p_value)
                    print(causative_topo)
                    print(nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)))
                    if nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)):
                        print("graph is connected")
                        print(causative_topo)
                        print(total_graph)
                        return causative_topo, total_graph
                else:
                    print("no significant links found")
                    break
            print("done adding from max_p, now adding from 3/4 p")
            i = 0
            while(i < 10e3):
                i += 1
                min_p_value = 2.0
                min_p_row = None
                min_p_col = None
                for row in causative_topo.index:
                    for col in causative_topo.columns:
                        if three_quarters_p.loc[row,col] < 0: 
                            continue # not valid
                        if three_quarters_p.loc[row,col] < min_p_value and causative_topo.loc[row,col] == 'n':
                            min_p_value = three_quarters_p.loc[row,col]
                            min_p_row = row
                            min_p_col = col
                        elif three_quarters_p.loc[row,col] == min_p_value and causative_topo.loc[row,col] == 'n':
                            if min_p_value < 0.05:
                                print("tie in significant p")
                                # take the one with the higher f value
                                if three_quarters_f.loc[row,col] > three_quarters_f.loc[min_p_row,min_p_col]:
                                    min_p_value = three_quarters_p.loc[row,col]
                                    min_p_row = row
                                    min_p_col = col

                if min_p_value < 0.05:
                    causative_topo.loc[min_p_row,min_p_col] = 'd'
                    total_graph.loc[min_p_row,min_p_col] = min_p_value
                    print("added link from ", min_p_col, " to ", min_p_row, " with p = ", min_p_value)
                    print(causative_topo)
                    print(nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)))
                    if nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)):
                        print("graph is connected")
                        print(causative_topo)
                        print(total_graph)
                        return causative_topo, total_graph
                else:
                    print("no significant links found")
                    break
            print("done adding from three_quarters_p, now adding from median p")
            # move to the median
            i = 0
            while(i < 10e3):
                i += 1
                min_p_value = 2.0
                min_p_row = None
                min_p_col = None
                for row in causative_topo.index:
                    for col in causative_topo.columns:
                        if median_p.loc[row,col] < 0: 
                            continue
                        if median_p.loc[row,col] < min_p_value and causative_topo.loc[row,col] == 'n':
                            min_p_value = median_p.loc[row,col]
                            min_p_row = row
                            min_p_col = col
                        elif median_p.loc[row,col] == min_p_value and causative_topo.loc[row,col] == 'n':
                            if min_p_value < 0.05:
                                print("tie in significant p")
                                # take the one with the higher f value
                                if median_f.loc[row,col] > median_f.loc[min_p_row,min_p_col]:
                                    min_p_value = median_p.loc[row,col]
                                    min_p_row = row
                                    min_p_col = col
    
                if min_p_value < 0.05:
                    causative_topo.loc[min_p_row,min_p_col] = 'd'
                    total_graph.loc[min_p_row,min_p_col] = min_p_value
                    print("added link from ", min_p_col, " to ", min_p_row, " with p = ", min_p_value)
                    print(causative_topo)
                    print(nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)))
                    if nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)):
                        print("graph is connected")
                        print(causative_topo)
                        print(total_graph)
                        return causative_topo, total_graph
                else:
                    print("no significant links found")
                    break
            print("done adding from median p, now adding from min p")
            i = 0
            while(i < 10e3):
                i += 1
                min_p_value = 2.0
                min_p_row = None
                min_p_col = None
                for row in causative_topo.index:
                    for col in causative_topo.columns:
                        if one_quarter_p.loc[row,col] < 0: 
                            continue
                        if one_quarter_p.loc[row,col] < min_p_value and causative_topo.loc[row,col] == 'n':
                            min_p_value = one_quarter_p.loc[row,col]
                            min_p_row = row
                            min_p_col = col
                        elif one_quarter_p.loc[row,col] == min_p_value and causative_topo.loc[row,col] == 'n':
                            if min_p_value < 0.05:
                                print("tie in significant p")
                                # take the one with the higher f value
                                if one_quarter_f.loc[row,col] > one_quarter_f.loc[min_p_row,min_p_col]:
                                    min_p_value = one_quarter_p.loc[row,col]
                                    min_p_row = row
                                    min_p_col = col
    
                if min_p_value < 0.05:
                    causative_topo.loc[min_p_row,min_p_col] = 'd'
                    total_graph.loc[min_p_row,min_p_col] = min_p_value
                    print("added link from ", min_p_col, " to ", min_p_row, " with p = ", min_p_value)
                    print(causative_topo)
                    print(nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)))
                    if nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)):
                        print("graph is connected")
                        print(causative_topo)
                        print(total_graph)
                        return causative_topo, total_graph
                else:
                    print("no significant links found")
                    break
            print("done adding from median p, now adding from min p")
            '''
            # move to the min
            i = 0
            while(i < 10e3):
                i += 1
                min_p_value = 2.0
                min_p_row = None
                min_p_col = None
                for row in causative_topo.index:
                    for col in causative_topo.columns:
                        if min_p.loc[row,col] < 0: 
                            continue
                        if min_p.loc[row,col] < min_p_value and causative_topo.loc[row,col] == 'n':
                            min_p_value = min_p.loc[row,col]
                            min_p_row = row
                            min_p_col = col
                        elif min_p.loc[row,col] == min_p_value and causative_topo.loc[row,col] == 'n':
                            if min_p_value < 0.05:
                                print("tie in significant p")
                                # take the one with the higher f value
                                if min_p_f.loc[row,col] > min_p_f.loc[min_p_row,min_p_col]:
                                    min_p_value = min_p.loc[row,col]
                                    min_p_row = row
                                    min_p_col = col
                                            
                if min_p_value < 0.05 or True:
                    causative_topo.loc[min_p_row,min_p_col] = 'd'
                    total_graph.loc[min_p_row,min_p_col] = min_p_value
                    print("added link from ", min_p_col, " to ", min_p_row, " with p = ", min_p_value)
                    print(causative_topo)
                    print(nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)))
                    if nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)):
                        print("graph is connected")
                        print(causative_topo)
                        print(total_graph)
                        return causative_topo, total_graph
                else:
                    print("no significant links found")
                    break
            print("done adding from min p. if graph not connected now, it won't be")
            print(causative_topo)
            print(total_graph)
            return causative_topo, total_graph

    
    elif method == 'ccm': # convergent cross mapping per sugihara 2012
        
        correlations = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(0.0)        
        p_values = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(1.0)
        best_taus = pd.DataFrame(index=dependent_columns,columns=system_data.columns)
        best_Es = pd.DataFrame(index=dependent_columns,columns=system_data.columns)

        from causal_ccm.causal_ccm import ccm # move to initial imports if this ends up working
        
        for dep_col in dependent_columns: # for each column which is out
            if derivative:
                response = np.array(system_data[dep_col].diff().values[1:])
            else:
                response = np.array(system_data[dep_col].values)
            
            for other_col in system_data.columns: # for every other variable (input)
                plt.close('all')
                if other_col == dep_col:
                    continue # we're already accounting for autocorrelatoin in every fit
                print("check if ", other_col, " drives ", dep_col)
                if derivative:
                    forcing = np.array(system_data[other_col].values[:-1])
                else:
                    forcing = np.array(system_data[other_col].values)
                
                # start with tau_options to be between 1 and 25 timesteps
                tau_options = np.arange(1,2)#1)
                E_options = np.arange(1,3) # number of embedding dimensions
                best_p_value = 1.0 # null hypothesis is that there is no causality
                best_tau = -1 # then we'll know if no lags had good results
                for tau in tau_options:
                    for E in E_options:
                        cross_map = ccm(forcing,response,tau=tau,E=E,L=len(response))
                        correlation, p_value = cross_map.causality()
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_correlation = correlation
                            best_tau = tau
                            best_E = E
                            print("tau = ", tau, "E = ",E," | p = ", p_value, " | corr = ", correlation)
                            #cross_map.visualize_cross_mapping()
                            #cross_map.plot_ccm_correls()
                if best_tau > -1:
                    cross_map = ccm(forcing,response,best_tau,best_E)
                    '''
                    if best_tau > 0:
                        cross_map.visualize_cross_mapping()
                    cross_map.plot_ccm_correls()
                    '''
                    correlation, p_value = cross_map.causality()
                    correlations.loc[dep_col,other_col] = correlation
                    p_values.loc[dep_col,other_col] = p_value
                    if p_value == 0: # if the p value is exactly zero, make it the minimum float value
                        p_values.loc[dep_col,other_col] = sys.float_info.min
                    best_taus.loc[dep_col,other_col] = best_tau
                    best_Es.loc[dep_col,other_col] = best_E
                    '''
                    lengths = np.linspace(250, len(response), 100,dtype='int')
                    corr_L = lengths*0.0
                    for length_idx in range(len(lengths)):
                        trunc_forcing = forcing[:lengths[length_idx]]
                        trunc_response = response[:lengths[length_idx]]
                        cross_map = ccm(trunc_forcing,trunc_response,tau=best_tau,E=best_E)
                        correlation, p_value = cross_map.causality()
                        corr_L[length_idx] = correlation
                   
                
                    plt.plot(corr_L)
                    plt.ylabel("correlation")
                    plt.show(block=True)
                    '''
                elif best_tau == -1:
                    print("no good lags found for ", dep_col, " and ", other_col)
                    correlations.loc[dep_col,other_col] = 0.0
                    p_values.loc[dep_col,other_col] = 1.0
                    best_taus.loc[dep_col,other_col] = -1
                    best_Es.loc[dep_col,other_col] = -1    
                
        print(correlations)
        print(p_values)
        print(best_taus)
        print(best_Es)
        print("done")
        causative_topo = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna('n')
        total_graph = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(1.0)
        i = 0
        while(i < 10e3):
            i += 1
            min_p_value = 2.0
            min_p_corr = 0.0
            min_p_row = None
            min_p_col = None
            for row in causative_topo.index:
                for col in causative_topo.columns:
                    if p_values.loc[row,col] < 0: 
                        continue
                    if p_values.loc[row,col] < min_p_value and causative_topo.loc[row,col] == 'n':
                        min_p_value = p_values.loc[row,col]
                        min_p_corr = correlations.loc[row,col]
                        min_p_row = row
                        min_p_col = col
                    # if two p values are tied, pick the one with the higher correlation
                    elif p_values.loc[row,col] == min_p_value and causative_topo.loc[row,col] == 'n' and correlations.loc[row,col] > min_p_corr:
                        min_p_value = p_values.loc[row,col]
                        min_p_corr = correlations.loc[row,col]
                        min_p_row = row
                        min_p_col = col
            if min_p_value < 0.05:
                causative_topo.loc[min_p_row,min_p_col] = 'd'
                total_graph.loc[min_p_row,min_p_col] = min_p_value
                print("added link from ", min_p_col, " to ", min_p_row, " with p = ", min_p_value)
                print(causative_topo)
                print(total_graph.replace(1.0,0))
                print(nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)))
                if nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph.replace(1.0,0),create_using=nx.DiGraph)):
                    print("graph is connected")
                    break
            else:
                print("no significant links found")
                break
        
        print(causative_topo)
        print(total_graph)
        return causative_topo, total_graph
                    
    elif method == 'transfer-entropy':
        
        transfer_entropies = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(0.0)        
        
        from PyIF import te_compute as te
        
        for dep_col in dependent_columns: # for each column which is out
            if derivative:
                response = np.array(system_data[dep_col].diff().values[1:])
            else:
                response = np.array(system_data[dep_col].values)
            
            for other_col in system_data.columns: # for every other variable (input)
                plt.close('all')
                if other_col == dep_col:
                    continue # we're already accounting for autocorrelatoin in every fit
                print("check if ", other_col, " drives ", dep_col)
                if derivative:
                    forcing = np.array(system_data[other_col].values[:-1])
                else:
                    forcing = np.array(system_data[other_col].values)
                
               
                k_options = np.arange(1,11) # number of neighbors used in KD-tree queries
                E_options = np.arange(1,11) # number of embedding dimensions (delay)
                best_TE = -1.0 # best transfer entropy so far
                for k in k_options:
                    for E in E_options:
                        TE = te.te_compute(forcing,response,k,E) # "information transfer from X to Y"
                        if TE > best_TE:
                            best_TE = TE
                            best_k = k
                            best_E = E
                            print("k (# neighbors) = ", k, "E (embedding dim) = ",E, " | Transfer Entropy = ", TE)
                transfer_entropies.loc[dep_col,other_col] = best_TE
                  
        print("transfer entropies")
        print(transfer_entropies)
        
        causative_topo = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna('n')
        total_graph = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna(0.0)
        i = 0
        while(i < 10e3):
            i += 1
            max_te = 0.0
            max_te_row = None
            max_te_col = None
            for row in causative_topo.index:
                for col in causative_topo.columns:
                    if transfer_entropies.loc[row,col] > max_te and causative_topo.loc[row,col] == 'n':
                        max_te = transfer_entropies.loc[row,col]
                        max_te_row = row
                        max_te_col = col
            
            causative_topo.loc[max_te_row,max_te_col] = 'd'
            total_graph.loc[max_te_row,max_te_col] = max_te
            print("added link from ", max_te_col, " to ", max_te_row, " with p = ", max_te)
            print(causative_topo)

            print(nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph)))
            if nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph)):
                print("graph is connected")
                break

        print(causative_topo)
        print(total_graph)
        return causative_topo, total_graph

    elif method == 'modpods':
        # first, identify any immediate causal relationships (no delay)
        # only using linear models for the sake of speed.
        immediate_impact_strength = pd.DataFrame(index=system_data.columns,columns=system_data.columns).fillna(0.0)
        # read as: row variable is affected by column variable
        # that way we can read each row (kind of) as a linear differential equation (not exactly, because they're all trained separately)
        for dep_col in dependent_columns: # for each column which is out
            response = np.array(system_data[dep_col].values)
            for other_col in system_data.columns: # for every other variable (input)
                if other_col == dep_col:
                    continue # we're already accounting for autocorrelatoin in every fit
            
                print("fitting ", dep_col, " to ", other_col)
                forcing = np.array(system_data[other_col].values)
            
                model = ps.SINDy(
                        differentiation_method= ps.FiniteDifference(),
                        feature_library=ps.PolynomialLibrary(degree=1,include_bias = False), 
                        optimizer = ps.STLSQ(threshold=0),
                        feature_names = [str(dep_col),str(other_col)]
                )

                # windup latent states (if your windup is too long, this will error)
                model.fit(response, u = forcing)
                # training data score
                immediate_impact_strength.loc[dep_col,other_col] = model.score(response, u = forcing) 
                if verbose:
                    model.print(precision=5)
                    print(model.score(response, u = forcing))

        # set the entries in immediate_impact_strength to 0 if they explain less than X% of the variatnce
        immediate_impact_strength[immediate_impact_strength < 1/(len(dependent_columns))] = 0.0
        print(immediate_impact_strength)

        # is system already weakly connected?
        # if not, we'll need to add edges to make it weakly connected
        print("immediate impact already weakly connected?")
        print(nx.is_weakly_connected(nx.from_pandas_adjacency(immediate_impact_strength,create_using=nx.DiGraph)))

        # if graph_type == "Weak-Conn" - find the best weakly connected graph - the undirected graph can be fully traversed
        # this is a weak constraint. it's essentailly saying all the data belong to the same system and none of it can be completely isolated
        # every DAG is weakly connected, but not every weakly connected graph is a DAG (ex: node has no in-edges and an out-edge into a three node cycle)
        # "Weak-Conn" is the default value

        # if graph_type == "Strong-Conn" - find the best strongly connected graph - the directed graph can be fully traversed
        # this is a stronger constraint. it means that every variable is affected by every other variable. every strongly connected graph is weakly connected

        # could add unilaterally connected graphs

        # if verbose, plot the network after immediate impacts are accounted for
        if verbose:
            edges = immediate_impact_strength.stack().rename_axis(['source', 'target']).rename('weight').reset_index().query('(source != target) & (weight > 0.0)')

            G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph)
            try:
                pos = nx.planr_layout(G)
            except:
                pos = nx.kamada_kawai_layout(G)
            
            nx.draw_networkx_nodes(G, pos, node_size=100)
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(10)
            plt.close('all')
   
    
        # then, test every pair of variables for a causal relationship using delay_io_train. record the r2 score achieved with a siso model
        delayed_impact_strength = pd.DataFrame(index=system_data.columns,columns=system_data.columns).fillna(0.0)
        # this is read the same way as immediate_impact_strength
    
        for dep_col in dependent_columns: # for each column which is not forcing

            for other_col in system_data.columns: # for every other variable (including forcing)
                if other_col == dep_col:
                    continue # we're already accounting for autocorrelatoin in every fit
            
                if verbose:
                    print("fitting ", dep_col, " to ", other_col)

                subset = system_data[[dep_col,other_col]]
                # max iterations is very low here because we're not trying to create an accurate model, just trying to see what affects what
                # creating the accurate model is a later task for a different function
                # it would be wasteful to spend 100 iterations on each pair of variables
                # up the iterations to 10 or so for production. 1 is jsut for development
                results = delay_io_train(subset, [dep_col], [other_col], windup_timesteps=0,init_transforms=1, max_transforms=1, max_iter=max_iter, poly_order=1, 
                                         transform_dependent=False, verbose=False, extra_verbose=False, 
                                         include_bias=False, include_interaction=False, bibo_stable = False)

                delayed_impact_strength.loc[dep_col,other_col] = results[1]['final_model']['error_metrics']['r2']
            
                if verbose:
                    print("R2 score:",  results[1]['final_model']['error_metrics']['r2'])
    
        # iteratively add edges from delayed_impact_strength until the total graph is weakly connected
        causative_topo = pd.DataFrame(index=dependent_columns,columns=system_data.columns).fillna('n')
        # wherever there is a nonzero entry in immediate_impact_strength, put an "i" in causative_topo
        causative_topo[immediate_impact_strength > 0] = "i"

        total_graph = immediate_impact_strength.copy(deep=True)
        weakest_row = 0
    
        while not nx.is_weakly_connected(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph)) and weakest_row < 0.5:
            # find the edge with the highest r2 score
            max_r2 = delayed_impact_strength.max().max()
            max_r2_row = delayed_impact_strength.max(axis='columns').idxmax()
            max_r2_col = delayed_impact_strength.max(axis='index').idxmax()
            print("\n")
            print("max_r2_row", max_r2_row)
            print("max_r2_col", max_r2_col)
            print("max_r2", max_r2)
            print("already exists path from row to col?")
            print(nx.has_path(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph),max_r2_row,max_r2_col))
            if nx.has_path(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph),max_r2_row,max_r2_col):
                print("shortest path from row to col")
                print(nx.shortest_path(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph),max_r2_row,max_r2_col))
                print("shortest path length from row to col")
                print(len(nx.shortest_path(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph),max_r2_row,max_r2_col)))
                shortest_path = len(nx.shortest_path(nx.from_pandas_adjacency(total_graph,create_using=nx.DiGraph),max_r2_row,max_r2_col))
            else:
                shortest_path = 0 # no path exists, so the shortest path is 0

            # add that edge to the total graph if it's r2 score is more than twice the corresponding entry in immediate_impact_strength
            # and there is not already a path from the row to the column in the total graph
            # constraint 1 is to not include representation of delay when it's not necessary, because it's expensive
            # constarint 2 is to not "leapfrog" intervening states when there is some chain of instantaneously related states that allow that causality to flow 
            if (max_r2 > 2*immediate_impact_strength.loc[max_r2_row,max_r2_col] 
            and (shortest_path < 3 ) ):
                total_graph.loc[max_r2_row,max_r2_col] = max_r2
                causative_topo.loc[max_r2_row,max_r2_col] = "d"
            # remove that edge from delayed_impact_strength
            delayed_impact_strength.loc[max_r2_row,max_r2_col] = 0.0

            # make weakest_row the sum of the row of total_graph with the lowest sum
            weakest_row = total_graph.loc[dependent_columns,:].sum(axis='columns').min()
        
            print("total graph")
            print(total_graph)
            print("delayed impact strength")
            print(delayed_impact_strength)
            print("\n")
    
        print("total graph is now weakly connected")
        if verbose:
            print(total_graph)
            print("causative topo")
            print(causative_topo)
            edges = total_graph.stack().rename_axis(['source', 'target']).rename('weight').reset_index().query('(source != target) & (weight > 0.0)')

            G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph)
            try:
                pos = nx.planr_layout(G)
            except:
                pos = nx.kamada_kawai_layout(G)
            
            nx.draw_networkx_nodes(G, pos, node_size=100)
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(10)
            plt.close('all')
    
    # return an adjacency matrix with "i" for immediate, "d" for delayed, and "n" for no causal relationship
    # use "d" if there is strong immediate and delayed causation. immediate causation is always cheap to include, so it'll be in any delayed causation model

    return causative_topo, total_graph

    

# this function takes a swmm input file and returns a dataframe of the topography
# arguments are included that take dictionaries where the keys are the names of the object 
# and the values are lists of the observable quantities at that object
# each argument can also accept "ALL" as an argument, which will include all objects of that type. the observable quantities must still be specified
def topo_from_pystorms(pystorms_scenario):
    
    A = pd.DataFrame(index = pystorms_scenario.config['states'],
                     columns = pystorms_scenario.config['states'])
    B = pd.DataFrame(index = pystorms_scenario.config['states'],
                     columns = pystorms_scenario.config['action_space'])

    #print("A")
    #print(A)
    #print("B")
    #print(B)


    # use pyswmm to iterate through the network
    with pyswmm.Simulation(pystorms_scenario.config['swmm_input']) as sim:
        # start at each subcatchment and iterate down to the outfall
        # this should work even in the case of multiple outfalls
        # this should capture all the causation, because ultimately everything is precip driven
        
        # so i can view these while debugging
        Subcatchments = pyswmm.Subcatchments(sim)
        Nodes = pyswmm.Nodes(sim)
        Links = pyswmm.Links(sim)

        for subcatch in pyswmm.Subcatchments(sim):
            #print(subcatch.subcatchmentid)
            # create a string that records the path we travel to get to the outfall
            path_of_travel = list()
            # can i grab the rain gage id?
            path_of_travel.append((subcatch.subcatchmentid,"Subcatchment"))
            current_id = subcatch.connection # grab the id of the next object downstream
            
            
            try: # if the downstream connection is a subcatchment
                current = Subcatchments[current_id]
                current_id = current.subcatchmentid
                subcatch = Subcatchments[current_id]
                current_id = subcatch.connection # grab the id of the next object downstream
                path_of_travel.append((current_id,'Subcatchment'))
            except Exception as e:
                #print("downstream connection was not another subcatchment")
                #print(e)
                pass
            
            # other option is that downstream connection is a node
            # in which case we'll start iterating down through nodes and links to the outfall
            current = Nodes[current_id]
            path_of_travel.append((current_id,'Node'))
            while not current.is_outfall():
                #print(path_of_travel)
                # if the current object is a node, iterate through the links to find the downstream object
                if current_id in pyswmm.Nodes(sim):
                    for link in pyswmm.Links(sim):
                        #print(link.linkid)
                        if link.inlet_node == current_id:
                            path_of_travel.append((link.linkid,"Link"))
                            current_id = link.outlet_node
                            path_of_travel.append((current_id,"Node"))
                            break
                    else:
                        print("current element is a sink (no link draining). verify this is correct")
                        print(current_id)
                        break
                # if the current object is a link, grab the downstream node
                elif current_id in pyswmm.Links(sim):
                    path_of_travel.append((link.linkid,"Link"))
                    current_id = current.outlet_node
                    path_of_travel.append((current_id,"Node"))
                

                current = Nodes[current_id]

            #print("path of travel")
            #print(path_of_travel)
            # cut all the entries in path_of_travel that are not observable states or actions
            original_path_of_travel = path_of_travel.copy()
            
            for step in original_path_of_travel:
                step_is_state = False
                step_is_control_input = False
                for state in pystorms_scenario.config['states']:
                    if step[0] == state[0]: # same id
                        if ((step[1] == "Node" and "N" in state[1]) 
                        or (step[1] == "Node" and 'flooding' in state[1]) 
                        or (step[1] == "Node" and 'inflow' in state[1])
                        or (step[1] == "Link" and "L" in state[1]) 
                        or (step[1] == "Link" and 'flow' in state[1])): # types match
                            step_is_state = True
                for control_input in pystorms_scenario.config['action_space']:
                    if step[0] == control_input:
                        step_is_control_input = True
                if not step_is_state and not step_is_control_input:
                    path_of_travel.remove(step) # this will change the index, hence the "while"
            '''        
            print("full path of travel")
            print(original_path_of_travel)
            print("observable path of travel")
            print(path_of_travel)
            '''
            # iterate through the path of travel and rename the steps to align with the columns and indices of A and B
            for step in path_of_travel:
                for state in pystorms_scenario.config['states']:
                    if step[0] == state[0]: # same id
                        if ((step[1] == "Node" and "N" in state[1]) 
                        or (step[1] == "Node" and 'flooding' in state[1]) 
                        or (step[1] == "Node" and 'inflow' in state[1])
                        or (step[1] == "Link" and "L" in state[1]) 
                        or (step[1] == "Link" and 'flow' in state[1])): # types match    
                            path_of_travel[path_of_travel.index(step)] = state

                for control_input in pystorms_scenario.config['action_space']:
                    if step[0] == control_input:
                        path_of_travel[path_of_travel.index(step)] = control_input
                        
            #print("observable path of travel")
            #print(path_of_travel)

            # now, use this path of travel to update the A and B matrices
            #print("updating A and B matrices")
            
            # only use "i" if the entries have the same id. otherwise characterize everything as delayed, "d"
            # because our path of travel only includes the observable states and the action space, we just need to look immediately up and downstream
            # only looking upstream would simplify things and be sufficient for many scenarios, but it would miss backwater effects
            for step in path_of_travel: # all of these are either observable states or actions
                if path_of_travel.index(step) == 0: # first entry, previous step not meaningful
                    prev_step = False
                else:    
                    prev_step = path_of_travel[path_of_travel.index(step)-1]
                if path_of_travel.index(step) == len(path_of_travel)-1: # last entry, next step not meaningful)
                    next_step = False
                else:
                    next_step = path_of_travel[path_of_travel.index(step)+1]

                if step in pystorms_scenario.config['action_space']: 
                    continue # we're not learning models for the control inputs, so skip them

                if prev_step and prev_step in pystorms_scenario.config['states']:
                   
                    if re.search(r'\d+', ''.join(prev_step)).group() == re.search(r'\d+', ''.join(step)).group(): # same integer id
                        A.loc[[step],[prev_step]] = 'i'
                    else:
                        A.loc[[step],[prev_step]] = 'd'
                elif prev_step and prev_step in pystorms_scenario.config['action_space']:
                    
                    if re.search(r'\d+', ''.join(prev_step)).group() == re.search(r'\d+', ''.join(step)).group(): # same integer id
                        B.loc[[step],[prev_step]] = 'i'
                    else:
                        B.loc[[step],[prev_step]] = 'd'
                if next_step and next_step[0] in pystorms_scenario.config['states'] or next_step in pystorms_scenario.config['states']:
                    # this only handles integer ids, but some models have letter ids or alphanumeric ids (pystorms scenario delta)
                    if re.search(r'\d+', ''.join(next_step)).group() == re.search(r'\d+', ''.join(step)).group(): 
                        A.loc[[step],[next_step]] = 'i'
                    else:
                        A.loc[[step],[next_step]] = 'd'
                elif next_step and next_step[0] in pystorms_scenario.config['action_space'] or next_step in pystorms_scenario.config['action_space']:
                    
                    if re.search(r'\d+', ''.join(next_step)).group() == re.search(r'\d+', ''.join(step)).group():
                        B.loc[[step],[next_step]] = 'i'
                    else:
                        B.loc[[step],[next_step]] = 'd'
            
                            



            '''
            for step in path_of_travel:
                for state in pystorms_scenario.config['states']:
                    last_step = False
                    if step[0] == state[0]: # same id
                        if ((step[1] == "Node" and "N" in state[1]) 
                        or (step[1] == "Node" and 'flooding' in state[1]) 
                        or (step[1] == "Node" and 'inflow' in state[1])): # node type
                            # we've found a step in the path of travel which is an observable state
                            # are there any other observable states or controllabe assets in the path of travel?
                            for other_step in path_of_travel:
                                if path_of_travel.index(step) - path_of_travel.index(other_step) > 1: # other step is not immediately upstream
                                    continue
                                if other_step == step:
                                    last_step = True # we only want to look one object downstream
                                    continue # this is the same step, so skip it
                                    # if you want only objects that are upstream, substitude that continue with a "break"

                                # we'll include states that come after the examined state in case of feedback such as backwater effects
                                for other_state in pystorms_scenario.config['states']:
                                    if other_step[0] == other_state[0]: # same id
                                        if ((other_step[1] == "Node" and "N" in other_state[1]) 
                                            or (other_step[1] == "Node" and 'flooding' in other_state[1]) 
                                            or (other_step[1] == "Node" and 'inflow' in other_state[1])): # node type
                                            A.loc[[state],[other_state]] = 'd'
                                            #print(A)
                                        elif ((other_step[1] == "Link" and "L" in other_state[1]) 
                                            or (other_step[1] == "Link" and 'flow' in other_state[1])):
                                            A.loc[[state],[other_state]] = 'd'
                                            #print(A)
                                for control_asset in pystorms_scenario.config['action_space']:
                                    if other_step[0] == control_asset[0]:
                                        B.loc[[state],[control_asset]] = 'd'
                                        #print(B)
                                if last_step: # just look at the next little bit downstream for backwater effects
                                    break
                                    
                                
                        elif ((step[1] == "Link" and "L" in state[1]) 
                                or (step[1] == "Link" and 'flow' in state[1])):
                            for other_step in path_of_travel:
                                if path_of_travel.index(step) - path_of_travel.index(other_step) > 1: # other step is not immediately upstream
                                    continue
                                if other_step == step:
                                    last_step = True # we only want to look a limited distance downstream
                                    continue # this is the same step, so skip it
                                for other_state in pystorms_scenario.config['states']:
                                    if other_step[0] == other_state[0]: # same id
                                        if ((other_step[1] == "Node" and "N" in other_state[1]) 
                                            or (other_step[1] == "Node" and 'flooding' in other_state[1]) 
                                            or (other_step[1] == "Node" and 'inflow' in other_state[1])): # node type
                                            A.loc[[state],[other_state]] = 'd'
                                            #print(A)
                                        elif ((other_step[1] == "Link" and "L" in other_state[1]) 
                                            or (other_step[1] == "Link" and 'flow' in other_state[1])):
                                            A.loc[[state],[other_state]] = 'd'
                                            #print(A)
                                for control_asset in pystorms_scenario.config['action_space']:
                                    if other_step[0] == control_asset[0]:
                                        B.loc[[state],[control_asset]] = 'd'
                                if last_step: # just look at the next little bit downstream for backwater effects
                                    break
                for action in pystorms_scenario.config['action_space']:
                    if step[0] == action[0] or step[0] == action:
                        print(step)
                        print(action)
                   '''     

            #print(A)
            #print(B)
        
    # add "i's" on the diagonal of A (instantaneous autocorrelatoin)
    for idx in A.index:
        A.loc[[idx],[idx]] = 'i'
    # fill the na's in A and B with 'n'
    A.fillna('n',inplace=True)
    B.fillna('n',inplace=True)
    
    # concatenate the A and B matrices column-wise and return that result
    causative_topology = pd.concat([A,B],axis=1)

    #print(causative_topology)

    return causative_topology


# this is for visuzliation, not building models.
# to build models, use the function above
def subway_map_from_pystorms(pystorms_scenario):
    # remove any duplicates in the state or action space of the config
    # this is an error within pystorms
    pystorms_scenario.config['states'] = list(dict.fromkeys(pystorms_scenario.config['states']))
    pystorms_scenario.config['action_space'] = list(dict.fromkeys(pystorms_scenario.config['action_space']))

    # make the index the concatentation of the states and action space
    index = list(list(pystorms_scenario.config['states']) + list(pystorms_scenario.config['action_space']))
   


    adjacency = pd.DataFrame(index = index , columns = index ).fillna(0)
    
    
    # use pyswmm to iterate through the network
    with pyswmm.Simulation(pystorms_scenario.config['swmm_input']) as sim:
        # start at each subcatchment and iterate down to the outfall
        # this should work even in the case of multiple outfalls
        # this should capture all the causation, because ultimately everything is precip driven
        
        # so i can view these while debugging
        Subcatchments = pyswmm.Subcatchments(sim)
        Nodes = pyswmm.Nodes(sim)
        Links = pyswmm.Links(sim)

        for subcatch in pyswmm.Subcatchments(sim):
            #print(adjacency)
            #print(subcatch.subcatchmentid)
            # create a string that records the path we travel to get to the outfall
            path_of_travel = list()
            # can i grab the rain gage id?
            path_of_travel.append((subcatch.subcatchmentid,"Subcatchment"))
            current_id = subcatch.connection # grab the id of the next object downstream
            
            
            try: # if the downstream connection is a subcatchment
                current = Subcatchments[current_id]
                current_id = current.subcatchmentid
                subcatch = Subcatchments[current_id]
                current_id = subcatch.connection # grab the id of the next object downstream
                path_of_travel.append((current_id,'Subcatchment'))
            except Exception as e:
                #print("downstream connection was not another subcatchment")
                #print(e)
                pass
            
            # other option is that downstream connection is a node
            # in which case we'll start iterating down through nodes and links to the outfall
            current = Nodes[current_id]
            path_of_travel.append((current_id,'Node'))
            while not current.is_outfall():
                #print(path_of_travel)
                # if the current object is a node, iterate through the links to find the downstream object
                if current_id in pyswmm.Nodes(sim):
                    for link in pyswmm.Links(sim):
                        #print(link.linkid)
                        if link.inlet_node == current_id:
                            path_of_travel.append((link.linkid,"Link"))
                            current_id = link.outlet_node
                            path_of_travel.append((current_id,"Node"))
                            break
                    else:
                        print("current element is a sink (no link draining). verify this is correct")
                        print(current_id)
                        break
                # if the current object is a link, grab the downstream node
                elif current_id in pyswmm.Links(sim):
                    path_of_travel.append((link.linkid,"Link"))
                    current_id = current.outlet_node
                    path_of_travel.append((current_id,"Node"))
                

                current = Nodes[current_id]

            #print("path of travel")
            #print(path_of_travel)
            # cut all the entries in path_of_travel that are not observable states or actions
            original_path_of_travel = path_of_travel.copy()
            
            for step in original_path_of_travel:
                step_is_state = False
                step_is_control_input = False
                for state in pystorms_scenario.config['states']:
                    if step[0] == state[0]: # same id
                        if ((step[1] == "Node" and "N" in state[1]) 
                        or (step[1] == "Node" and 'flooding' in state[1]) 
                        or (step[1] == "Node" and 'inflow' in state[1])
                        or (step[1] == "Link" and "L" in state[1]) 
                        or (step[1] == "Link" and 'flow' in state[1])): # types match
                            step_is_state = True
                for control_input in pystorms_scenario.config['action_space']:
                    if step[0] == control_input:
                        step_is_control_input = True
                if not step_is_state and not step_is_control_input:
                    path_of_travel.remove(step) # this will change the index, hence the "while"
                   
            #print("full path of travel")
            #print(original_path_of_travel)
            #print("observable path of travel")
            #print(path_of_travel)
           
            
            # iterate through the path of travel and rename the steps to align with the columns of the adjacency
            for step in path_of_travel:
                for state in pystorms_scenario.config['states']:
                    if step[0] == state[0]: # same id
                        if ((step[1] == "Node" and "N" in state[1]) 
                        or (step[1] == "Node" and 'flooding' in state[1]) 
                        or (step[1] == "Node" and 'inflow' in state[1])
                        or (step[1] == "Link" and "L" in state[1]) 
                        or (step[1] == "Link" and 'flow' in state[1])): # types match    
                            path_of_travel[path_of_travel.index(step)] = state

                for control_input in pystorms_scenario.config['action_space']:
                    if step[0] == control_input:
                        path_of_travel[path_of_travel.index(step)] = control_input

                       
            #print("observable path of travel")
            #print(path_of_travel)

            # now, use this path of travel to update the adjacency
            
            # only use "i" if the entries have the same id. otherwise characterize everything as delayed, "d"
            # because our path of travel only includes the observable states and the action space, we just need to look immediately up and downstream
            # only looking upstream would simplify things and be sufficient for many scenarios, but it would miss backwater effects
            for step in path_of_travel: # all of these are either observable states or actions
                if path_of_travel.index(step) == 0: # first entry, previous step not meaningful
                    prev_step = False
                else:    
                    prev_step = path_of_travel[path_of_travel.index(step)-1]
                if path_of_travel.index(step) == len(path_of_travel)-1: # last entry, next step not meaningful
                    next_step = False
                else:
                    next_step = path_of_travel[path_of_travel.index(step)+1]

                # formatted as from row to column
                if prev_step:
                    adjacency.loc[[prev_step],[step]] = 1
                if next_step:
                    adjacency.loc[[step],[next_step]] = 1
            
    # some of the networks aren't completely dendritic and so require some manual connections to be added
    if pystorms_scenario.config['name'] == 'alpha':
        adjacency.iloc[7,26] = 1 # R1 to Or1
        adjacency.iloc[26,25] = 1 # Or1 to I5
        adjacency.iloc[9,14] = 1 # R3 to JC3a
        adjacency.iloc[14,19] = 1 # JC3a to C3b
        adjacency.iloc[19,20] = 1 # C3b to C4a
        adjacency.iloc[12,7] = 1 # JC1b to R1

    graph = nx.from_pandas_adjacency(adjacency,create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(graph):
        print("graph is not a DAG")
        plt.figure(figsize=(20,10))
        pos = nx.planar_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_size=500)
        nx.draw_networkx_labels(graph, pos, font_size=12)
        nx.draw_networkx_edges(graph, pos, arrows=True,arrowsize=30,style='solid',alpha=1.0)
        plt.show()
        
    # we're gauranteed to have a directed acycilce graph, so get the topological generations and use that as the subset key
    gens = nx.topological_generations(graph)
    gen_idx = 1
    for generation in gens:
        #print(generation)
        for node in graph.nodes:
            if node in generation:
                graph.nodes[node]['generation'] = gen_idx
        gen_idx += 1
    
    # but to draw without overlaps, we need to partition by the root node, not the generation
    # give each node a key corresponding to its most distant ancestor
    # then, we can use that key to partition the nodes and draw them in separate columns
    for node in graph.nodes:
        #print(node)
        #print(nx.ancestors(graph,node))
        ancestors = nx.ancestors(graph,node)
        most_distant_ancestor = node
        for ancestor in ancestors:
            distance = nx.shortest_path_length(graph,ancestor,node)
            if distance > nx.shortest_path_length(graph,most_distant_ancestor,node):
                most_distant_ancestor = ancestor
        graph.nodes[node]['root'] = most_distant_ancestor
        #print(most_distant_ancestor)
        
            
    return {'adjacency':adjacency,'index':index,'graph':graph}
                     

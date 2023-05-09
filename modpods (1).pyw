import pandas as pd
import numpy as np
import pysindy as ps
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt


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
# RETURNS:
# models for each number of transformations from min to max
# NOTE: this code works for MIMO models, however, if output variables are dependent on each other
# poor simulation fidelity is likely due to their errors contributing to each other
# if the learned dynamics are highly accurate such that errors do not grow too large in any dependent variable, a MIMO model should work fine
# if you anticipate significant errors in the simulation of any dependent variable, you should use multiple MISO models instead
# as the model predicts derivatives, system_data must represent a *causal* system
# that is, forcing and the response to that forcing cannot occur at the same timestep
# it may be necessary for the user to shift the forcing data back to make the system causal (especially for time aggregated data like daily rainfall-runoff)
def delay_io_model(system_data, dependent_columns, independent_columns, windup_timesteps=0,init_transforms=1, max_transforms=4, max_iter=250, poly_order=3, transform_dependent=False, verbose=True, extra_verbose=False, include_bias=False, include_interaction=False):
    forcing = system_data[independent_columns].copy(deep=True)
    orig_forcing_columns = forcing.columns
    response = system_data[dependent_columns].copy(deep=True)

    results = dict() # to store the optimized models for each number of transformations

    # the transformation factors should be pandas dataframes where the index is which transformation it is and the columns are the variables
    shape_factors = pd.DataFrame(columns = forcing.columns, index = range(init_transforms, max_transforms+1))
    shape_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
    scale_factors = pd.DataFrame(columns = forcing.columns, index = range(init_transforms, max_transforms+1))
    scale_factors.iloc[0,:] = 1 # first transformation is [1,1,0] for each input
    loc_factors = pd.DataFrame(columns = forcing.columns, index = range(init_transforms, max_transforms+1))
    loc_factors.iloc[0,:] = 0 # first transformation is [1,1,0] for each input
    print(shape_factors)
    print(scale_factors)
    print(loc_factors)
    # first transformation is [1,1,0] for each input
    '''
    shape_factors = np.ones(shape=(forcing.shape[1] , init_transforms)   )
    scale_factors = np.ones(shape=(forcing.shape[1] , init_transforms)   )
    loc_factors = np.zeros(shape=(forcing.shape[1] , init_transforms)   )
    '''
    speeds =  list([500,200,50,10, 5,2, 1.1, 1.05,1.01]) 

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
            print("starting factors for additional transformation\nshape\nscale\nlocation")
            print(shape_factors)
            print(scale_factors)
            print(loc_factors)



        prev_model = SINDY_delays_MI(shape_factors, scale_factors, loc_factors, system_data.index, forcing, response,verbose, poly_order , include_bias, include_interaction,windup_timesteps)

        print("\nInitial model:\n")
        print("R^2")
        print(prev_model['error_metrics']['r2'])
        print("shape factors")
        print(shape_factors)
        print("scale factors")
        print(scale_factors)
        print("location factors")
        print(loc_factors)
        print("\n")

        for iterations in range(0,max_iter ):
            tuning_input = orig_forcing_columns[(iterations // num_transforms) %  len(orig_forcing_columns)] # row =  iter // width % height
            tuning_line = iterations % num_transforms + 1 # col =  % width (plus one because there's no zeroth transformation)
            print(str("tuning input: {i} | tuning transformation: {l:g}".format(i=tuning_input,l=tuning_line)))


            sooner_locs = loc_factors.copy(deep=True)
            sooner_locs[tuning_input][tuning_line] = float(loc_factors[tuning_input][tuning_line] - speed/10  )
            if ( sooner_locs[tuning_input][tuning_line] < 0):
                sooner = {'error_metrics':{'r2':-1}}
            else:
                sooner = SINDY_delays_MI(shape_factors ,scale_factors ,sooner_locs, 
                system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)
      
      
            later_locs = loc_factors.copy(deep=True)
            later_locs[tuning_input][tuning_line] = float ( loc_factors[tuning_input][tuning_line]  +   1.01*speed/10 )
            later = SINDY_delays_MI(shape_factors , scale_factors,later_locs, 
                system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)
      

            shape_up = shape_factors.copy(deep=True)
            shape_up[tuning_input][tuning_line] = float ( shape_factors[tuning_input][tuning_line]*speed*1.01 )
            shape_upped = SINDY_delays_MI(shape_up , scale_factors, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)
      
            shape_down = shape_factors.copy(deep=True)
            shape_down[tuning_input][tuning_line] = float ( shape_factors[tuning_input][tuning_line]/speed )
            if (shape_down[tuning_input][tuning_line] < 1):
                shape_downed = {'error_metrics':{'r2':-1}} # return a score of negative one as this is illegal
            else:
                shape_downed = SINDY_delays_MI(shape_down , scale_factors, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)

            scale_up = scale_factors.copy(deep=True)
            scale_up[tuning_input][tuning_line] = float(  scale_factors[tuning_input][tuning_line]*speed*1.01 )
            scaled_up = SINDY_delays_MI(shape_factors , scale_up, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)


            scale_down = scale_factors.copy(deep=True)
            scale_down[tuning_input][tuning_line] = float ( scale_factors[tuning_input][tuning_line]/speed )
            scaled_down = SINDY_delays_MI(shape_factors , scale_down, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)
      
            # rounder
            rounder_shape = shape_factors.copy(deep=True)
            rounder_shape[tuning_input][tuning_line] = shape_factors[tuning_input][tuning_line]*(speed*1.01)
            rounder_scale = scale_factors.copy(deep=True)
            rounder_scale[tuning_input][tuning_line] = scale_factors[tuning_input][tuning_line]/(speed*1.01)
            rounder = SINDY_delays_MI(rounder_shape , rounder_scale, loc_factors, 
                                    system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)

            # sharper
            sharper_shape = shape_factors.copy(deep=True)
            sharper_shape[tuning_input][tuning_line] = shape_factors[tuning_input][tuning_line]/speed
            if (sharper_shape[tuning_input][tuning_line] < 1):
                sharper = {'error_metrics':{'r2':-1}} # lower bound on shape to avoid inf
            else:
                sharper_scale = scale_factors.copy(deep=True)
                sharper_scale[tuning_input][tuning_line] = scale_factors[tuning_input][tuning_line]*speed
                sharper = SINDY_delays_MI(sharper_shape ,sharper_scale,loc_factors, 
                                            system_data.index, forcing, response, extra_verbose, poly_order , include_bias, include_interaction,windup_timesteps)


    

            scores = [prev_model['error_metrics']['r2'], shape_upped['error_metrics']['r2'], shape_downed['error_metrics']['r2'], 
                      scaled_up['error_metrics']['r2'], scaled_down['error_metrics']['r2'], sooner['error_metrics']['r2'], 
                      later['error_metrics']['r2'], rounder['error_metrics']['r2'], sharper['error_metrics']['r2'] ]
            #print(scores)

            if (sooner['error_metrics']['r2'] >= max(scores) and sooner['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = sooner.copy()
                loc_factors = sooner_locs.copy(deep=True)
            elif (later['error_metrics']['r2'] >= max(scores) and later['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = later.copy()
                loc_factors = later_locs.copy(deep=True)
            elif(shape_upped['error_metrics']['r2'] >= max(scores) and shape_upped['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = shape_upped.copy()
                shape_factors = shape_up.copy(deep=True)
            elif(shape_downed['error_metrics']['r2'] >=max(scores) and shape_downed['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = shape_downed.copy()
                shape_factors = shape_down.copy(deep=True)
            elif(scaled_up['error_metrics']['r2'] >= max(scores) and scaled_up['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = scaled_up.copy()
                scale_factors = scale_up.copy(deep=True)
            elif(scaled_down['error_metrics']['r2'] >= max(scores) and scaled_down['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = scaled_down.copy()
                scale_factors = scale_down.copy(deep=True)
            elif (rounder['error_metrics']['r2'] >= max(scores) and rounder['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = rounder.copy()
                shape_factors = rounder_shape.copy(deep=True)
                scale_factors = rounder_scale.copy(deep=True)
            elif (sharper['error_metrics']['r2'] >= max(scores) and sharper['error_metrics']['r2'] > prev_model['error_metrics']['r2']):
                prev_model = sharper.copy()
                shape_factors = sharper_shape.copy(deep=True)
                scale_factors = sharper_scale.copy(deep=True)
            # the middle was best, but it's bad, tighten up the bounds (if we're at the last tuning line of the last input)

            elif( num_transforms == tuning_line and tuning_input == orig_forcing_columns[-1]): 
                speed_idx = speed_idx + 1
                print("\n\ntightening bounds\n\n")

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
                print("\n")

    
    
        final_model = SINDY_delays_MI(shape_factors, scale_factors ,loc_factors,system_data.index, forcing, response, verbose, poly_order , include_bias, include_interaction,windup_timesteps)
        results[num_transforms] = {'final_model':final_model.copy(), 'shape_factors':shape_factors.copy(deep=True), 'scale_factors':scale_factors.copy(deep=True), 'loc_factors':loc_factors.copy(deep=True)}
    return results


def SINDY_delays_MI(shape_factors, scale_factors, loc_factors, index, forcing, response, final_run, poly_degree, include_bias, include_interaction,windup_timesteps):
    forcing = transform_inputs(shape_factors, scale_factors,loc_factors, index, forcing)

    feature_names =  response.columns.tolist() +  forcing.columns.tolist()

    # SINDy
    model = ps.SINDy(
        differentiation_method= ps.FiniteDifference(),
        feature_library=ps.PolynomialLibrary(degree=poly_degree,include_bias = include_bias, include_interaction=include_interaction), 
        optimizer = ps.STLSQ(threshold=0), 
        feature_names = feature_names
    )

    # windup latent states (if your windup is too long, this will error)
    model.fit(response.values[windup_timesteps:,:], t = np.arange(0,len(index),1)[windup_timesteps:], u = forcing.values[windup_timesteps:,:])
    r2 = model.score(response.values[windup_timesteps:,:],t=np.arange(0,len(index),1)[windup_timesteps:],u=forcing.values[windup_timesteps:,:]) # training data score
    # r2 is how well we're doing across all the outputs. that's actually good to keep model accuracy lumped because that's what makes most sense to drive the optimization
    # even though the metrics we'll want to evaluate models on are individual output accuracy
    print("training R^2", r2)
    model.print(precision=5)
    # return false for things not evaluated / don't exist
    error_metrics = {"MAE":False,"RMSE":False,"NSE":False,"alpha":False,"beta":False,"HFV":False,"HFV10":False,"LFV":False,"FDC":False,"r2":r2}
    simulated = False
    if (final_run): # only simulate final runs because it's slow
        # try: once in high volume training put this back in, but want to see the errors during development
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
            hfv.append(np.sum(np.sort(simulated[:,col_idx])[-int(0.02*len(index)):])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.02*len(index)):]))
            hfv10.append(np.sum(np.sort(simulated[:,col_idx])[-int(0.1*len(index)):])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.1*len(index)):]))
            lfv.append(np.sum(np.sort(simulated[:,col_idx])[:int(0.3*len(index))])/np.sum(np.sort(response.values[windup_timesteps+1:,col_idx])[:int(0.3*len(index))]))
            fdc.append(np.mean(np.sort(simulated[:,col_idx])[-int(0.6*len(index)):-int(0.4*len(index))])/np.mean(np.sort(response.values[windup_timesteps+1:,col_idx])[-int(0.6*len(index)):-int(0.4*len(index))]))

        print("MAE = ", mae)
        print("RMSE = ", rmse)
        # all the below error metrics were generated by copilot and should be checked
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


        '''
        except:
            print("Simulation diverged.")
            return {"error_metrics": False, "model": model, "simulated": False, "response": response, "forcing": forcing, "index": index}

        '''
          
    return {"error_metrics": error_metrics, "model": model, "simulated": simulated, "response": response, "forcing": forcing, "index": index}
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
    return forcing
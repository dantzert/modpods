import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import modpods

# basic funcionality tests and a bit of a tutorial

# some data from the CAMELS dataset
# change the filepath to wherever you have modpods at
# "C:\modpods\03439000_05_model_output.txt"
filepath = "./03439000_05_model_output.txt"


df = pd.read_csv(filepath, sep='\s+')
print(df)
print(df.columns)
# combine the columns YR, MNTH, DY, and YR into a single datetime column
df.rename({'YR':'year','MNTH':'month','DY':'day','HR':'hour'},axis=1,inplace=True)
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])

# set the index to the datetime column
df.set_index('datetime',inplace=True)
# shift the forcing back one timestep (one day) to make the system causal

print(df[['OBS_RUN','RAIM']])
df.RAIM = df.RAIM.shift(-1)
df.dropna(inplace=True)
print(df[['OBS_RUN','RAIM']])


# for better results (and slower run) up the max iterations, model complexity (poly_order and max_transforms), and the number of years used to train



# drop all columns except for RAIM (surface water input) and OBS_RUN (observed runoff) for actual CAMELS training
# but for testing the MIMO delay_io_model I want multiple inputs and multiple outputs
windup_timesteps = 30 # days of windup
years = 1
df_train = df.iloc[:365*years + windup_timesteps,:] # total data used, actually trained on this less the windup period
df_eval = df.iloc[-(365*years + windup_timesteps):,:] # data for evaluation, not used in training
 
#df.plot(y=['OBS_RUN','RAIM'])
#plt.show()



#df['ones'] = np.ones(len(df.OBS_RUN)) # to make sure MIMO error metrics are working correctly
print(df_train)
forcing_coef_constraints = {'RAIM':-1, 'PET':1,'PRCP':-1}
df_train = df_train[['OBS_RUN','RAIM','PET','PRCP']]
rainfall_runoff_model = modpods.delay_io_train(df_train, ['OBS_RUN'],['RAIM','PET','PRCP'],windup_timesteps=windup_timesteps,
                       init_transforms=1, max_transforms=1,max_iter=10, verbose=True, forcing_coef_constraints= forcing_coef_constraints,
                       poly_order=1, bibo_stable=False)


print(rainfall_runoff_model)
print(rainfall_runoff_model[1])
print("error metrics")
print(rainfall_runoff_model[1]['final_model']['error_metrics'])
#print(rainfall_runoff_model[2]['final_model']['error_metrics'])
#print(rainfall_runoff_model[3]['final_model']['error_metrics'])
print("shapes")
print(rainfall_runoff_model[1]['shape_factors'])
#print(rainfall_runoff_model[2]['shape_factors'])
#print(rainfall_runoff_model[3]['shape_factors'])

# plot the results
fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(df_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['response']['OBS_RUN'][windup_timesteps+1:],label='observed')
ax.plot(df_train.index[windup_timesteps+1:],rainfall_runoff_model[1]['final_model']['simulated'][:,0],label='simulated')
#ax.set_title('1 transformation')
ax.legend()
plt.title("training")
'''
ax[1].plot(df.index[windup_timesteps+1:],rainfall_runoff_model[2]['final_model']['response']['OBS_RUN'][windup_timesteps+1:],label='observed')
ax[1].plot(df.index[windup_timesteps+1:],rainfall_runoff_model[2]['final_model']['simulated'][:,0],label='simulated')
ax[1].set_title('2 transformations')
ax[1].legend()
ax[2].plot(df.index[windup_timesteps+1:],rainfall_runoff_model[3]['final_model']['response']['OBS_RUN'][windup_timesteps+1:],label='observed')
ax[2].plot(df.index[windup_timesteps+1:],rainfall_runoff_model[3]['final_model']['simulated'][:,0],label='simulated')
ax[2].set_title('3 transformations')
ax[2].legend()
'''
plt.show()
plt.close('all')



# now test prediction / evaluation
eval_sim = modpods.delay_io_predict(rainfall_runoff_model, df_eval, 1,evaluation=True)
print("error metrics")
print(eval_sim['error_metrics'])
fig, ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(df_eval.index[windup_timesteps+1:],df_eval['OBS_RUN'][windup_timesteps+1:],label='observed')
ax.plot(df_eval.index[windup_timesteps+1:],eval_sim['prediction'],label='simulated')
#ax.set_title('1 transformation')
ax.legend()
plt.title("evaluation")
plt.show()




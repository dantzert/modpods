import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import modpods

import control as ct

cartoon = False
plot = False

# set the random seed
np.random.seed(0)

# define an LTI matrix where I know what the causative topology (connections) should look like
# this is an easy test case. one input affects two states, which then affect the output. so only four connections total
A = np.diag(-1*np.ones(10))

# define cascade connections
A[1,0] = 1
A[2,1] = 1
A[3,2] = 1
A[4,3] = 1
A[5,4] = 1 # two reservoirs
A[6,5] = 1
A[7,6] = 1

# define connections to output
A[9,7] = -1
A[9,8] = 1

# define input
B = np.zeros(shape=(10,2))
B[0,0]= 1
B[8,1] = 1

C = np.zeros(shape=(3,10))
C[0,2] = 1 # x2 is observed
C[1,8] = 1 # x8 is observed
C[2,9] = 1 # x9 is observed

parallel_reservoirs = ct.ss(A,B,C,0,inputs=['u1','u2'],outputs=['x2','x8','x9'])
time_base = 150
# dt = .1
# dt = 0.5
# dt = 1 
# dt = 2

dt = .05


if cartoon:
    dt = 0.05
u = np.zeros((int(time_base / dt),2))

u[int(25/dt):int(50/dt),0] = np.random.rand(len(u[int(25/dt):int(50/dt),0]))-0.5
u[int(5/dt):int(20/dt),1] = np.random.rand(len(u[int(5/dt):int(20/dt),1]))-0.5
u[abs(u) < 0.40] = 0 # make it sparse (~80% of timesteps set to zero)
u[:,0] = u[:,0]*np.random.rand(len(u))*1000
u[:,1] = u[:,1]*np.random.rand(len(u))*100


if cartoon: # make the forcing simple
    u = np.zeros((int(time_base / dt),2))
    u[int(5/dt):int(6/dt),0] = 1
    u[int(0/dt):int(1/dt),1] = 1


T = np.arange(0,time_base,dt)
response = ct.forced_response(parallel_reservoirs,T,np.transpose(u))

system_data = pd.DataFrame(index=T)
system_data['u1'] = response.inputs[0][:]
system_data['u2'] = response.inputs[1][:]
system_data['x2'] = response.outputs[0][:]
system_data['x8'] = response.outputs[1][:]
system_data['x9'] = response.outputs[2][:]

'''
system_data['x2'] = response.states[2][:]
system_data['x8'] = response.states[8][:]
system_data['x9'] = response.states[9][:]
'''


if plot:
    system_data.plot(figsize=(10,5), subplots=True,legend=True)
    plt.show()
    if cartoon:
        # also make a more cartoony version
        cartoon_plot_data = system_data.copy()
        # normalize all the magnitudes in cartoon_plot_data such as that all columns have maximum of 1
        for col in cartoon_plot_data.columns:
            cartoon_plot_data[col] = cartoon_plot_data[col]/np.max(np.abs(cartoon_plot_data[col]))

        cartoon_plot_data.iloc[:int(len(cartoon_plot_data)/10)].plot(figsize=(5,5), subplots=False,legend=True,fontsize='xx-large',style=['r','b','g','m','k'],xticks=[],yticks=[],linewidth=5)
        plt.gca().axis('off') # get rid of bounding box
        plt.savefig("C:/modpods/test_lti_system_gen_cartoon.svg")
        plt.show()
    
forcing_coef_constraints = dict()
for column in system_data.columns:
    if 'u' in column:
        forcing_coef_constraints[column] = -1 # assume both forcing coefficients are negative
        # they are both actually positive, so the coefficients would only be negative if the constraint was in action

forcing_coef_constraints['u1'] = 1 # make u1 (and its powers and transformations) positive  
forcing_coef_constraints['u2'] = -1 # make u2 (and its powers and transformations) positive
rainfall_runoff_model = modpods.delay_io_train(system_data,dependent_columns = ['x8'],independent_columns=['u1','u2'],windup_timesteps = 0, 
                                                           init_transforms=1,max_transforms=1,max_iter=100,
                                                           poly_order = 1, verbose = True,bibo_stable=True, forcing_coef_constraints=forcing_coef_constraints)


print(rainfall_runoff_model)

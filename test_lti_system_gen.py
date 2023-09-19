import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import modpods

import control as ct

cartoon=False
plot = False

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
A[9,7] = 1
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
dt = 1 
# dt = 2
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
    # also make a more cartoony version
    cartoon_plot_data = system_data.copy()
    # normalize all the magnitudes in cartoon_plot_data such as that all columns have maximum of 1
    for col in cartoon_plot_data.columns:
        cartoon_plot_data[col] = cartoon_plot_data[col]/np.max(np.abs(cartoon_plot_data[col]))

    cartoon_plot_data.iloc[:int(len(cartoon_plot_data)/10)].plot(figsize=(5,5), subplots=False,legend=True,fontsize='xx-large',style=['r','b','g','m','k'],xticks=[],yticks=[],linewidth=5)
    plt.gca().axis('off') # get rid of bounding box
    plt.savefig("G:/My Drive/modpods/test_lti_system_gen_cartoon.svg")
    plt.show()
    



# define the causative topology
# if this wasn't known a priori, we could use the modpods.infer_causative_topology function to find it
# but that function is expensive so I excluded it from this testing script
causative_topology = pd.DataFrame(index=['u1','u2','x2','x8','x9'], columns=['u1','u2','x2','x8','x9']).fillna('n')
causative_topology.loc['x2','u1'] = 'd'
causative_topology.loc['x8','u2'] = 'i'
causative_topology.loc['x9','x8'] = 'i'
causative_topology.loc['x9','x2'] = 'd'
print(causative_topology)
# max iterations is for learning the delay model. increase it for better accuracy
lti_sys_from_data = modpods.lti_system_gen(causative_topology,system_data,['u1','u2'],['x2','x8','x9'],max_iter=100,bibo_stable=True)

# print all columns of the pandas dataframe
pd.set_option('display.max_columns', None)
print("final lti system")
print("A")
print(lti_sys_from_data['A'])
print("B")
print(lti_sys_from_data['B'])
print("C")
print(lti_sys_from_data['C'])

# how good is the system response aproximation for the training forcing?
approx_response = ct.forced_response(lti_sys_from_data['system'],T,np.transpose(u))
approx_data = pd.DataFrame(index=T)
approx_data['u1'] = approx_response.inputs[0][:]
approx_data['u2'] = approx_response.inputs[1][:]
approx_data['x2'] = approx_response.outputs[0][:]
approx_data['x8'] = approx_response.outputs[1][:]
approx_data['x9'] = approx_response.outputs[2][:]


# create a vertical subplot of 3 axes
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
# plot the error of each output
output_columns = ['x2','x8','x9']
for idx in range(len(output_columns)):
    axes[idx,0].plot(system_data[output_columns[idx]],label='actual')
    axes[idx,0].plot(approx_data[output_columns[idx]],label='approx')
    if idx == 0:
        axes[idx,0].legend(fontsize='x-large',loc='best')
    axes[idx,0].set_ylabel(output_columns[idx],fontsize='xx-large')
    if idx == len(output_columns)-1:
        axes[idx,0].set_xlabel("time",fontsize='x-large')

# label the left column of plots "training"
axes[0,0].set_title("training",fontsize='xx-large')


# what about a different forcing? (test case)
u2 = np.zeros((int(time_base / dt),2))
u2[int(0/dt):int(35/dt),0] = np.random.rand(len(u2[int(0/dt):int(35/dt),0]))-0.5
u2[int(10/dt):int(80/dt),1] = np.random.rand(len(u2[int(10/dt):int(80/dt),1]))-0.5
u2[abs(u2) < 0.40] = 0 # make it sparse
u2[:,0] = u2[:,0]*np.random.rand(len(u2))*1000
u2[:,1] = u2[:,1]*np.random.rand(len(u2))*100

approx_response = ct.forced_response(lti_sys_from_data['system'],T,np.transpose(u2))
approx_data = pd.DataFrame(index=T)
approx_data['u1'] = approx_response.inputs[0][:]
approx_data['u2'] = approx_response.inputs[1][:]
approx_data['x2'] = approx_response.outputs[0][:]
approx_data['x8'] = approx_response.outputs[1][:]
approx_data['x9'] = approx_response.outputs[2][:]

actual_response = ct.forced_response(parallel_reservoirs,T,np.transpose(u2))
actual_data = pd.DataFrame(index=T)
actual_data['u1'] = actual_response.inputs[0][:]
actual_data['u2'] = actual_response.inputs[1][:]
actual_data['x2'] = actual_response.outputs[0][:]
actual_data['x8'] = actual_response.outputs[1][:]
actual_data['x9'] = actual_response.outputs[2][:]


for idx in range(len(output_columns)):
    axes[idx,1].plot(actual_data[output_columns[idx]],label='actual')
    axes[idx,1].plot(approx_data[output_columns[idx]],label='approx')
    if idx == len(output_columns)-1:
        axes[idx,1].set_xlabel("time",fontsize='x-large')
    
axes[0,1].set_title("testing",fontsize='xx-large')
plt.tight_layout()
plt.savefig("G:/My Drive/modpods/test_lti_system_gen.png")
plt.savefig("G:/My Drive/modpods/test_lti_system_gen.svg")
plt.show()
#plt.close()



# now try LQR using u1 (slow) as the disturbance and u2 (fast) as the control
# make the objective to minimize the magnitude of x9

# define the cost function
Q = np.eye(len(lti_sys_from_data['A'].columns))*0 # no states matter
Q[lti_sys_from_data['A'].columns.get_loc('x9'),lti_sys_from_data['A'].columns.get_loc('x9')] = 10e6 # other than x9
R = np.eye(len(lti_sys_from_data['B'].columns)) / 10e6 # don't constrain the control effort
# define the system
B_u = lti_sys_from_data['B'].values # use u2 as the control
B_u[:,0] = 0 # don't use u1 as the control
B_d = lti_sys_from_data['B'].values # use u1 as the disturbance
B_d[:,1] = 0 # don't use u2 as the disturbance
sys_response_to_control = ct.ss(lti_sys_from_data['A'],B_u,lti_sys_from_data['C'],0) # just for defining the controller gain
# find the state feedback gain for the linear quadratic regulator
K,S,E = ct.lqr(sys_response_to_control,Q,R) # one row of K should be zeros to reflect that u1 is not used as a control but is the disturbance


# find the estimator gain for the kalman filter
# observe performance degrade as the assumed measurement noise covariance increases (slower poles on the observer)
assumed_noise_levels = [10e-8, 10e-1]

# plot the results
fig,axes = plt.subplots(len(output_columns),2,figsize=(8,8))
graph_labels = ['d','u','y']

# define the disturbance (only through the slow, u1 channel)
d = np.zeros((int(time_base / dt),2))
d[int(0/dt):int(35/dt),0] = np.random.rand(len(d[int(0/dt):int(35/dt),0]))-0.5
d[abs(d) < 0.40] = 0 # make it sparse
d[:,0] = d[:,0]*np.random.rand(len(d))*1000

for noise_level_idx in range(2):
    noisiness = assumed_noise_levels[noise_level_idx]
    L,P,E = ct.lqe(lti_sys_from_data['system'],np.eye(len(lti_sys_from_data['B'].columns)),noisiness*np.eye(len(lti_sys_from_data['C'].index)) ) # unit covariance on process noise and measurement error 

    # define the observer based compensator (per freudenberg 560 course notes 2.4)
    obc_A = lti_sys_from_data['A'].values-lti_sys_from_data['B'].values@K - L@lti_sys_from_data['C'].values
    obc = ct.ss(obc_A, L, K, 0, inputs=['x2_m','x8_m','x9_m'],outputs=['u1','u2']) # K positive because negative feedback is assumed

    # define the closed loop system with the original plant and the observer based compensator designed using the identified model
    closed_loop = ct.feedback(parallel_reservoirs,obc)

    obc_feedback_control = ct.forced_response(closed_loop,T,np.transpose(d))
    obc_feedback_data = pd.DataFrame(index=T)
    obc_feedback_data['u1'] = obc_feedback_control.inputs[0][:]
    obc_feedback_data['u2'] = obc_feedback_control.inputs[1][:]
    obc_feedback_data['x2'] = obc_feedback_control.outputs[0][:]
    obc_feedback_data['x8'] = obc_feedback_control.outputs[1][:]
    obc_feedback_data['x9'] = obc_feedback_control.outputs[2][:]

    uncontrolled = ct.forced_response(parallel_reservoirs,T,np.transpose(d))
    uncontrolled_data = pd.DataFrame(index=T)
    uncontrolled_data['u1'] = uncontrolled.inputs[0][:]
    uncontrolled_data['u2'] = uncontrolled.inputs[1][:]
    uncontrolled_data['x2'] = uncontrolled.outputs[0][:]
    uncontrolled_data['x8'] = uncontrolled.outputs[1][:]
    uncontrolled_data['x9'] = uncontrolled.outputs[2][:]

    
    for idx in range(len(output_columns)):
        axes[idx,noise_level_idx].plot(uncontrolled_data[output_columns[idx]],label='uncontrolled')
        axes[idx,noise_level_idx].plot(obc_feedback_data[output_columns[idx]],label='obc feedback')
        if idx == 0:
            axes[idx,noise_level_idx].legend(fontsize='x-large')
            if noise_level_idx == 0:
                axes[idx,noise_level_idx].set_title("measurements assumed clean",fontsize='large')
            else:
                axes[idx,noise_level_idx].set_title("measurements assumed noisy",fontsize='large')
        if idx == len(output_columns)-1:
            axes[idx,noise_level_idx].set_xlabel("time",fontsize='x-large')
        if noise_level_idx == 0:
            axes[idx,noise_level_idx].set_ylabel(graph_labels[idx],fontsize='xx-large',rotation='horizontal')
    
plt.tight_layout()
plt.savefig("G:/My Drive/modpods/test_lti_system_gen_obc.png")
plt.savefig("G:/My Drive/modpods/test_lti_system_gen_obc.svg")
plt.show()

print("done")

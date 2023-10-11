# reference: https://colab.research.google.com/github/kLabUM/pystorms/blob/master/tutorials/Scenario_Gamma.ipynb

import sys
#from modpods import topo_from_pystorms
#sys.path.append("G:/My Drive/modpods")
import modpods
import pystorms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import control as ct
import dill as pickle



# uncontrolled
env = pystorms.scenarios.gamma()
done = False
print("simulating uncontrolled case")
while not done:
    # Query the current state of the simulation
    state = env.state()
    
    # Initialize actions to have each asset open
    actions = np.ones(11)
    
    # Set the actions and progress the simulation
    done = env.step(actions)
    
# Calculate the performance measure for the uncontrolled simulation
uncontrolled_perf = sum(env.data_log["performance_measure"])
'''
print("The calculated performance for the uncontrolled case of Scenario gamma is:")
print("{:.4e}".format(uncontrolled_perf))

basin_max_depths = [5., 10., 10., 10.]

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(np.asarray(env.data_log['depthN']['1'])/basin_max_depths[0], label='Basin 1')
plt.plot(np.asarray(env.data_log['depthN']['2'])/basin_max_depths[1], label='Basin 2')
plt.plot(np.asarray(env.data_log['depthN']['3'])/basin_max_depths[2], label='Basin 3')
plt.plot(np.asarray(env.data_log['depthN']['4'])/basin_max_depths[3], label='Basin 4')
plt.xlabel('Simulation Timestep')
plt.ylabel('Filling Degree')
plt.legend()

plt.subplot(1,2,2)
plt.plot(env.data_log['flow']['O1'], label='Basin 1')
plt.plot(env.data_log['flow']['O2'], label='Basin 2')
plt.plot(env.data_log['flow']['O3'], label='Basin 3')
plt.plot(env.data_log['flow']['O4'], label='Basin 4')
plt.xlabel('Simulation Timestep')
plt.ylabel('Basin Outflow (cfs)')
plt.legend()
#plt.show()
plt.close()
'''
uncontrolled_data = env.data_log

def controller_efd(state, max_depths):
    # Initialize the action space so that we can compute the new settings
    new_settings = np.ones(len(state))
    # Set equal filling degree parameters
    c = 1.5
    theta = 0.25
    
    # Assign the current depth in each basin
    depths = state
    
    # Compute the filling degrees
    fd = depths/max_depths
    # Compute the average filling degree across each controlled basin
    fd_average = sum(fd)/len(fd)
    
    # Update each valve setting based on the relative fullness of each basin
    for i in range(0,len(fd)):
        
        # If a basin is very full compared to the average, we should open its
        # valve to release some water
        if fd[i] > fd_average:
            new_settings[i] = c*(fd[i]-fd_average)
        
        # If a basin's filling degree is close to the average (within some value
        # theta), its setting can be close to that average
        elif fd_average-fd[i] <= theta:
            new_settings[i] = fd_average
            
        # If a basin is very empty compared to the average, we can close its
        # valve to store more water at that location, prioritizing releasing at
        # the other locations
        else:
            new_settings[i] = 0.
        
        # Make sure the settings are in bounds [0,1]
        new_settings[i] = min(new_settings[i], 1.)
        new_settings[i] = max(new_settings[i], 0.)

    return new_settings


env = pystorms.scenarios.gamma()
done = False

# Specify the maximum depths for each basin we are controlling
basin_max_depths = [5., 10., 10., 10.]
print("simulating equal filling degree")
while not done:
    # Query the current state of the simulation
    state = env.state()
    # Isolate only the states that we need (the 4 downstream basin depths)
    states_relevant = state[0:4]
    
    # Pass the current, relevant states and the maximum basin 
    # depths into our equal filling degree logic
    actions_efd = controller_efd(states_relevant, basin_max_depths)
    # Specify that the other 7 valves in the network should be 
    # open since we are not controlling them here
    actions_uncontrolled = np.ones(7)
    # Join the two above action arrays
    actions = np.concatenate((actions_efd, actions_uncontrolled), axis=0)
    
    # Set the actions and progress the simulation
    done = env.step(actions)
    
# Calculate the performance measure for the uncontrolled simulation
equalfilling_perf = sum(env.data_log["performance_measure"])
ef_data = env.data_log
'''
print("The calculated performance for the equal filling degree case of Scenario gamma is:")
print("{}.".format(equalfilling_perf))

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(np.asarray(env.data_log['depthN']['1'])/basin_max_depths[0], label='Basin 1')
plt.plot(np.asarray(env.data_log['depthN']['2'])/basin_max_depths[1], label='Basin 2')
plt.plot(np.asarray(env.data_log['depthN']['3'])/basin_max_depths[2], label='Basin 3')
plt.plot(np.asarray(env.data_log['depthN']['4'])/basin_max_depths[3], label='Basin 4')
plt.xlabel('Simulation Timestep')
plt.ylabel('Filling Degree')
plt.legend()

plt.subplot(1,2,2)
plt.plot(env.data_log['flow']['O1'], label='Basin 1')
plt.plot(env.data_log['flow']['O2'], label='Basin 2')
plt.plot(env.data_log['flow']['O3'], label='Basin 3')
plt.plot(env.data_log['flow']['O4'], label='Basin 4')
plt.xlabel('Simulation Timestep')
plt.ylabel('Basin Outflow (cfs)')
plt.legend()
'''

'''
ef_flows = pd.DataFrame.from_dict(ef_data['flow'])
ef_flows.columns = env.config['action_space']
ef_depthN = pd.DataFrame.from_dict(ef_data['depthN'])
ef_depthN.columns = env.config['states']
ef_response = pd.concat([ef_flows, ef_depthN], axis=1)
ef_response.index = env.data_log['simulation_time']
print(ef_response)
# for the columns of ef_response which do not contain the string "O" (i.e. the depths)
# if the current column name is "X", make it "(X, depthN)"
# this is because the modpods library expects the depths to be named "X, depthN"
# where X is the name of the corresponding flow
for col in ef_response.columns:
    if "O" in col: # the orifices
        # if there's a number in that column name that's greater than 4, drop this column
        # this is because we're only controlling the first 4 orifices and these measurements are redundant to the storage node depths if the orifices are always open
        if int(col[1:]) > 4:
            ef_response.drop(columns=col, inplace=True)    

        ef_response.rename(columns={col: (col, "flow")}, inplace=True)
        

    
print(ef_response)
'''

# do a training simulation such that the flows are actually independent of the depths
# in both the uncontrolled and efd scenarios the flows at the orifice are highly coupled to the depths
env = pystorms.scenarios.gamma()
done = False

# Specify the maximum depths for each basin we are controlling
basin_max_depths = [5., 10., 10., 10.]
actions_characterize = np.ones(4)
step = 0
print("running characterization simulation")
while not done:

    if step % 1000 == 0:
        actions_characterize = np.ones(4)*0.3 # mostly close all the valves
        actions_characterize[np.random.randint(0,4)] = np.random.rand() # open one valve a random amount

    actions_uncontrolled = np.ones(7)
    # Join the two above action arrays
    actions = np.concatenate((actions_characterize, actions_uncontrolled), axis=0)
    
    # Set the actions and progress the simulation
    done = env.step(actions)
    step += 1
random_perf = sum(env.data_log["performance_measure"])
print("performance of characterization:")
print("{:.4e}".format(random_perf))
'''
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(np.asarray(env.data_log['depthN']['1'])/basin_max_depths[0], label='Basin 1')
plt.plot(np.asarray(env.data_log['depthN']['2'])/basin_max_depths[1], label='Basin 2')
plt.plot(np.asarray(env.data_log['depthN']['3'])/basin_max_depths[2], label='Basin 3')
plt.plot(np.asarray(env.data_log['depthN']['4'])/basin_max_depths[3], label='Basin 4')
plt.xlabel('Simulation Timestep')
plt.ylabel('Filling Degree')
plt.legend()

plt.subplot(1,2,2)
plt.plot(env.data_log['flow']['O1'], label='Basin 1')
plt.plot(env.data_log['flow']['O2'], label='Basin 2')
plt.plot(env.data_log['flow']['O3'], label='Basin 3')
plt.plot(env.data_log['flow']['O4'], label='Basin 4')
plt.xlabel('Simulation Timestep')
plt.ylabel('Basin Outflow (cfs)')
plt.legend()
plt.show()
'''

training_data = env.data_log
training_flows = pd.DataFrame.from_dict(training_data['flow'])
training_flows.columns = env.config['action_space']
training_depthN = pd.DataFrame.from_dict(training_data['depthN'])
training_depthN.columns = env.config['states']
training_response = pd.concat([training_flows, training_depthN], axis=1)
training_response.index = env.data_log['simulation_time']
print(training_response)


# for the columns of training_response which do not contain the string "O" (i.e. the depths)
# if the current column name is "X", make it "(X, depthN)"
# this is because the modpods library expects the depths to be named "X, depthN"
# where X is the name of the corresponding flow
for col in training_response.columns:
    if "O" in col: # the orifices
        # if there's a number in that column name that's greater than 4, drop this column
        # this is because we're only controlling the first 4 orifices and these measurements are redundant to the storage node depths if the orifices are always open
        if int(col[1:]) > 4:
            training_response.drop(columns=col, inplace=True)    

        training_response.rename(columns={col: (col, "flow")}, inplace=True)
        

#training_response['rain'] = 0
#training_response['rain'][0] = 1 # just an impulse that says "something happened". not even real rain data

print(training_response) 



# for debugging resample to a coarser time step (native resolution is about one minute but not consistent)
# need a consistent time step for modpods
orig_index_length = len(training_response.index)
training_response = training_response.resample('10T',axis='index').mean().copy(deep=True)
training_dt =  orig_index_length / len(training_response.index)
print(training_response)
# get rid of the initial filling, this is confusing to the model because the forcing (rainfall) isn't observed so the storage nodes seem to rise autonomously
# could also include dummy forcing
training_response = training_response.iloc[60:,:] # start ten hours in
print(training_response)

# we'll only use the training response to infer the topology and dynamics
# for this experiment assume all of flow O1-O11 and depth 1-11 are observable
# but only O1-O4 are controllable 
#independent_columns = training_response.columns[0:4] # orifices O1 through O4
#dependent_columns = training_response.drop(columns=independent_columns).columns

dependent_columns = training_response.columns[4:8] # just depths at 1 through 4
independent_columns = training_response.drop(columns = dependent_columns).columns

use_blind = False
# learn the topology from the data
# this will be the "blind" plant model
if use_blind: # don't have this on all the time because it's very expensive
    blind_topo = modpods.infer_causative_topology(training_response, dependent_columns = dependent_columns,
                                                  independent_columns = independent_columns, verbose=True,swmm=True)

    print(blind_topo.causative_topo)
    print(blind_topo.total_graph)


# read the topology from the swmm file (this is much cheaper)
env.config['states'] = dependent_columns
env.config['action_space'] = independent_columns 
# the default is controlling all 11 orifices so we need to edit the environment
print("defining topology")
swmm_topo = modpods.topo_from_pystorms(env)
#swmm_topo['rain'] = 'd' 
# define delayed connection to rain, delayed impact on all storage nodes
#independent_columns.append('rain')

# the index of the causative topology should be the dependent columns
#swmm_topo.index = dependent_columns
# the columns of the causative topology should be the dependent columns plus the independent columns
#swmm_topo.columns = dependent_columns.append(independent_columns)

# show all columns when printing dataframes
pd.set_option('display.max_columns', None)
#print(swmm_topo)

if use_blind:
    print("differences in topology")
    print(blind_topo.causative_topo.compare(swmm_topo))

# learn the dynamics now, or load a previously learned model
'''
# learn the dynamics from the trainingd response
print("learning dynamics")
lti_plant_approx_seeing = modpods.lti_system_gen(swmm_topo, training_response, 
                                                 independent_columns= independent_columns,
                                                 dependent_columns = dependent_columns, max_iter = 100,
                                                 swmm=True,bibo_stable=True,max_transition_state_dim=5)
# pickle the plant approximation to load later
with open('G:/My Drive/modpods/swmm_lti_plant_approx_seeing.pickle', 'wb') as handle:
    pickle.dump(lti_plant_approx_seeing, handle)
'''

# load the plant approximation from a pickle
with open('G:/My Drive/modpods/swmm_lti_plant_approx_seeing.pickle', 'rb') as handle:
    print("loading previously trained model")
    lti_plant_approx_seeing = pickle.load(handle)
    
if use_blind:
    lti_plant_approx_blind = modpods.lti_system_gen(blind_topo, training_response, 
                                                 independent_columns= independent_columns,
                                                 dependent_columns = dependent_columns)



# is the plant approximation internally stable?
plant_eigenvalues,_ = np.linalg.eig(lti_plant_approx_seeing['A'].values)

# cast the columns of dataframes to strings for easier indexing
training_response.columns = training_response.columns.astype(str)
dependent_columns = [str(col) for col in dependent_columns]
independent_columns = [str(col) for col in independent_columns]
# reindex the training_response to an integer step
training_response.index = np.arange(0,len(training_response),1)
'''
# evaluate the plant approximation accuracy
# only plot the depths at 1, 2, 3, and 4
# the forcing is the flows at O1, O2, O3, and O4
approx_response = ct.forced_response(lti_plant_approx_seeing['system'], U=np.transpose(training_response[independent_columns].values), T=training_response.index.values)
approx_data = pd.DataFrame(index=training_response.index.values)
approx_data[dependent_columns[0]] = approx_response.outputs[0][:]
approx_data[dependent_columns[1]] = approx_response.outputs[1][:]
approx_data[dependent_columns[2]] = approx_response.outputs[2][:]
approx_data[dependent_columns[3]] = approx_response.outputs[3][:]

output_columns = dependent_columns[0:4] # depth at 1,2,3,4

# create a vertical subplot of 3 axes
fig, axes = plt.subplots(4, 1, figsize=(10, 10))

for idx in range(len(output_columns)):
    axes[idx].plot(training_response[output_columns[idx]],label='actual')
    axes[idx].plot(approx_data[output_columns[idx]],label='approx')
    if idx == 0:
        axes[idx].legend(fontsize='x-large',loc='best')
    axes[idx].set_ylabel(output_columns[idx],fontsize='large')
    if idx == len(output_columns)-1:
        axes[idx].set_xlabel("time",fontsize='x-large')
# label the left column of plots "training"
axes[0].set_title("outputs",fontsize='xx-large')

plt.savefig("G:/My Drive/modpods/test_lti_control_of_swmm_plant_approx.png")
plt.savefig("G:/My Drive/modpods/test_lti_control_of_swmm_plant_approx.svg")
#plt.show()
plt.close()
# same plot, but just the first few timesteps (this is the accuracy that matters for feedback control)
# create a vertical subplot of 3 axes
fig, axes = plt.subplots(4, 1, figsize=(10, 10))

for idx in range(len(output_columns)):
    axes[idx].plot(training_response[output_columns[idx]][:10],label='actual')
    axes[idx].plot(approx_data[output_columns[idx]][:10],label='approx')
    if idx == 0:
        axes[idx].legend(fontsize='x-large',loc='best')
    axes[idx].set_ylabel(output_columns[idx],fontsize='large')
    if idx == len(output_columns)-1:
        axes[idx].set_xlabel("time",fontsize='x-large')
# label the left column of plots "training"
axes[0].set_title("outputs",fontsize='xx-large')

plt.savefig("G:/My Drive/modpods/test_lti_control_of_swmm_plant_approx_first10.png")
plt.savefig("G:/My Drive/modpods/test_lti_control_of_swmm_plant_approx_first10.svg")
#plt.show()
plt.close()
'''
# define the cost function
Q = np.eye(len(lti_plant_approx_seeing['A'].columns)) / 10e12 # we don't want to penalize the transition states as their magnitude doesn't have directly tractable physical meaning
# bryson's rule based on the maxiumum depth of each basin
# note: the swmm file gamma.inp actually specifies the maximum depth of storage node 1 as 10 feet, 
# but i'll be consistent with the configuration of the efd controller for consistent comparison
basin_max_depths_all = [5.0, 10.0, 10.0, 10.0, 10.0,20.0, 10.0, 10.0,10.0, 13.72, 14.96]
for asset_index in range(len(dependent_columns)):
    Q[lti_plant_approx_seeing['A'].columns.get_loc(dependent_columns[asset_index]),lti_plant_approx_seeing['A'].columns.get_loc(dependent_columns[asset_index])] = 1 / ((basin_max_depths_all[asset_index])**2 )

flood_weighting = 1.5 # how much more important is flooding than the other objectives?
Q = Q * flood_weighting # set flood_weighting = 1 for basic bryson's rule (care about flows as much as flooding)
# threshold on flows at 0.11 m^3 / s which is 3.9 cfs
R = np.eye(4) / (3.9**2) # bryson's rule on maximum allowable flow
# define the system
# sys_response_to_control = ct.ss(lti_plant_approx_seeing['A'],lti_plant_approx_seeing['B'],lti_plant_approx_seeing['C'],0) # just for defining the controller gain 
# (not necessary in this case, but would be if you've got disturbances)
# find the state feedback gain for the linear quadratic regulator
print("defining controller")
K,S,E = ct.lqr(lti_plant_approx_seeing['A'],lti_plant_approx_seeing['B'].values[:,0:4],Q,R) # only the first four columns of B represent control inputs, the rest are disturbances 
#print("feedback poles (A-BK)")
feedback_poles,_ = np.linalg.eig(lti_plant_approx_seeing['A'].values - lti_plant_approx_seeing['B'].values[:,0:4]@K)
#print(feedback_poles)
# define the observer gain

# "fast" relative to the controller
print("defining observer")
#L = ct.place(np.transpose(lti_plant_approx_seeing['A'].values), np.transpose(lti_plant_approx_seeing['C'].values),5*np.real(feedback_poles))


# based on assumed noise
measurement_noise = 0.05 # more measurement noise means a worse sensor
process_noise = 1 # more process noise means a worse model
L,P,E = ct.lqe(lti_plant_approx_seeing['system'],process_noise*np.eye(len(lti_plant_approx_seeing['B'].columns)),measurement_noise*np.eye(len(lti_plant_approx_seeing['C'].index)) ) # unit covariance on process noise and measurement error 
#print("observer poles")

observer_poles,_ =  np.linalg.eig(lti_plant_approx_seeing['A'].values - L@lti_plant_approx_seeing['C'].values)
#print(observer_poles)

'''
# define the observer based compensator (per freudenberg 560 course notes 2.4)
obc_A = lti_plant_approx_seeing['A'].values-lti_plant_approx_seeing['B'].values@K - L@lti_plant_approx_seeing['C'].values
# ingests measurements, returns control actions
obc = ct.ss(obc_A, L, -K, 0, inputs=list(lti_plant_approx_seeing['C'].index),outputs=list(lti_plant_approx_seeing['B'].columns)) # negate K to give back commanded flows which are positive


# need to separately define the observer and controller because we can't close the loop in the typical way
# the observer takes the control input and measured output as inputs and outputs the estimated full state
#observer_input = np.concatenate((lti_plant_approx_seeing['B'].values,L),axis = 1)
#observer = ct.ss(lti_plant_approx_seeing['A'] - L@lti_plant_approx_seeing['C'].values, observer_input, np.eye(len(lti_plant_approx_seeing['A']) ) , 0 )

# the controller takes in an estiamte of the state and returns a control command
# this is just, u = -K @ xhat, not necessary to define a state space model for that as there's no evolution

# can't form the closed loop system because the plant is not a state space system, but rather a software model
# the state estimate and control actions will be computed iteratively as the simulation is stepped through
'''

env = pystorms.scenarios.gamma()
done = False

u = np.zeros((4,1) ) # start with all orifices completely closed
u_open_pct = np.zeros((4,1)) # start with all orifices completely closed
xhat = np.zeros((len(lti_plant_approx_seeing['A'].columns),1)) # initial state estimate

steps = 0 # make sure the estimator and controller operate at the frequency the approxiation was trained at

# convert control command (flow) into orifice open percentage
# per the EPA-SWMM user manual volume ii hydraulics, orifices (section 6.2, page 107) - https://nepis.epa.gov/Exe/ZyPDF.cgi/P100S9AS.PDF?Dockey=P100S9AS.PDF 
# all orifices in gamma are "bottom"
Cd = 0.65 # happens to be the same for all of them
Ao = 1 # area is one square foot. again, happens to be the same for all of them. 
g = 32.2 # ft / s^2
# the expression for discharge is found using Torricelli's equation: Q = Cd * (Ao*open_pct) sqrt(2*g*H_e)
# H_e is the effective head in feet, which is just the depth in the basin as the orifices are "bottom"
# to get the action command as a percent open, we solve as: open_pct = Q_desired / (Cd * Ao * sqrt(2*g*H_e))

while not done:
    # Query the current state of the simulation
    observables = env.state()
    y_measured = observables[:4].reshape(-1,1) # depths at 1-4
    d = observables[4:].reshape(-1,1) # "disturbances", depths at 5-11


    # for updating the plant, calculate the "u" that is actually applied to the plant, not the desired control input
    for idx in range(len(u)):
        u[idx,0] = Cd*Ao*u_open_pct[idx,0]*np.sqrt(2*g*observables[idx]) # calculate the actual flow through the orifice

    # update the observer based on these measurements -> xhat_dot = A xhat + B u + L (y_m - C xhat)
    # right now the state evolution is exactly cancelling the observer. why would this be?
    state_evolution = lti_plant_approx_seeing['A'].values @ xhat 
    # causing the state estimation to diverge to crazy values
    impact_of_control = lti_plant_approx_seeing['B'].values @ np.concatenate((u,d),axis=0) 
    yhat = lti_plant_approx_seeing['C'].values @ xhat # just for reference, could be useful for plotting later
    y_error =  y_measured - yhat # cast observables to be 2 dimensional
    output_updating = L @ y_error 
    xhat_dot = (state_evolution + impact_of_control + output_updating) / training_dt # divide by the training frequency to get the change in state over the time step
    #xhat_dot = (impact_of_control + output_updating) / training_dt # don't listen to the state dynamics, just use them to define the feedback -> that lost stability
    xhat += xhat_dot # update the state estimate
    
    # note that this is only truly the error if there is zero error in the measurements. Generally, data is not truth.
    u = -K @ xhat # calculate control command

        
    u_open_pct = u*-1
    
    for idx in range(len(u)):
        head = 2*g*observables[idx]
        
        if head < 0.01: # if the head is less than 0.01 ft, the basin is empty, so close the orifice
            u_open_pct[idx,0] = 0
        else:
            u_open_pct[idx,0] = u[idx,0] / (Cd*Ao * np.sqrt(2*g*observables[idx])) # open percentage for desired flow rate
        
        if u_open_pct[idx,0] > 1: # if the calculated open percentage is greater than 1, the orifice is fully open
            u_open_pct[idx,0] = 1
        elif u_open_pct[idx,0]< 0: # if the calculated open percentage is less than 0, the orifice is fully closed
            u_open_pct[idx,0] = 0

   
    # Specify that the other 7 valves in the network should be 
    # open since we are not controlling them here
    actions_uncontrolled = np.ones(7)
    # Join the two above action arrays
    actions = np.concatenate((u_open_pct.flatten(), actions_uncontrolled), axis=0)
    
    # Set the actions and progress the simulation
    done = env.step(actions)
    steps += 1
    
    if steps % 1000 == 0: 
        print("u_open_pct")
        print(u_open_pct)
        print("yhat")
        print(yhat)
        print("y error")
        print(y_error)
        

    
# Calculate the performance measure for the uncontrolled simulation
obc_perf = sum(env.data_log["performance_measure"])
obc_data = env.data_log

print("The calculated performance for the uncontrolled case of Scenario gamma is:")
print("{:.4e}".format(uncontrolled_perf))
print("for the equal filling degree case of Scenario gamma is:")
print("{:.4e}".format(equalfilling_perf))
print("for the random control case:")
print("{:.4e}".format(random_perf))
print("for the observer based compensator case of Scenario gamma is:")
print("{:.4e}".format(obc_perf))

fig,axes = plt.subplots(4,2,figsize=(20,10))
fig.suptitle("Pystorms Scenario Gamma")
axes[0,0].set_title("Valves",fontsize='xx-large')
axes[0,1].set_title("Storage Nodes",fontsize='xx-large')

valves = ["O1","O2","O3","O4"]
storage_nodes = ["1","2","3","4"]
cfs2cms = 35.315
ft2meters = 3.281
# plot the valves
for idx in range(4):
    axes[idx,0].plot(uncontrolled_data['simulation_time'],np.array(uncontrolled_data['flow'][valves[idx]])/cfs2cms,label='Uncontrolled',color='k',linewidth=2)
    axes[idx,0].plot(ef_data['simulation_time'],np.array(ef_data['flow'][valves[idx]])/cfs2cms,label='Equal Filling',color='b',linewidth=2)
    axes[idx,0].plot(obc_data['simulation_time'],np.array(obc_data['flow'][valves[idx]])/cfs2cms,label='LTI Feedback',color='g',linewidth=2)
    # add a dotted red line indicating the flow threshold
    axes[idx,0].hlines(3.9/cfs2cms, uncontrolled_data['simulation_time'][0],uncontrolled_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
    #axes[idx,0].set_ylabel( str(  str(valves[idx]) + " Flow" ),rotation='horizontal',labelpad=8)
    axes[idx,0].annotate(str(  str(valves[idx]) + " Flow" ),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    if idx == 0:
        axes[idx,0].legend(fontsize='xx-large')
    if idx != 3:
        axes[idx,0].set_xticks([])
       

# plot the storage nodes
for idx in range(4):
    axes[idx,1].plot(uncontrolled_data['simulation_time'],np.array(uncontrolled_data['depthN'][storage_nodes[idx]])/ft2meters,label='Uncontrolled',color='k',linewidth=2)
    axes[idx,1].plot(ef_data['simulation_time'],np.array(ef_data['depthN'][storage_nodes[idx]])/ft2meters,label='Equal Filling',color='b',linewidth=2)
    axes[idx,1].plot(obc_data['simulation_time'],np.array(obc_data['depthN'][storage_nodes[idx]])/ft2meters,label='LTI Feedback',color='g',linewidth=2)
    #axes[idx,1].set_ylabel( str( str(storage_nodes[idx]) + " Depth"),rotation='horizontal',labelpad=8)
    axes[idx,1].annotate( str( str(storage_nodes[idx]) + " Depth"),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    
    # add a dotted red line indicating the depth threshold
    axes[idx,1].hlines(basin_max_depths_all[idx]/ft2meters,uncontrolled_data['simulation_time'][0],uncontrolled_data['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
    if idx != 3:
        axes[idx,1].set_xticks([])

plt.tight_layout()
plt.savefig("G:/My Drive/modpods/pystorms_gamma_comparison.png",dpi=450)
plt.savefig("G:/My Drive/modpods/pystorms_gamma_comparison.svg",dpi=450)
#plt.show()

print("done")


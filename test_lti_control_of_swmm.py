# reference: https://colab.research.google.com/github/kLabUM/pystorms/blob/master/tutorials/Scenario_Gamma.ipynb

import sys

from modpods import topo_from_pystorms
sys.path.append("G:/My Drive/modpods")
import modpods
import pystorms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

use_blind = False

'''
# uncontrolled
env = pystorms.scenarios.gamma()
done = False

while not done:
    # Query the current state of the simulation
    state = env.state()
    
    # Initialize actions to have each asset open
    actions = np.ones(11)
    
    # Set the actions and progress the simulation
    done = env.step(actions)
    
# Calculate the performance measure for the uncontrolled simulation
uncontrolled_perf = sum(env.data_log["performance_measure"])

print("The calculated performance for the uncontrolled case of Scenario gamma is:")
print("{}.".format(uncontrolled_perf))

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
plt.show()

uncontrolled_data = env.data_log
'''
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

print("The calculated performance for the equal filling degree case of Scenario gamma is:")
print("{}.".format(equalfilling_perf))
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
'''
ef_data = env.data_log

# get the responses into one dataframe with zero pads in between 
#uncontrolled_flows = pd.DataFrame.from_dict(uncontrolled_data['flow'])
#uncontrolled_depthN = pd.DataFrame.from_dict(uncontrolled_data['depthN'])
#uncontrolled_response = pd.concat([uncontrolled_flows, uncontrolled_depthN], axis=1)
#print(uncontrolled_response)
ef_flows = pd.DataFrame.from_dict(ef_data['flow'])
ef_flows.columns = env.config['action_space']
ef_depthN = pd.DataFrame.from_dict(ef_data['depthN'])
ef_depthN.columns = env.config['states']
ef_response = pd.concat([ef_flows, ef_depthN], axis=1)
print(ef_response)
# for the columns of ef_response which do not contain the string "O" (i.e. the depths)
# if the current column name is "X", make it "(X, depthN)"
# this is because the modpods library expects the depths to be named "X, depthN"
# where X is the name of the corresponding flow
for col in ef_response.columns:
    if "O" in col:
        ef_response.rename(columns={col: (col, "flow")}, inplace=True)
print(ef_response)


# we'll only use the ef response to infer the topology and dynamics
# for this experiment assume all of flow O1-O11 and depth 1-11 are observable
# but only O1-O4 are controllable 
independent_columns = ef_response.columns[0:4] # orifices O1 through O4
dependent_columns = ef_response.drop(columns=independent_columns).columns


# learn the topology from the data
# this will be the "blind" plant model
if use_blind: # don't have this on all the time because it's very expensive
    blind_topo = modpods.infer_causative_topology(ef_response, dependent_columns = dependent_columns,
                                                  independent_columns = independent_columns, verbose=True)

    print(blind_topo.causative_topo)
    print(blind_topo.total_graph)


# read the topology from the swmm file (this is much cheaper)
env.config['states'] = dependent_columns
env.config['action_space'] = independent_columns 
# the default is controlling all 11 orifices so we need to edit the environment
swmm_topo = topo_from_pystorms(env)
# the index of the causative topology should be the dependent columns
#swmm_topo.index = dependent_columns
# the columns of the causative topology should be the dependent columns plus the independent columns
#swmm_topo.columns = dependent_columns.append(independent_columns)

# show all columns when printing dataframes
pd.set_option('display.max_columns', None)
print(swmm_topo)

if use_blind:
    print("differences in topology")
    print(blind_topo.causative_topo.compare(swmm_topo))

# learn the dynamics from the efd response
lti_plant_approx_seeing = modpods.lti_system_gen(swmm_topo, ef_response, 
                                                 independent_columns= independent_columns,
                                                 dependent_columns = dependent_columns, max_iter = 0)
print("A")
print(lti_plant_approx_seeing['A'])
print("B")
print(lti_plant_approx_seeing['B'])
print("C")
print(lti_plant_approx_seeing['C'])

if use_blind:
    lti_plant_approx_blind = modpods.lti_system_gen(blind_topo, ef_response, 
                                                 independent_columns= independent_columns,
                                                 dependent_columns = dependent_columns)



# for now, just demo real-time LQR control
# define the cost matrices using bryson's rule 
# create a weighting variable between flooding and flows to prioritize flood avoidance





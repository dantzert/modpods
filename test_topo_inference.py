import numpy as np
import pandas as pd
import scipy.stats as stats
import os as os
import matplotlib.pyplot as plt
import modpods as modpods
import control as ct


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
#A[9,4] = 1
A[9,7] = 1
#A[9,8] = 1
A[9,8] = 1

# define input
B = np.zeros(shape=(10,2))
B[0,0]= 1
#B[5] = 2
B[8,1] = 1

C = np.eye(10)
D = np.zeros(shape=(10,2))

parallel_reservoirs = ct.ss(A,B,C,D)
'''
response = ct.impulse_response(parallel_reservoirs)
plt.figure(figsize=(10,5))
states_to_plot = np.array([0,8,9])
for state in states_to_plot:
  plt.plot(response.outputs[state][0], '--', label=str(state))

plt.legend()
plt.title("impulse response")
plt.show()
'''

time_base = 50
dt = .05
u = np.zeros((int(time_base / dt),2))

#u[2020:2120,1] = np.random.rand(100) -0.5

#u[1220:1320,1] = np.random.rand(100) -0.5

# u1 -> x5 -> x9
#u[3180:3280,0] = np.random.rand(100) -0.5
u[int(25/dt):int(40/dt),0] = np.random.rand(len(u[int(25/dt):int(40/dt),0]))-0.5
u[int(0/dt):int(15/dt),1] = np.random.rand(len(u[int(0/dt):int(15/dt),1]))-0.5
u[abs(u) < 0.40] = 0 # make it sparse
u[:,0] = u[:,0]*np.random.rand(len(u))*1000
u[:,1] = u[:,1]*np.random.rand(len(u))*100


#u[700:800,0] = 10

#u[900:950,1] = -10

#u[1100:1400,0] = 2
#u[1600:1750,1] = -7
#u[0:20,1] = 20
#u[int(15/dt),1] = -50
#u[int(25/dt):int(35/dt),0] = 6
#u[int(29/dt):int(31/dt),0] = -10
#u[int(25/dt):int(35/dt),1] = np.random.rand(len(u[int(25/dt):int(35/dt),1]) )*5 - 2.5
# u2 -> x8 -> x9
#u[2020:2120,1] = np.random.rand(100) -0.5

#u[1220:1320,1] = np.random.rand(100) -0.5

# u1 -> x5 -> x9
#u[3180:3280,0] = np.random.rand(100) -0.5

#u[abs(u) < 0.45] = 0 # make it sparse
#u[:600,0] = u[:600,0]*np.random.rand(600)*100
#u[:600,1] = u[:600,1]*np.random.rand(600)*100




T = np.arange(0,time_base,dt)
response = ct.forced_response(parallel_reservoirs,T,np.transpose(u))

system_data = pd.DataFrame(index=T)
system_data['u1'] = response.inputs[0][:]
system_data['u2'] = response.inputs[1][:]
system_data['x2'] = response.states[2][:]
system_data['x8'] = response.states[8][:]
system_data['x9'] = response.states[9][:]

plot = False

system_data.plot(figsize=(10,5), subplots=True,legend=True)
plt.savefig('test_topo_inference.png')
if plot:
    plt.show()
plt.close('all')

# now try to infer the topology
# assume i know this is data from a drainage system and so I assume it's a directed acyclic graph
causative_topo = modpods.infer_causative_topology(system_data,dependent_columns=['x2','x8','x9'], 
    independent_columns=['u1','u2'], verbose=True, max_iter=0, method='granger')
# the correct answer here is that there is immediate causation flowing from u2 to x8 to x9
# and there is delayed causation flowing from u1 to x2 to x9

#modpods.delay_io_train(system_data, dependent_columns=['x5','x8','x9'], independent_columns=['u1','u2'], 
#    max_transforms=1,max_iter=250,poly_order=1,transform_dependent=True,bibo_stable=True,verbose=True)

print("done")
# a randomized (but stable) lti system
'''
A = np.matrix(np.random.rand(5,5)) - 0.5
A = (A - np.diag(np.ones(5)))


# add an oscillatory pair
A = np.concatenate((A , 0.001*(np.random.rand(2,5) -0.5) ), axis=0)
A = np.concatenate((A, np.array([[0,0,],[0,0],[0,0],[0,0],[0,0] , [-0.3,-1], [1,-0.3]])), axis=1)
#A = [[0,-1], [1,0]]

print(A)
A = A/100
l,v = np.linalg.eig(A)
print(l)

B = (np.random.rand(7,1) - 0.5)*4



C = np.eye(7)
D = np.zeros(shape=(7,1))


random_system = ct.ss(A,B,C,D)

response = ct.impulse_response(random_system)
plt.figure(figsize=(25,10))
for state in range(7):
  plt.plot(response.outputs[state][0], label=str(state))

plt.legend()
plt.title("impulse response")


u = np.concatenate( (np.zeros(100), np.ones(10), np.zeros(120), -.5*np.ones(10), np.zeros(150), np.ones(10), 
                     np.zeros(70), -np.ones(1), np.zeros(190), np.ones(100), np.zeros(140) ,
                     np.linspace(-2,2,50) , np.zeros(130) , np.linspace(0,0.5,200), np.zeros(20),
                     np.sin(np.arange(0,10,0.01) ) , np.zeros(30), np.ones(500), np.zeros(1000)  ) )

#u = np.concatenate( (np.zeros(10), np.ones(1), np.zeros(500)))

response = ct.forced_response(random_system,np.arange(0,len(u)),u)

plt.figure(figsize=(25,10))
for state in range(7):
  plt.plot(response.outputs[state][:], label=str(state))

plt.plot(u, label='forcing')
plt.legend()
plt.title("forced response")
'''
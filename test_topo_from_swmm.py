import pyswmm
import pystorms 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("G:/My Drive/modpods")
import modpods

# display all columns of pandas dataframes
pd.set_option('display.max_columns', None)

#epsilon_topo = modpods.topo_from_pystorms(pystorms.scenarios.epsilon())
#theta = pystorms.scenarios.theta()
#theta_topo = modpods.topo_from_pystorms(theta)

gamma = pystorms.scenarios.gamma()
gamma.config['states'] = [gamma.config['states'][0], gamma.config['states'][3], gamma.config['states'][5], gamma.config['states'][9]]
gamma.config['action_space'] = [gamma.config['action_space'][0], gamma.config['action_space'][3], gamma.config['action_space'][5], gamma.config['action_space'][9]]
print(gamma.config['states'])
print(gamma.config['action_space'])
gamma_topo = modpods.topo_from_pystorms(gamma)
print(gamma_topo)


#beta = pystorms.scenarios.beta()
#beta_topo = modpods.topo_from_pystorms(beta)

#zeta = pystorms.scenarios.zeta()
#zeta_topo = modpods.topo_from_pystorms(zeta)

print("done")



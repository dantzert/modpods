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
theta = pystorms.scenarios.theta()
theta_topo = modpods.topo_from_pystorms(theta)




#beta = pystorms.scenarios.beta()
#beta_topo = modpods.topo_from_pystorms(beta)

#zeta = pystorms.scenarios.zeta()
#zeta_topo = modpods.topo_from_pystorms(zeta)

print("done")



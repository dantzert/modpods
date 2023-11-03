import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import modpods


# make the gamma distribution to be fit
# make shape, scale, and location random floats between 0 and 1
shape = np.random.uniform(1, 10)
scale = np.random.uniform(0, 10)
loc = np.random.uniform(0, 10)
dt = 0.1 # pretend this was the sampling frequency of the data from which we got this
# worst case input is a sudden (sharp) change after some delay
# that is, shape=1, scale small, and location large
#shape = 1#6
#scale = 1#0
#loc = 1

# will also need to match the dt at which the gamma distribution is sampled

# make something very sharp and fast
shape = 1
scale = 0.1
loc = 0

# make something very diffuse and slow - peak at 1000 timesteps with half-maximum width of about 200 timesteps
#shape = 100
#scale = 10
#loc = 0

# a transformation that generates a decay rate within the bounds should have a very accurate approximation
shape = 10
scale = 1
loc = 0


print("shape, scale, loc: ", shape, scale, loc)


approx = modpods.lti_from_gamma(shape,scale,loc,dt,verbose=True)


plt.figure(figsize=(8,6))
plt.plot(approx['t'],approx['gamma_pdf'], label="gamma pdf")
plt.plot(approx['t'],approx['lti_approx_output'], label="lti approximation")
plt.legend()
plt.show()

print("done")
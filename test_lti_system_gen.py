import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import modpods


# make the gamma distribution to be fit
# make shape, scale, and location random floats between 0 and 1
shape = np.random.uniform(1, 10)
scale = np.random.uniform(0, 10)
loc = np.random.uniform(0, 5)
# worst case input is a sudden (sharp) change after some delay
# that is, shape=1, scale small, and location large
#shape = 1#6
#scale = 1#0
#loc = 1#0


print("shape, scale, loc: ", shape, scale, loc)

approx = modpods.lti_from_gamma(shape,scale,loc,verbose=True)


plt.figure(figsize=(8,6))
plt.plot(approx['t'],approx['gamma_pdf'], label="gamma pdf")
plt.plot(approx['t'],approx['lti_approx_output'], label="lti approximation")
plt.legend()
plt.show()

print("done")
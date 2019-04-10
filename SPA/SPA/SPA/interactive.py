
from mydistribution import *
from matplotlib.pylab import plot
import numpy as np
from scipy.integrate import quad

x = np.linspace(-5, 5, 201)
gme = MyGME(10)
y = gme.tail_expectation(x)
func = lambda z, k: (z-k)*gme.density(z)
yy = np.array([quad((lambda z: func(z,k)), k, 50) for k in x])
yy = yy[:, 0]
plot(x, y - yy)
#plot(x. gme.cdf(x))
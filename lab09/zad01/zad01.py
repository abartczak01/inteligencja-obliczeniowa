import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import math
import numpy as np
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

def endurance(params):
    x, y, z, u, v, w = params
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

def f(x):
    n_particles = x.shape[0]
    j = [endurance(x[i]) for i in range(n_particles)]
    return np.array(j)

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

cost, pos = optimizer.optimize(lambda x: -f(x), iters=1000)

print("Best cost:", cost)
print("Best position:", pos)

cost_history = optimizer.cost_history

plot_cost_history(cost_history)
plt.savefig("pso-endurance.png")

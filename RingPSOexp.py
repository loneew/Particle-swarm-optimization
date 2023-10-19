from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.single import LocalBestPSO
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.formatters import Animator
from pyswarms.utils.plotters.formatters import Designer

def goldstein_price(params):
    if len(params.shape) == 1:
        params = np.array([params])
    
    num_particles = params.shape[0]
    costs = np.zeros(num_particles)
    
    for i in range(num_particles):
        x, y = params[i]
        term1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
        term2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
        costs[i] = term1 * term2
    
    return costs

if __name__ == "__main__":
    num_particles = 100
    num_iterations = 20
    fig, ax = plt.subplots()
    for index, i in enumerate([0, 0.25, 0.5, 0.75, 1]):
        optimizer = LocalBestPSO(n_particles=num_particles, dimensions=2, options={'c1': i, 'c2': 0.5, 'w': 0.5, 'k': 3, 'p': 2})
        optimizer.optimize(goldstein_price, iters=num_iterations)

        cost_history = optimizer.cost_history
        
        plt.plot(cost_history, label=str(i))
    plt.legend(title='c1', loc='upper right')
    plt.show()
    fig, ax = plt.subplots()
    for index, i in enumerate([0, 0.25, 0.5, 0.75, 1]):
        optimizer = LocalBestPSO(n_particles=num_particles, dimensions=2, options={'c1': 0.5, 'c2': i, 'w': 0.5, 'k': 3, 'p': 2})
        optimizer.optimize(goldstein_price, iters=num_iterations)

        cost_history = optimizer.cost_history
        
        plt.plot(cost_history, label=str(i))
    plt.legend(title='c2', loc='upper right')
    plt.show()
    fig, ax = plt.subplots()
    for index, i in enumerate([0, 0.25, 0.5, 0.75, 1]):
        optimizer = LocalBestPSO(n_particles=num_particles, dimensions=2, options={'c1': 0.5, 'c2': 0.5, 'w': i, 'k': 3, 'p': 2})
        optimizer.optimize(goldstein_price, iters=num_iterations)

        cost_history = optimizer.cost_history
        
        plt.plot(cost_history, label=str(i))
    plt.legend(title='w', loc='upper right')
    plt.show()
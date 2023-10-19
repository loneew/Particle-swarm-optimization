from matplotlib import pyplot as plt
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.backend.topology import VonNeumann
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.formatters import Designer


def goldstein_price(params):
    if len(params.shape) == 1:
        params = np.array([params])

    num_particles = params.shape[0]
    costs = np.zeros(num_particles)

    for i in range(num_particles):
        x, y = params[i]
        term1 = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
        term2 = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
        costs[i] = term1 * term2

    return costs


if __name__ == "__main__":
    num_particles = 100
    num_iterations = 20
    my_topology = VonNeumann(static=False)
    my_bounds = [(-2, -2), (2, 2)]
    optimizer1 = GeneralOptimizerPSO(n_particles=num_particles,
                                     dimensions=2,
                                     options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p': 2, 'r': 1},
                                     topology=my_topology)
    best_cost1, best_pos1 = optimizer1.optimize(goldstein_price, iters=num_iterations)
    cost_history1 = optimizer1.cost_history
    title1 = "Without bounds"
    print(f"Найкраща позиція: {best_pos1}")
    print(f"Значення функції в цій точці: {best_cost1}")

    optimizer2 = GeneralOptimizerPSO(n_particles=num_particles,
                                     dimensions=2,
                                     options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p': 2, 'r': 1},
                                     topology=my_topology, bounds=my_bounds)
    best_cost2, best_pos2 = optimizer2.optimize(goldstein_price, iters=num_iterations)
    cost_history2 = optimizer2.cost_history
    title2 = "With bounds"
    print(f"Найкраща позиція: {best_pos2}")
    print(f"Значення функції в цій точці: {best_cost2}")

    my_designer = Designer(title_fontsize=20, figsize=(5, 4), limits=[(-2, 2), (-2, 2)])

    plot_cost_history(cost_history1, designer=my_designer, title=title1)
    plot_cost_history(cost_history2, designer=my_designer, title=title2)
    plt.show()

from matplotlib import pyplot as plt
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.single import LocalBestPSO
from pyswarms.utils.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Animator
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
    num_iterations = 100

    optimizer = LocalBestPSO(n_particles=num_particles, dimensions=2,
                             options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2})

    best_cost, best_pos = optimizer.optimize(goldstein_price, iters=num_iterations)

    print(f"Найкраща позиція: {best_pos}")
    print(f"Значення функції в цій точці: {best_cost}")

    pos_history = optimizer.pos_history
    my_animator = Animator(interval=200)

    my_designer = Designer(limits=[(-2, 2), (-2, 2)])

    anim = plot_contour(pos_history, animator=my_animator, designer=my_designer)
    plt.show()

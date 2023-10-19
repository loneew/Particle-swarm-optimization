from matplotlib import pyplot as plt
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
from pyswarms.utils.plotters.formatters import Designer


def goldstein_price(x):
    if not x.shape[1] == 2:
        raise IndexError(
            "Goldstein function only takes two-dimensional " "input."
        )

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (
                1
                + (x_ + y_ + 1) ** 2.0
                * (
                        19
                        - 14 * x_
                        + 3 * x_ ** 2.0
                        - 14 * y_
                        + 6 * x_ * y_
                        + 3 * y_ ** 2.0
                )
        ) * (
                30
                + (2 * x_ - 3 * y_) ** 2.0
                * (
                        18
                        - 32 * x_
                        + 12 * x_ ** 2.0
                        + 48 * y_
                        - 36 * x_ * y_
                        + 27 * y_ ** 2.0
                )
        )

    return j


if __name__ == "__main__":
    num_particles = 100
    num_iterations = 20
    my_bounds = [(-2, -2), (2, 2)]
    optimizer1 = GlobalBestPSO(n_particles=num_particles, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
    best_cost1, best_pos1 = optimizer1.optimize(goldstein_price, iters=num_iterations)
    cost_history1 = optimizer1.cost_history
    title1 = "Without bounds"
    print(f"Найкраща позиція: {best_pos1}")
    print(f"Значення функції в цій точці: {best_cost1}")

    optimizer2 = GlobalBestPSO(n_particles=num_particles, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                               bounds=my_bounds)
    best_cost2, best_pos2 = optimizer2.optimize(goldstein_price, iters=num_iterations)
    cost_history2 = optimizer2.cost_history
    title2 = "With bounds"
    print(f"Найкраща позиція: {best_pos2}")
    print(f"Значення функції в цій точці: {best_cost2}")

    my_designer = Designer(title_fontsize=20, figsize=(5, 4), limits=[(-2, 2), (-2, 2)])

    plot_cost_history(cost_history1, designer=my_designer, title=title1)
    plot_cost_history(cost_history2, designer=my_designer, title=title2)
    plt.show()

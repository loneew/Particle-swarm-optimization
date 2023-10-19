from matplotlib import pyplot as plt
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.single import GlobalBestPSO
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.single import LocalBestPSO
from pyswarms.backend.topology import VonNeumann
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
    num_particles = 10
    num_iterations = 25
    opt1 = GlobalBestPSO(n_particles=num_particles, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
    opt2 = GeneralOptimizerPSO(n_particles=num_particles, 
                                    dimensions=2, 
                                    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'p': 2, 'r': 1}, 
                                    topology=VonNeumann(static=False))
    opt3 = LocalBestPSO(n_particles=num_particles, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2})

    titles = ["GlobalBestPSO", "VonNeumann", "LocalBestPSO"]
    options = [opt1, opt2, opt3]
    my_designer = Designer(title_fontsize=20, figsize=(5,4), limits=[(-2, 2), (-2, 2)])
    for index, i in enumerate(options):
        best_cost, best_pos = i.optimize(goldstein_price, iters=num_iterations)

        print(f"Найкраща позиція: {best_pos}")
        print(f"Значення функції в цій точці: {best_cost}")

        plot_cost_history(i.cost_history, title=titles[index], designer=my_designer)
    plt.show()


    
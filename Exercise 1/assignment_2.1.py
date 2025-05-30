from functools import partial
from typing import Callable

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import math

from utils.sampling import monte_carlo

Function = Callable[[float], float]


def f(x: float) -> float:
    # TODO: define the target function.
    # ====================================================================
    return math.sin(x)
    # ====================================================================


def analytical_integral(a: float, b: float) -> float:
    # TODO: compute the analytical integral of f on [a, b].
    # ====================================================================
    return -math.cos(b) + math.cos(a)
    # ====================================================================


def transform(samples: npt.NDArray, a: float, b: float) -> npt.NDArray:
    # TODO: implement the transformation of U from [0, 1] to [a, b].
    # ====================================================================
    samples = np.zeros_like(samples)
    # ====================================================================
    return samples


def integrate_mc(
    f: Function,
    a: float,
    b: float,
    n_samples: int,
    with_transform: bool = False,
    seed: int = 42,
) -> tuple[float, float]:
    # TODO: compute the integral with the Monta Carlo method.
    # Depending on 'with_transform', use the uniform distribution on [a, b]
    # directly or transform the uniform distribution on [0, 1] to [a, b].
    # Return the integral estimate and the corresponding RMSE.
    # ====================================================================

    if with_transform:
        x, y = 0, 1
    else:
        x, y = a, b

    # Calculate samples
    distr = cp.Uniform(x, y)
    samples = distr.sample(size=n_samples, seed=seed)

    if with_transform:
        oldRange = 1
        newRange = b - a
        samples = (((samples - 0)))

    # Estimating Integral
    evaluated_values = np.array([f(xi) for xi in samples])
    I_f = (1/n_samples) * np.sum(evaluated_values)

    # Analytical
    target_I_f = analytical_integral(a, b)

    # Errors
    error = np.abs(target_I_f - I_f)
    var = (1/(n_samples - 1)) * np.sum((evaluated_values - I_f)**2)
    rmse = np.sqrt(var / n_samples)

    integral, rmse = I_f, rmse
    # ====================================================================
    return integral, rmse, error


if __name__ == "__main__":

    # TODO: define the parameters of the simulation.
    # ====================================================================
    sample_size = [10, 100, 1000, 10000]
    rmse_per_s = []
    error_per_s = []
    # ====================================================================

    print(f"Objective of approximation: {-math.cos(1) + math.cos(0)}")

    # TODO: compute the integral and the errors.
    # ====================================================================
    for i, s in enumerate(sample_size):
        integral, rmse, error = integrate_mc(f, 0, 1, s, with_transform=False)
        rmse_per_s.append(rmse)
        error_per_s.append(error)
        print(f"For sample size {s}, the approximation is {integral}")
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    fig, ax = plt.subplots()
    
    ax.plot(sample_size, rmse_per_s)  # Plot the chart
    ax.plot(sample_size, error_per_s)  # Plot the chart
    ax.set_xscale('log')
    ax.set_title("Errors for estimator Integral of Sin 0-1")
    ax.set_ylabel("Value")
    ax.set_xlabel("Number of Samples")
    ax.legend(["RMSE", "Absolute error"])

    fig.tight_layout()
    fig.savefig('bonus_exercise_1/outputs/assignment_2_01.png', bbox_inches='tight')  # save_image 
    # ====================================================================

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
    return (-math.cos(b) + math.cos(a))
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
        samples = (samples * (b - a)) + a

    # Estimating Integral
    evaluated_values = np.array([f(xi) for xi in samples])

    I_f = (1/n_samples) * np.sum(evaluated_values) * (b-a)

    ### TODO: Change formula above with the correct density function ##########
    # Multiply or divide by 2 ##

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
    rmse_per_s = np.zeros((len(sample_size), 2))
    error_per_s = np.zeros((len(sample_size), 2))
    # ====================================================================

    print(f"Objective of approximation: {-math.cos(4) + math.cos(2)}")

    # TODO: compute the integral and the errors.
    # ====================================================================
    print("Without Transformation")
    for i, s in enumerate(sample_size):
        integral, rmse, error = integrate_mc(f, a=2, b=4, n_samples=s, with_transform=False, seed=24+i)  
        rmse_per_s[i,0] = rmse
        error_per_s[i,0] = error
        print(f"For sample size {s}, the approximation is {integral}")
    
    print("With Transformation")
    for i, s in enumerate(sample_size):
        integral_t, rmse_t, error_t = integrate_mc(f, a=2, b=4, n_samples=s, with_transform=True, seed=24+i)
        rmse_per_s[i,1] = rmse_t
        error_per_s[i,1] = error_t
        print(f"For sample size {s}, the approximation is {integral_t}")

    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(sample_size, rmse_per_s)  # Plot the chart
    ax1.set_xscale('log')
    ax1.set_title("RMSE for estimator Integral of Sin 2-4")
    ax1.set_ylabel("RMSE")
    ax1.legend(["No transformation", "With transformation"])
    
    ax2.plot(sample_size, error_per_s)  # Plot the chart
    ax2.set_title("Errors for estimator Integral of Sin 2-4")
    ax2.set_ylabel("Error")
    ax2.set_xlabel("Number of Samples")
    ax2.legend(["No transformation", "With transformation"])

    fig.tight_layout()
    fig.savefig('bonus_exercise_1/outputs/assignment_2_24.png', bbox_inches='tight')  # save_image 
    # ====================================================================

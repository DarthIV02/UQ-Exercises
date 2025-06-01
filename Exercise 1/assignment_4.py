from collections import defaultdict

import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from utils.oscillator import Oscillator

def plot_solutions(
    ax: plt.Axes,
    t_grid: npt.NDArray[np.float64],
    solutions: dict[str, npt.NDArray[np.float64]],
):
    for i, possible_solution in enumerate(solutions):
        ax.plot(t_grid, possible_solution, linestyle="--", label=f"{i}")
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.legend()

def load_reference(filename: str) -> tuple[float, float]:
    # TODO: load reference values for the mean and variance.
    # ====================================================================

    with open(filename) as f:
        lines = f.readlines()

    mean, var = float(lines[0]), float(lines[1])
    # ====================================================================
    return mean, var


def simulate(
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=42,
) -> npt.NDArray:
    # TODO: simulate the oscillator with the given parameters and return
    # generated solutions.
    # ====================================================================

    omegas = omega_distr.sample(size=n_samples, seed=seed)
    solutions = []

    for w in omegas:
        oscillator = Oscillator(c=model_kwargs["c"], k=model_kwargs["k"], f=model_kwargs["f"], omega=w)
        solutions.append(oscillator.discretize("odeint", init_cond["y0"], init_cond["y1"], t_grid))

    sample_solutions = np.array(solutions)
    # ====================================================================
    return sample_solutions


def compute_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    # TODO: compute the relative errors of the mean and variance
    # estimates.
    # ====================================================================

    mean_error = np.abs(1 - (samples[0] / mean_ref))
    var_error = np.abs(1 - (samples[1] / var_ref))

    # ====================================================================
    return mean_error, var_error


if __name__ == "__main__":
    # ====================================================================
    # TODO: define the parameters of the simulations.
    # ====================================================================
    
    sample_size = [10, 100, 1000, 10000]
    model_kwargs = {"c": 0.5, "k": 2.0, "f": 0.5}    
    init_cond = {"y0": 0.5, "y1": 0.0}
    t_grid = np.arange(0, 10 + 0.1, 0.1)

    # Distributions of omega
    uniform_omega = cp.Uniform(0.95, 1.05)
    halton_omega = cp.J(cp.Uniform(0.95, 1.05))
    omega = {"uniform": uniform_omega, "halton": halton_omega}

    # ====================================================================
    # TODO: run the simulations.
    # ====================================================================

    solutions = {}

    for name, d in omega.items():
        solutions[name] = {}
        for n in tqdm(sample_size):
            sample_solutions = simulate(t_grid, d, n, model_kwargs, init_cond)
            solutions[name][n] = sample_solutions

    # ====================================================================

    # TODO: compute the statistics.
    # ====================================================================
    
    # Change path respectively
    ref_mean, ref_var = load_reference("./bonus_exercise_1/template/data/oscillator_ref.txt")
    print("\nReference Values - mean: ", ref_mean, " var: ", ref_var)
    relative_errors = np.zeros((2, len(sample_size), 2))

    for j, name in enumerate(omega.keys()):
        print(f"\nFor {name} distribution in omega:")
        for i, n in enumerate(sample_size):
            mean_y_10 = np.mean(solutions[name][n][:, -1])
            var_y_10 = np.var(solutions[name][n][:, -1], ddof=1)
            samples = np.array([mean_y_10, var_y_10], dtype=float)

            relative_errors[j, i] = compute_errors(samples, ref_mean, ref_var)
            print(f"For sample size {n} - mean: ", mean_y_10, " var: ", var_y_10)

    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(11, 4))
    
    axs[0, 0].plot(sample_size, relative_errors[0, :, 0])  # Plot the chart
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title("Relative error for mean value at y(10) with normal sequence")
    axs[0, 0].set_ylabel("Relative error")
    
    axs[0, 1].plot(sample_size, relative_errors[0, :, 1])  # Plot the chart
    axs[0, 1].set_title("Relative error for variance at y(10) with normal sequence")
    axs[0, 1].set_ylabel("Relative error")
    axs[0, 1].set_xlabel("Number of Samples")

    axs[1, 0].plot(sample_size, relative_errors[1, :, 0])  # Plot the chart
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_title("Relative error for mean value at y(10) with halton sequence")
    axs[1, 0].set_ylabel("Relative error")
    
    axs[1, 1].plot(sample_size, relative_errors[1, :, 1])  # Plot the chart
    axs[1, 1].set_title("Relative error for variance at y(10) with halton sequence")
    axs[1, 1].set_ylabel("Relative error")
    axs[1, 1].set_xlabel("Number of Samples")

    fig.tight_layout()
    fig.savefig('bonus_exercise_1/outputs/assignment_4_error.png', bbox_inches='tight')  # save_image 

    # ====================================================================

    # TODO: plot sampled trajectories.
    # ====================================================================
    fig, axes = plt.subplots(figsize=(6, 4))
    plot_solutions(axes, t_grid, solutions['uniform'][10])
    axes.set_title(f"Trajectories")
    fig.tight_layout()
    fig.savefig("bonus_exercise_1/outputs/assignment_4_traj.png")
    # ====================================================================

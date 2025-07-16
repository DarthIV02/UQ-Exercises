import time
from functools import partial
from typing import Callable

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.oscillator import Oscillator
from utils.wiener import WienerProcess
from tqdm import tqdm


def generate_f_samples(
    mu: float,
    t_grid: npt.NDArray,
    n_samples: int,
    M: int | None,
    rng: np.random.Generator,
    approx: bool,
) -> list[Callable[[float], float]]:

    # TODO: generate realizations of the Wiener process for f(t).
    # If M is None, we generate samples using the standard definition.
    # If M is not None, we generate samples using the KL expansion with M terms.
    # The samples are returned as a list of callable functions that
    # evaluate the Wiener process at a given time point.

    """Generates samples of the Wiener process as callable functions at discrete t_grid points."""

    wiener_process = WienerProcess(mu, t_grid=t_grid) # Initialize the class
    
    # Call different process for general Weiner process and the KL approximation
    if approx:
        kl_approx = wiener_process.approximate_kl(n_samples, M, rng)  # Shape: (n_samples, len(t_grid))
    else:
        kl_approx = wiener_process.generate(n_samples, rng)

    # Map from time value to its index in t_grid
    t_index_map = {round(t, 2): i for i, t in enumerate(t_grid)}

    # Create one function per sample
    functions = []
    for i in range(n_samples): # For each sample
        sample_values = kl_approx[i] # Return that particular ample f_t

        def make_func(values: np.ndarray) -> Callable[[float], float]:
            return lambda t: float(values[t_index_map[round(t, 2)]]) if round(t, 2) in t_index_map else None

        # Return each approximation for each sample
        functions.append(make_func(sample_values))

    return functions
    

def simulate(
    t_grid: npt.NDArray,
    f_samples: list[Callable[[float], float]],
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
) -> npt.NDArray:
    """Simulates the oscillator model for each sample of f(t)."""

    # TODO: simulate the oscillator model for each sample of f(t) and
    # return the trajectories as 2D array.

    oscillator_eval = np.zeros((len(f_samples), len(t_grid)))

    # Evaluate the oscillator at each step
    for sample in tqdm(range(len(f_samples))):
        oscillator = Oscillator(c=model_kwargs["c"], k=model_kwargs["k"], f=f_samples[sample], omega=model_kwargs["omega"])
        oscillator_eval[sample] = oscillator.discretize("euler", *init_cond.values(), t_grid)

    return oscillator_eval


def compute_metrics(solutions: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Computes the mean and standard deviation of the solutions."""

    # Solutions has shape np.zeros(len(f_samples), len(t_grid)) 

    # TODO: compute the metrics of the output

    mean = np.mean(solutions, axis=0)
    var = np.var(solutions, axis=0, ddof=1)

    return mean, var


def plot_solutions(
    t_grid: npt.NDArray, sampler_solutions: dict[str, npt.NDArray]
) -> plt.Figure:
    """Plots the oscillator trajectories for each sample of f."""
    n_plots = len(sampler_solutions)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 4), sharex=True, sharey=True
    )
    for ax, (name, solutions) in zip(axes, sampler_solutions.items()):
        mean, std = compute_metrics(solutions)
        ax.plot(t_grid, solutions.T, alpha=0.01, c="b")
        ax.plot(t_grid, mean, c="r", label="mean")
        ax.fill_between(
            t_grid, mean - std, mean + std, color="red", alpha=0.5, label="std"
        )

        # Add legend for samples manually.
        handles, _ = ax.get_legend_handles_labels()
        line = lines.Line2D([0], [0], color="b", label="Monte Carlo samples")
        handles.append(line)
        ax.legend(handles=handles)

        ax.set_title(name)
    return fig


if __name__ == "__main__":
    # TODO: set parameters of the model.
    f_mean = 0.5
    model_kwargs = {"c": 0.5, "k": 2.0, "omega": 1.0}
    init_cond = {"y0": 0.5, "y1": 0}

    # TODO: set the time domain.
    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    # TODO: set the number of Monte-Carlo samples and KL terms.
    N = 1000
    Ms = [5, 10, 100]
    seed = 42
    rng = np.random.default_rng(seed)

    ###########################################################################

    # TODO: generate samples of the Wiener process for f using the stadard
    # generation and the KL expansion for different M.
    
    y_10_mean = np.zeros((1+len(Ms)))
    y_10_var = np.zeros((1+len(Ms)))
    oscillator_sim = {"Weiner":None, "KL_expan_5":None, "KL_expan_10":None, "KL_expan_100":None}
    
    # For general Weiner process
    f_t = generate_f_samples(f_mean, t_grid, N, 1000, rng, approx=False)
    oscillator_sim["Weiner"] = simulate(t_grid, f_t, model_kwargs, init_cond) # Save solution
    metrics_mean, metrics_var = compute_metrics(oscillator_sim["Weiner"])
    y_10_mean[0], y_10_var[0] = metrics_mean[-1], metrics_var[-1] # Compute statistics for last point
    print("Weiner")
    print(y_10_mean)

    for m, m_i in enumerate(Ms): # For each possible M
        f_t = generate_f_samples(f_mean, t_grid, N, m_i, rng, approx=True) # Generate f_t
        oscillator_sim[f"KL_expan_{m_i}"] = simulate(t_grid, f_t, model_kwargs, init_cond) # Simulate the oscillator
        metrics_mean, metrics_var = compute_metrics(oscillator_sim[f"KL_expan_{m_i}"])

        # TODO: simulate the oscillator model for each sample of f and record the
        # mean and standard deviation of the solutions at T_max.
        
        y_10_mean[m+1], y_10_var[m+1] = metrics_mean[-1], metrics_var[-1]

        print(f"KL_expan_{m_i}")
        print(y_10_mean[m+1])


    # TODO: optionally, plot the solutions for each sample of f.

    print("Means:")
    print(y_10_mean)
    print("Vars:")
    print(y_10_var)

    plot = plot_solutions(t_grid, oscillator_sim)
    plot.savefig("outputs/bonus_3_task_32.png")
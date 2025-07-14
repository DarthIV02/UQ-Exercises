import time

import chaospy as cp
import numpy as np

from utils.sobol import monte_carlo_sobol, pseudo_spectral_sobol
from utils.oscillator import Oscillator

import matplotlib.pyplot as plt

def get_distribution(
    c_lims: tuple[float, float],
    k_lims: tuple[float, float],
    f_lims: tuple[float, float],
    y0_lims: tuple[float, float],
    y1_lims: tuple[float, float],
) -> cp.Distribution:
    """Creates the joint distribution over the stochastic parameters."""

    # TODO: create a joint distribution over the stochastic parameters.

    c_dist = cp.Uniform(*c_lims)
    k_dist = cp.Uniform(*k_lims)
    f_dist = cp.Uniform(*f_lims)
    y0_dist = cp.Uniform(*y0_lims)
    y1_dist = cp.Uniform(*y1_lims)

    distr_5D = cp.J(c_dist, k_dist, f_dist, y0_dist, y1_dist)

    return distr_5D


def run_method(method, **kwargs):
    """Runs the specified method and prints the results.

    The results include the first and total order Sobol' indices as well as
    the elapsed time to run the method."""

    # TODO: run the method and print the results.

    if method == "monte_carlo":

        start = time.time()
        first_order, total_order = monte_carlo_sobol(kwargs["n_samples"], kwargs["distribution"], kwargs["t_grid"], kwargs["fixed_args"])
        elapsed_time = time.time() - start
        method = f"Monte Carlo with {kwargs['n_samples']} samples"

    elif method == "pseudo_spectral":

        start = time.time()
        first_order, total_order = pseudo_spectral_sobol(kwargs["pce_degree"], kwargs["quadrature_degree"], kwargs["distribution"], kwargs["t_grid"],
                              kwargs["fixed_args"], kwargs["sparse"])
        elapsed_time = time.time() - start
        method = f'Pseudo Spectral with pce degree {kwargs["pce_degree"]}, quad degree {kwargs["quadrature_degree"]}'
        if kwargs["sparse"]:
            method = method + " Sparse"

    else:

        raise Exception("Not a valid method")
    
    print(method)
    print("First Order:")
    print(first_order)
    print("Total Order:")
    print(total_order)
    
    parameters = ['c', 'k', 'f', 'y_0', 'y_1']
    
    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Define a color map for consistency
    colors = plt.get_cmap('tab10')
    color_dict = {param: colors(i) for i, param in enumerate(parameters)}

    # First-order Sobol indices plot
    ax1.bar(parameters, first_order, color=[color_dict[p] for p in parameters])
    ax1.set_yscale('log')
    ax1.set_ylabel('Sobol Index (log scale)')
    ax1.set_title('First-order Sobol Indices')
    ax1.set_xlabel('Parameter')

    # Total-order Sobol indices plot
    ax2.bar(parameters, total_order, color=[color_dict[p] for p in parameters])
    ax2.set_yscale('log')
    ax2.set_title('Total-order Sobol Indices')
    ax2.set_xlabel('Parameter')

    # Super title and save
    fig.suptitle(f'{method} Sobol Indices (Time: {elapsed_time} s)', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'outputs/bonus_3_task_1_{method}.png')
    plt.close()

if __name__ == "__main__":
    # TODO: set the stochastic parameters.
    c_lims = [0.08, 0.12]
    k_lims = [0.03, 0.04]
    f_lims = [0.08, 0.12]
    y0_lims = [0.45, 0.55]
    y1_lims = [-0.05, 0.05]

    # TODO: set the determinisic parameters.
    fixed_args = {"omega": 1.0}

    # TODO: set the parameters of the methods.
    quadrature_degree = [3,4]
    pce_degree = [3,4]
    n_samples = 1024

    # TODO: set the time domain
    T_max = 10
    dt = 0.01
    t_grid = np.arange(0, T_max + dt, dt)

    ###########################################################################

    # TODO: define the distribution over the stochastic parameters.

    distr_5D = get_distribution(c_lims, k_lims, f_lims, y0_lims, y1_lims)

    # For each defined quadrature degree -> assume pce degree is the same
    for q, quad_deg_1D in enumerate(quadrature_degree): 
        # Compute both sparse and not sparse
        for sparse in [True, False]:
            run_method("pseudo_spectral", pce_degree = pce_degree[q], quadrature_degree = quad_deg_1D,
                    distribution = distr_5D, t_grid = t_grid, fixed_args = fixed_args, sparse=sparse)
    
    run_method("monte_carlo", distribution = distr_5D, t_grid = t_grid, fixed_args = fixed_args,
               n_samples = n_samples)

    ###########################################################################

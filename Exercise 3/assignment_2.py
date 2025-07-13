from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from itertools import permutations
from scipy.spatial.distance import cdist

def exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: float) -> npt.NDArray:
    """Computes the exponential covariance function between a sets of points."""
    
    # TODO: compute the exponential covariance function.

    #The eucledian distance for each set of points    
    dists = cdist(x, y, 'euclidean')  # Shape: (81, 81)

    #Covariance calculation   
    cov = np.exp(-dists/scale)

    return cov


def squared_exp_cov_fn(x: npt.NDArray, y: npt.NDArray, scale: npt.NDArray):
    """Computes the squared exponential covariance function between a sets of points."""

    # TODO: compute the squared exponential covariance function.

    #The squared distance for each set of points
    dists_sq = cdist(x, y, 'sqeuclidean') 

    # Covariance calculation
    cov = np.exp(-dists_sq / (2 * scale**2))    

    return cov

def get_xy_mesh(
    x_lims: tuple[float, float],
    y_lims: tuple[float, float],
    x_mesh_size: int,
    y_mesh_size: int,
) -> npt.NDArray:
    """Creates a 2D mesh grid for the given limits and mesh sizes."""
    x_step = (x_lims[1] - x_lims[0]) / x_mesh_size
    y_step = (y_lims[1] - y_lims[0]) / y_mesh_size
    x_grid = np.arange(x_lims[0] + x_step / 2, x_lims[1], x_step)
    y_grid = np.arange(y_lims[0] + y_step / 2, y_lims[1], y_step)
    mesh = np.stack(np.meshgrid(x_grid, y_grid), axis=-1)
    return mesh


def sample(mesh, mean_fn, cov_fn, n_samples, rng, reg_scale=1e-7):
    """Samples from a Gaussian process defined by the mean and covariance functions."""

    # TODO: sample a Gaussian field suing the Cholesky decomposition.

    # Add small jitter for numerical stability
    # Otherwise the small eigenvalues will lead to a function that's "not PSD"
    cov_fn += reg_scale * np.eye(cov_fn.shape[0]) 

    # Cholesky Decomposition
    L = np.linalg.cholesky(cov_fn)

    x_mesh_size, y_mesh_size = mesh.shape[:-1]
    
    # Create random values -> shape (3,81) with mean 0 and variance 1
    psi = rng.multivariate_normal(np.zeros((y_mesh_size**2,)), np.eye(x_mesh_size**2, y_mesh_size**2), (n_samples))

    G = np.zeros((n_samples, *mesh.shape[:-1])) # Store the values

    for i in range(n_samples): # For each sample
        G[i] = mean_fn + np.reshape(np.dot(L, psi[i]), (x_mesh_size, y_mesh_size))

    return G


def plot_samples(samples, x_lims, y_lims, titles=None, main_title=None, cbar_label=None):
    """Plots the samples from the Gaussian process."""
    n_plots = len(samples)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    # Ensure axes is iterable
    if n_plots == 1:
        axes = [axes]

    #Find global vmin and vmax across all samples so that the values are normalized
    all_data = np.array(samples)
    vmin = all_data.min()
    vmax = all_data.max()

    for i, (ax, sample) in enumerate(zip(axes, samples)):
        
        im = ax.imshow(sample, cmap="coolwarm", origin="lower", extent=(*x_lims, *y_lims),
                       vmin=vmin, vmax=vmax) # Set values to the same scale across samples

        # Optional individual subplot title
        if titles and i < len(titles):
            ax.set_title(titles[i])

        # Optional colorbar (legend for color intensity)
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)

    # Optional overall figure title
    if main_title:
        fig.suptitle(main_title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    return fig


if __name__ == "__main__":
    # TODO: set the condiguration.
    x_lims, y_lims = [0,1], [0,1]
    x_mesh_size, y_mesh_size = 50,50
    scale = 1
    mean = 0.1
    seed = 42
    n_samples = 3
    cov_funcion_name = "exp_cov_fn"
    rng = np.random.default_rng(seed)

    # TODO: create a 2D mesh.

    # Creates a grid of size (9,9,2), where each [0,0,:] -> point in space
    mesh = get_xy_mesh(x_lims, y_lims, x_mesh_size, y_mesh_size)

    ### Evaluate the covariance function
    cov_function = {"exp_cov_fn": exp_cov_fn, "squared_exp_cov_fn": squared_exp_cov_fn}


    points = np.reshape(mesh, (-1, 2)) # Flatten to (81,2)

    # Previous one -> inneficient
    #for i in range(x_mesh_size ** 2):
    #    for j in range(y_mesh_size ** 2):
    #        covariance[i, j] = cov_function[cov_funcion_name](points[i], points[j], scale)

    # Parallelize
    # Calculate the covariance function for the vectorized set of points
    covariance = cov_function[cov_funcion_name](points, points, scale)

    # TODO: sample from the Gaussian process with different kernels.
    
    G = sample(mesh, mean, covariance, n_samples, rng)

    # TODO: plot the samples. 
    # 3 plots -> 1 per sample generated
    
    titles = [f"Sample {i}" for i in range(n_samples)]
    main_title = {"exp_cov_fn": "Exponential Covariance Function", "squared_exp_cov_fn": "Squared Exponential Covariance Function"}
    fig = plot_samples(G, x_lims, y_lims, titles=titles, main_title=f"Samples for {main_title[cov_funcion_name]}")
    
    # Save fig
    fig.savefig(f'outputs/bonus_3_task_2_{cov_funcion_name}.png')

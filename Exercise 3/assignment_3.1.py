import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.wiener import WienerProcess

def plot_eigenpairs(
    wiener: WienerProcess, n_terms: int, t_grid: npt.NDArray[np.float64]
) -> plt.Figure:
    """Plots the first n_terms eigenvalues and eigenfunctions of the Wiener process."""
    eigenvalues, eigenfunctions = wiener.kl_eigenpairs(n_terms)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, n_terms + 1), eigenvalues, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_title(f"First {n_terms} eigenvalues")
    axes[1].plot(t_grid, eigenfunctions(t_grid))
    axes[1].set_title(f"First {n_terms} eigenfunctions")
    return fig


if __name__ == "__main__":
    # TODO: set the configuration.
    T = 1 # Because we are looking at time 0->1
    n_points = 1000
    t_grid = np.linspace(0, T, n_points)
    Ms = [10, 100, 1000]
    seed = 100
    n_samples = 1 # 1 Sample of the Weiner process
    rng = np.random.default_rng(seed)

    weiner_process = WienerProcess(0, T, n_points) # Set the weiner process class
    # Set zeta to use the same random values for different Ms
    weiner_process.zeta = rng.random(size=(n_points)) 

    # TODO: generate one realization of the Wiener process using the
    kl_result = weiner_process.generate(n_samples, rng)

    # TODO: visualize first eigenvalues and eigenfunctions.
    fig = plot_eigenpairs(weiner_process, 1000, t_grid)
    fig.savefig(f'outputs/bonus_3_task_31.png')

    # TODO: generate approximations of the Wiener process using the KL expansion.
    
    kl_approx = [] # Save the approximations

    for i, m_i in enumerate(Ms): # Save the approximation for each M
        approx = weiner_process.approximate_kl(n_samples, m_i, rng)
        kl_approx.append(approx) # Returns shape (n_samples, n_points)

    print("Show the first 10 values")
    print("Weiner:")
    print(kl_result[0,:10])
    print("Approximation M=10:")
    print(kl_approx[0][0, :10])
    print("Approximation M=100:")
    print(kl_approx[1][0, :10])
    print("Approximation M=1000:")
    print(kl_approx[2][0, :10])

    # TODO: plot the approximation results.
    plt.figure(figsize=(12, 6))
    plt.plot(t_grid, kl_approx[0][0], label='Approximation M=10')
    plt.plot(t_grid, kl_approx[1][0], label='Approximation M=100')
    plt.plot(t_grid, kl_approx[2][0], label='Approximation M=1000')

    # Add title and labels
    plt.title('Karhunen-Loeve Expansion')
    plt.xlabel('T_grid')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    plt.savefig('outputs/bonus_3_task_31_approx.png')

    # Visualize the output of the Weiner process
    plt.figure(figsize=(12, 6))
    plt.plot(t_grid, kl_result[0], label='True')

    # Add title and labels
    plt.title('Weiner Process')
    plt.xlabel('T_grid')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    plt.savefig('outputs/bonus_3_task_31_weiner.png')
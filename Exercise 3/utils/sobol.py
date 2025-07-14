import chaospy as cp
import numpy as np
import numpy.typing as npt

from .oscillator import Oscillator


def _evaluate_oscillator(
    samples: npt.NDArray, t_grid: npt.NDArray, fixed_args: dict[str, float]
) -> npt.NDArray:
    """Evaluates the oscillator model for given samples."""

    # TODO: evaluate the oscillator model for each sample.

    oscillator_result = np.zeros((samples.shape[0]))

    for i, node in enumerate(samples):
        params = node[:3] # Seperate parameters
        pos_vel = node[3:5] # Position and velocity

        oscillator_function = Oscillator(*params, omega=fixed_args["omega"])
        # Solve using odeint and store the last value
        oscillator_result[i] = oscillator_function.discretize("odeint", *pos_vel, t_grid=t_grid)[-1] 

    return oscillator_result


def monte_carlo_sobol(
    n_samples: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
) -> tuple[float, float]:
    """Computes the Sobol' indices using Monte Carlo sampling."""
    
    # TODO: implement the algorithm from the paper.

    estimate_Si = np.zeros((5, n_samples)) # 5 * 1024 -> # parameters * # n_samples
    estimate_Stotal = np.zeros((5, n_samples)) # 5 * 1024 -> # parameters * # n_samples

    A = distribution.sample(n_samples).T # 1024,5
    B = distribution.sample(n_samples).T # 1024,5

    eval_A_B = np.zeros((5, n_samples))

    # Evaluating the oscillator for each point in the matrix (A and B)
    # q = # of deterministic parameters + 1

    # For each 1024 -> evaluate the function
    eval_A = _evaluate_oscillator(A, t_grid, fixed_args)
    eval_B = _evaluate_oscillator(B, t_grid, fixed_args)

    for i in range(A.shape[1]): # Calculating S_i
        AB_i = A.copy()

        # Following the radial sampling method proposed in the paper
        AB_i[:, i] = B[:, i] # Change the ith row of A

        # For each 1024 -> evaluate the function
        eval_A_B[i] = _evaluate_oscillator(AB_i, t_grid, fixed_args)

        # Follow Table2 equation b and f to calculate the estimates
        estimate_Si[i] = eval_B * (eval_A_B[i] - eval_A)
        estimate_Stotal[i] = (eval_A - eval_A_B[i])**2

    # Calculate the variance based on the the evaluated samples of both A and B
    var = np.var(np.vstack([eval_A, eval_B])) 

    S_i = np.mean(estimate_Si, axis=1) / var
    S_total = 0.5 * np.mean(estimate_Stotal, axis=1) / var

    return S_i, S_total


def pseudo_spectral_sobol(
    pce_degree: int,
    quadrature_degree: int,
    distribution: cp.Distribution,
    t_grid: npt.NDArray[np.float64],
    fixed_args: dict[str, float],
    sparse=True,
) -> tuple[float, float]:
    """Computes the Sobol' indices using a pseudo-spectral method."""
    
    # TODO: implement the pseduo-spectral method.

    # Generate polynomials
    poly = cp.generate_expansion(pce_degree, distribution)

    # Generate nodes and weights of the quadrature
    nodes, weights = cp.generate_quadrature(quadrature_degree, distribution, rule='G', sparse=sparse)

    # Evaluate for the set of nodes
    evaluated_oscilator = _evaluate_oscillator(nodes.T, t_grid, fixed_args)

    # Fit quadrature
    model_approx = cp.fit_quadrature(poly, nodes=nodes, weights=weights, solves=evaluated_oscilator)
    
    # Get sensitivity analysis based on chaospy
    first_order = cp.Sens_m(model_approx, distribution)
    total_order = cp.Sens_t(model_approx, distribution)

    return first_order, total_order

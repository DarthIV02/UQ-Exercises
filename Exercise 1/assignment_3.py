import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np

from utils.sampling import control_variates, importance_sampling, monte_carlo


def f(x: float) -> float:
    # TODO: define the target function.
    # ====================================================================

    result = np.exp(x)

    return result
    # ====================================================================


def analytical_integral() -> float:
    # TODO: compute the analytical integral of f on [0, 1].
    # ====================================================================
    return np.exp(1) - np.exp(0)
    # ====================================================================


def run_monte_carlo(Ns: list[int], seed: int = 42):
    # TODO: run the Monte Carlo method and return the absolute error
    # of the estimation.
    # ====================================================================
    distr = cp.Uniform(0, 1)
    I_f = np.zeros((len(Ns)))
    target_I_f = np.zeros((len(Ns)))
    target_I_f[:] = analytical_integral()

    for i, n in enumerate(Ns):
        samples = distr.sample(size=n, seed=seed)
        evaluated_values = np.array([f(xi) for xi in samples])
        I_f[i] = (1/n) * np.sum(evaluated_values)

    error = np.abs(I_f - target_I_f)
    
    return error

    # ====================================================================


def run_control_variates(
    Ns: list[int], seed: int = 42
):
    # TODO: run the control variate method for and return the absolute
    # errors of the resulting estimations.
    # ====================================================================
    
    distr = cp.Uniform(0, 1)
    I_phi = np.zeros((3, len(Ns)))
    target_I_f = np.zeros((3, len(Ns)))
    target_I_f[:] = analytical_integral()

    phi_1 = lambda a : a
    phi_2 = lambda a : 1 + a
    phi_3 = lambda a : 1 + a + (a**2/2)

    phi = [phi_1, phi_2, phi_3]
    E_phi = [1/2, 3/2, 5/3]
    rho_phi = [np.sqrt(1/12), np.sqrt(1/12), np.sqrt((89/30) - (5/3)**2)]

    for i, n in enumerate(Ns):
        I_f = np.zeros((len(Ns)))
        samples = distr.sample(size=n, seed=seed)
        evaluated_values_f = np.array([f(xi) for xi in samples])
        I_f[i] = (1/n) * np.sum(evaluated_values_f)

        rho_f = np.sqrt(max(0,np.sum(evaluated_values_f - I_f[i]))/(n-1))

        for j, g in enumerate(phi):
            evaluated_values_g = np.array([g(xi) for xi in samples])

            cov = np.sum((evaluated_values_f - I_f[i]) * (evaluated_values_g - E_phi[j]))
            rhos = np.sqrt(np.sum((evaluated_values_f - I_f[i])**2)*np.sum((evaluated_values_g - E_phi[j])**2))
            pearson_coef = cov/rhos

            alpha = pearson_coef * rho_f / rho_phi[j]

            I_phi[j, i] = I_f[i] + alpha*(E_phi[j] - ((1/n) * np.sum(evaluated_values_g)))

            variance_I_phi = (1 - pearson_coef)**2 * (np.var(evaluated_values_f, ddof=1) / n)
            print("Variance phi: ", variance_I_phi)

    error = np.abs(I_phi - target_I_f)
    
    return error

    # ====================================================================


def run_importance_sampling(
    Ns: list[int], seed: int = 42
):
    # TODO: run the importance sampling method and return the absolute
    # errors of the resulting estimations.
    # ====================================================================

    distr_1 = cp.Beta(5, 1)
    distr_2 = cp.Beta(0.5, 0.5)

    distr = [distr_1, distr_2]

    I_f = np.zeros((2, len(Ns)))
    target_I_f = np.zeros((2, len(Ns)))
    target_I_f[:] = analytical_integral()

    for i, d in enumerate(distr):
        for j, n in enumerate(Ns):
            samples = d.sample(size=n, seed=seed)
            evaluated_values = np.array([f(xi)/d.pdf(xi) for xi in samples])
            I_f[i, j] = (1/n) * np.sum(evaluated_values)

    error = np.abs(I_f - target_I_f)
    
    return error

    # ====================================================================


if __name__ == "__main__":
    # TODO: define the parameters of the simulation.
    # ====================================================================

    #value_to_for = f(10)
    sample_size = [10, 100, 1000, 10000]
    error_per_s = np.zeros((len(sample_size), 2))
    seed = 42

    # ====================================================================

    # TODO: run all the methods.
    # ====================================================================

    # Monte Carlo Sampling
      
    errors_mc = run_monte_carlo(Ns=sample_size, seed=seed)

    # Control Variates

    error_cv = run_control_variates(Ns=sample_size, seed=seed+1)

    # Importance Sampling

    error_is = run_importance_sampling(Ns=sample_size, seed=seed+2)
    print(error_is)
        
    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    fig, ax = plt.subplots()
    
    ax.plot(sample_size, errors_mc)  # Plot the chart
    for i in range(3):
        ax.plot(sample_size, error_cv[i])  # Plot the chart
    for i in range(2):
        ax.plot(sample_size, error_is[i])  # Plot the chart
    ax.set_xscale('log')
    ax.set_title("Absolute errors in estimating Integral of Exp 0-1")
    ax.set_ylabel("Value")
    ax.set_xlabel("Number of Samples")
    ax.legend(["Error_mc", "Error_cv phi_1", "Error_cv phi_2", "Error_cv phi_3", "Error_is a=5 & b=1", "Error_is a=0.5 & b=0.5"])

    fig.tight_layout()
    fig.savefig('bonus_exercise_1/outputs/assignment_3.png', bbox_inches='tight')  # save_image 
    # ====================================================================

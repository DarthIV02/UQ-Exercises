import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time
import matplotlib.pyplot as plt

from typing import Union, Optional
import numpy.typing as npt
# if you want you can rely also on already implemented Oscillator class
# from utils.oscillator import Oscillator


# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(init_cond: tuple[float, float], t: Union[float, npt.NDArray], args: tuple[float, float, float, float]) -> list[float]:
    x1, x2 = init_cond
    c, k, f, w = args
    f = [x2, f * np.cos(w * t) - k * x1 - c * x2]
    return f


# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol: float, rtol: float, init_cond: tuple[float, float], args: tuple[float, float, float, float], 
t: npt.NDArray, t_interest: int) -> float:
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
    return sol[t_interest, 0]


if __name__ == '__main__':
    ### deterministic setup ###

    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.0
    # initial conditions setup
    init_cond   = y0, y1
    # model_kwargs = {"c": c, "k": k, "f": f}  # if you want to use the Oscillator class, you can uncomment this line
    # init_cond = {"y0": y0, "y1": y1}  # if you want to use the Oscillator class, you can uncomment this line

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t_grid          = np.array([i*dt for i in range(grid_size)])
    #t_grid = np.arange(0, t_max + dt, dt)
    t_interest  = -1

    ### stochastic setup ####
    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05
    # TODO: create uniform distribution object
    distr_w = cp.Uniform(w_left, w_right)

    # the truncation order of the polynomial chaos expansion approximation
    N = [1, 2, 3, 4, 5, 6]
    # the quadrature degree of the scheme used to computed the expansion coefficients
    K = [1, 2, 3, 4, 5, 6]

    assert(len(N)==len(K))
    
    # vector to save the statistics
    exp_m = np.zeros(len(N))
    var_m = np.zeros(len(N))

    exp_cp = np.zeros(len(N))
    var_cp = np.zeros(len(N))

    # compute relative error
    stat_ref    = [-0.43893703, 0.00019678]
    relative_error = lambda approx, ref: np.abs(1. - approx/ref)

    error_K_mean = np.zeros((len(N)))
    error_K_var = np.zeros((len(N)))

    # perform polynomial chaos approximation + the pseudo-spectral
    for h in range(len(N)):

        # TODO: create N[h] orthogonal polynomials using chaospy
        poly = cp.generate_expansion(h, distr_w, normed=True)

        # TODO: create K[h] quadrature nodes using chaospy
        nodes, weights = cp.generate_quadrature(h, distr_w, rule='G')
        nodes = nodes[0]

        ########################################################
        # Evauate the function at the nodes                    #
        ########################################################
        M_eval = np.zeros_like(weights)
        for i in range(len(weights)):
            M_eval[i] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args=(c,k,f,nodes[i]), t=t_grid, t_interest=t_interest)
        
        # TODO: perform polynomial chaos approximation + the pseudo-spectral approach manually
        
        # Evaluate each polinomial at each node -> results in a matrix of n*k
        evaluated_poly = np.array([[p(val) for val in nodes] for p in poly])

        # M_eval * weights -> Is the f(t,x_i)w_k
        # The dot product is going to multiply f(t,x_i)w_k * the respecitive evalution of all the nodes for a single plynomial
        # This will automize the sum from 0 -> K-1
        f_hat = np.dot(evaluated_poly, M_eval * weights)
        
        # Append a 0 just for code purposes -> so that f_hat[1:] doesn't crash but it won't affect the values
        f_hat = np.append(f_hat, 0)
        
        exp_m[h] = f_hat[0]
        var_m[h] = np.sum(np.power(f_hat[1:], 2))

        error_K_mean[h] = relative_error(exp_m[h], stat_ref[0])
        error_K_var[h] = relative_error(var_m[h], stat_ref[1])

        # TODO: perform polynomial chaos approximation + the pseudo-spectral approach using chaospy
        
        ########################################################
        # Fit to create the gPC                                #
        ########################################################
        
        gPCM, gPCcoeff = cp.fit_quadrature(poly, nodes, weights, M_eval, retall=True)
        
        ########################################################
        # Compute quantities of interest                       #
        ########################################################

        exp_cp[h] = cp.E(gPCM, distr_w)
        var_cp[h] = cp.Var(gPCM, distr_w)
        
    print('MEAN')
    print("K | N | Manual \t\t\t| ChaosPy")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=exp_m[h]), '\t|', "{a:1.12f}".format(a=exp_cp[h]))

    print('VARIANCE')
    print("K | N | Manual \t\t| ChaosPy")
    for h in range(len(N)):
        print(K[h], '|', N[h], '|', "{a:1.12f}".format(a=var_m[h]), '\t|', "{a:1.12f}".format(a=var_cp[h]))
    
    print("-"*50)
    print("Relative error for mean at K")
    print(error_K_mean)
    print("Relative error for var at K")
    print(error_K_var)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 6))
    
    ax1.plot(N, error_K_mean)  # Plot the chart
    ax1.set_yscale('log')
    ax1.set_ylabel("Relative Error for Mean")

    ax2.plot(N, error_K_var)  # Plot the chart
    ax2.set_yscale('log')
    ax2.set_ylabel("Relative Error for Variance")
    ax2.set_xlabel("M samples")

    fig.tight_layout()
    fig.savefig(f'bonus_exercise_2/outputs/assignment_4.png', bbox_inches='tight')  # save_image 

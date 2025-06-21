import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time
import scipy.special as sp
import matplotlib.pyplot as plt

from typing import Union, Optional
import numpy.typing as npt
# if you want you can rely also on already implemented Oscillator class
#from utils.oscillator import Oscillator


# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid: npt.NDArray) -> npt.NDArray:
    size    = len(grid)
    w       = np.ones(size)

    for j in range(1, size):
        for k in range(j):
            diff = grid[k] - grid[j]

            w[k] *= diff
            w[j] *= -diff

    for j in range(size):
        w[j] = 1./w[j]

    return w


# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point: Union[float, npt.NDArray], grid: Union[list, npt.NDArray],
 weights: Union[list, npt.NDArray], func_eval: Union[list, npt.NDArray]) -> float:
    interp_size = len(func_eval)
    L_G         = 1.
    res         = 0.

    for i in range(interp_size):
        L_G   *= (eval_point - grid[i])

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            res = func_eval[i]
            L_G    = 1.0
            break
        else:
            res += (weights[i]*func_eval[i])/(eval_point - grid[i])

    res *= L_G 

    return res


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
    #model_kwargs = {"c": c, "k": k, "f": f}  # if you want to use the Oscillator class, you can uncomment this line
    #init_cond = {"y0": y0, "y1": y1}  # if you want to use the Oscillator class, you can uncomment this line

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t_grid      = np.array([i*dt for i in range(grid_size)])
    #t_grid = np.arange(0, t_max + dt, dt)
    t_interest  = -1
    seed = 42

    # TODO: w is no longer deterministic

    ########################################################
    # Set the edges of the uniform distribution of w       #
    ########################################################

    w_left      = 0.95
    w_right     = 1.05
    stat_ref    = [-0.43893703, 0.00019678]

    # TODO: create (chaospy) uniform distribution object
    w = cp.Uniform(w_left, w_right)

    # set the number of samples for Monte Carlo sampling
    no_grid_points_vec = [2, 5, 10, 20]
    # set the number of grid points for building Lagrange interpolation
    no_samples_vec = [10, 100, 1000, 10000]

    # create vectors that contain the expectations and variances (for Lagrange+MC and for only MC)
    err_exps_lagrange = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    err_vars_lagrange = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    err_exps_mcs = np.zeros(len(no_samples_vec))
    err_vars_mcs = np.zeros(len(no_samples_vec))

    # create vectors for storing time measurements (for Lagrange+MC and for only MC)
    lagrange_time = np.zeros( (len(no_grid_points_vec), len(no_samples_vec)) )
    mc_time = np.zeros(len(no_samples_vec))

    # compute relative error
    relative_error = lambda approx, ref: np.abs(1. - approx/ref)

    rel_error_mean_compilation_interpol = []
    rel_error_var_compilation_interpol = []
    time_compilation_interpol = []
    rel_error_mean_compilation_mc = []
    rel_error_var_compilation_mc = []
    time_compilation_mc = []

    plot_N = 5

    # TODO: builde interpolation-based surrogate model and comparing the stat. computed using the surrogate with a simple Monte Carlo sampling
    # iterate over vector containing different numbers of interpolation points
    for j, no_grid_points in enumerate(no_grid_points_vec):
        # TODO: a) Create the interpolant and evaluate the integral on the Lagrange interpolant using MC

        # TODO: a.1) generate the uniform grid and/or Chebyshev grid (i.e., experiments with one or the other),

        start_time_inter = time.time()

        ########################################################
        # Build the grid over the w domain of U()              #
        ########################################################

        nodes, _ = sp.roots_chebyc(no_grid_points)

        ###########################################################
        # roots_chebyc returns the nodes in the interval [-2, 2]  #
        # We need to scale them to be in the [w_right, w_left].   #
        ###########################################################

        chebyshev_grid = (nodes + 2) / 4 * (w_right - w_left) + w_left

        ########################################################
        # Compute the weights                                  #
        ########################################################

        weights = compute_barycentric_weights(chebyshev_grid)
        
        # TODO: a.2) evaluate the function, and perform the interpolation
        
        y_10 = []
        for i in range(no_grid_points):
            y_10.append(discretize_oscillator_odeint(model, atol, rtol, init_cond, args=(c,k,f,chebyshev_grid[i]), t=t_grid, t_interest=t_interest))

        time_x_fx = time.time() - start_time_inter
        
        for m in no_samples_vec:
            w_s = w.sample(size=m, seed=seed)
            I_n_g_y = np.zeros_like(w_s)

            ########################################################
            # Evaluate the interpolation model                     #
            ########################################################

            start_iter_interpolation = time.time()

            for i, w_i in enumerate(w_s):
                I_n_g_y[i] = barycentric_interp(w_i, chebyshev_grid, weights, y_10)

            end_inter_interpolation = time.time() - start_iter_interpolation

            # TODO: b) Evaluate the integral directly using MC sampling

            ########################################################
            # Evaluate the actual model with MC                    #
            ########################################################

            start_iter_mc = time.time()

            y_10_mc = np.zeros_like(w_s)
            for i, w_i in enumerate(w_s):
                y_10_mc[i] = discretize_oscillator_odeint(model, atol, rtol, init_cond, args=(c,k,f,w_i), t=t_grid, t_interest=t_interest)

            end_time_mc = time.time() - start_iter_mc

            # TODO: c) compute expectation and variance and measure runtime

            ########################################################
            # Quantities of Interest for Interpolation             #
            ########################################################

            Exp_Interpolation = np.mean(I_n_g_y)
            Var_Interpolation = np.var(I_n_g_y, ddof=1)

            ########################################################
            # Quantities of Interest for MC                        #
            ########################################################

            Exp_MC = np.mean(y_10_mc)
            Var_MC = np.var(y_10_mc, ddof=1)
            
            print(f"-"*50)
            print(f"With N = {no_grid_points} and M = {m}")
            print(f"Interpolation: Mean = {Exp_Interpolation}, Var = {Var_Interpolation}")
            rel_error_mean_int = relative_error(Exp_Interpolation, stat_ref[0])
            rel_error_var_int = relative_error(Var_Interpolation, stat_ref[1])
            print(f"Interpolation: Rel_Error Mean = {rel_error_mean_int}, Rel_Error Var = {rel_error_var_int}")
            print(f"Interpolation Total time: {time_x_fx + end_inter_interpolation}")
            print(f"MC: Mean = {Exp_MC}, Var = {Var_MC}")
            rel_error_mean_mc = relative_error(Exp_MC, stat_ref[0])
            rel_error_var_mc = relative_error(Var_MC, stat_ref[1])
            print(f"MC: Rel_Error Mean = {rel_error_mean_mc}, Rel_Error Var = {rel_error_var_mc}")
            print(f"MC Total time: {end_time_mc}")
            print(f"-"*50)

            if no_grid_points == plot_N:
                rel_error_mean_compilation_interpol.append(rel_error_mean_int)
                rel_error_mean_compilation_mc.append(rel_error_mean_mc)
                rel_error_var_compilation_interpol.append(rel_error_var_int)
                rel_error_var_compilation_mc.append(rel_error_var_mc)
                time_compilation_interpol.append(time_x_fx + end_inter_interpolation)
                time_compilation_mc.append(end_time_mc)


    ########################################################
    # Plot results                                         #
    ########################################################

    rel_error_mean_compilation_interpol = np.array(rel_error_mean_compilation_interpol)
    rel_error_var_compilation_interpol = np.array(rel_error_var_compilation_interpol)
    time_compilation_interpol = np.array(time_compilation_interpol)
    rel_error_mean_compilation_mc = np.array(rel_error_mean_compilation_mc)
    rel_error_var_compilation_mc = np.array(rel_error_var_compilation_mc)
    time_compilation_mc = np.array(time_compilation_mc)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    
    ax1.plot(no_samples_vec, rel_error_mean_compilation_interpol)  # Plot the chart
    ax1.plot(no_samples_vec, rel_error_mean_compilation_mc)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title(f"Mean Relative Error for N={plot_N}")
    ax1.set_ylabel("Relative Error for Mean")
    ax1.legend(["Interpolation", "Simple MC"])

    ax2.plot(no_samples_vec, rel_error_var_compilation_interpol)  # Plot the chart
    ax2.plot(no_samples_vec, rel_error_var_compilation_mc)
    ax2.set_yscale('log')
    ax1.set_xscale('log')
    ax2.set_title(f"Var Relative Error for N={plot_N}")
    ax2.set_ylabel("Relative Error for Variance")
    ax2.legend(["Interpolation", "Simple MC"])

    ax3.set_xlabel("M samples")
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.plot(no_samples_vec, time_compilation_interpol)  # Plot the chart
    ax3.plot(no_samples_vec, time_compilation_mc)
    ax3.set_title(f"Time for N={plot_N}")
    ax3.set_ylabel("Overall Time (s)")
    ax3.legend(["Interpolation", "Simple MC"])

    fig.tight_layout()
    fig.savefig(f'bonus_exercise_2/outputs/assignment_1_N{plot_N}.png', bbox_inches='tight')  # save_image 
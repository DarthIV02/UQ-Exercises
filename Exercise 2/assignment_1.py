import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

from typing import Union, Optional
import numpy.typing as npt
# if you want you can rely also on already implemented Oscillator class
# from utils.oscillator import Oscillator


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
    # model_kwargs = {"c": c, "k": k, "f": f}  # if you want to use the Oscillator class, you can uncomment this line
    # init_cond = {"y0": y0, "y1": y1}  # if you want to use the Oscillator class, you can uncomment this line

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t_grid          = np.array([i*dt for i in range(grid_size)])
    #t_grid = np.arange(0, t_max + dt, dt)
    t_interest  = -1

    # TODO: w is no longer deterministic
    w_left      = None
    w_right     = None
    stat_ref    = [-0.43893703, 0.00019678]

    # TODO: create (chaospy) uniform distribution object

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

    # TODO: builde interpolation-based surrogate model and comparing the stat. computed using the surrogate with a simple Monte Carlo sampling
    # iterate over vector containing different numbers of interpolation points
    for j, no_grid_points in enumerate(no_grid_points_vec):
        # TODO: a) Create the interpolant and evaluate the integral on the Lagrange interpolant using MC
        # TODO: a.1) generate the uniform grid and/or Chebyshev grid (i.e., experiments with one or the other),
        # TODO: a.2) evaluate the function, and perform the interpolation
        # TODO: b) Evaluate the integral directly using MC sampling
        # TODO: c) compute expectation and variance and measure runtime



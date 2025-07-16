from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class WienerProcess:
    mu: float
    T: float | None = None
    n_points: float | None = None
    t_grid: npt.NDArray | None = None

    def __post_init__(self):
        if self.T is None and self.n_points is None:
            self.T = self.t_grid[-1]
            self.n_points = len(self.t_grid)
        if self.t_grid is None:
            self.t_grid = np.linspace(0, self.T, self.n_points)

    def generate(self, n_samples: int, rng: np.random.Generator):

        # TODO: generate n_samples realizations of the Wiener process
        # using the standard definition.
        
        W = np.zeros(((n_samples, self.n_points)))
        M = self.n_points - 1

        for n in range(n_samples): # For each sample -> generate a different Wiener realization over t_grid
            dW = rng.normal(size=(M), loc=0, scale=np.sqrt(self.T / M))   # increments ~ N(0, dt)
            W[n] = np.concatenate([[0], np.cumsum(dW)])  # W_0 = 0, then add the current increment + the previous value

        return W

    def approximate_kl(self, n_samples: int, M: int, rng: np.random.Generator):

        kl = np.zeros((n_samples, self.n_points)) # Storing values
        # eigenvalues = function -> return (m,)
        # eigenfunctions = function: input t -> returns (m) 
        eigenvalues, eigenfunctions = self.kl_eigenpairs(M)

        try:
            self.zeta = self.zeta
        except:
            self.zeta = rng.random(size=(n_samples, self.n_points))

        for n in range(n_samples):
            #self.zeta is created outside to use the same values 
            # Then it is cropped to the needed value of M
            part = np.multiply(np.sqrt(eigenvalues), self.zeta[n, :M])

            for t, ti in enumerate(self.t_grid): # At each time step
                functions = eigenfunctions(ti) # Functions evaluated at t
                kl[n, t] = np.sum(np.multiply(functions, part)) # Follow the approximation formula

        return kl # Realizations for the t_grid for each sample

    def kl_eigenvalues(self, M: int):

        # TODO: compute the first M eigenvalues of the Wiener process.
        m = np.arange(1, M + 1)
        return (self.T ** 2) / (((m+0.5)**2)*(np.pi**2))

    def kl_eigenfunctions(self, M: int):

        # TODO: compute the first M eigenfunctions of the Wiener process.
        # It might be more conveniet to return a callable function that
        # returns evaluations of the first M eigenfunctions for the provided
        # time points.
        m = np.arange(1, M + 1)
        return lambda t: np.sqrt(2 / self.T) * np.sin((np.pi*t*(m + 1/2))/self.T)

    def kl_eigenpairs(self, M: int):
        eigenvalues = self.kl_eigenvalues(M)
        eigenfunctions = self.kl_eigenfunctions(M)
        return eigenvalues, eigenfunctions
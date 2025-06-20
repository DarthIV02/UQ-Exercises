import chaospy as cp
import numpy as np

if __name__ == '__main__':
	# TODO: define the two distributions
	unif_distr = None
	norm_distr = None

    # degrees of the polynomials
    N = [2, 5, 8]  # N = [8,]

	# generate orthogonal polynomials for all N's
	for i, n in enumerate(N):
		
		# TODO: employ the three terms recursion scheme using chaospy to generate orthonormal polynomials w.r.t. the two distributions

		# TODO: compute <\phi_j(x), \phi_k(x)>_\rho, i.e. E[\phi_j(x) \phi_k(x)]

		# TODO: print result for specific n

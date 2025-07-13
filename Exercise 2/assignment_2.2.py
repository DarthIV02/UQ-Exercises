import chaospy as cp
import numpy as np

if __name__ == '__main__':
	# TODO: define the two distributions
	unif_distr = cp.Uniform(-1,1)
	norm_distr = cp.Normal(10, 1)

    # degrees of the polynomials
	N = [8]

	# generate orthogonal polynomials for all N's


	for i, n in enumerate(N):
		# TODO: employ the three terms recursion scheme using chaospy to generate orthonormal polynomials w.r.t. the two distributions

		unif_pol = cp.generate_expansion(n, dist=unif_distr, normed=False) 
		norm_pol = cp.generate_expansion(n, dist=norm_distr, normed=False)

		unif_pol_normed = cp.generate_expansion(n, dist=unif_distr, normed=True) 
		norm_pol_normed = cp.generate_expansion(n, dist=norm_distr, normed=True)

		# TODO: compute <\phi_j(x), \phi_k(x)>_\rho, i.e. E[\phi_j(x) \phi_k(x)]

		#unif_expect = [[cp.E(phi_j * phi_k, dist=unif_distr) for phi_j in unif_pol] for phi_k in unif_pol]
		#norm_expect = [[cp.E(phi_j * phi_k, dist=norm_distr) for phi_j in norm_pol] for phi_k in norm_pol]

		unif_expect_normed = [[cp.E(phi_j * phi_k, dist=unif_distr) for phi_j in unif_pol_normed] for phi_k in unif_pol_normed]
		norm_expect_normed = [[cp.E(phi_j * phi_k, dist=norm_distr) for phi_j in norm_pol_normed] for phi_k in norm_pol_normed]


		# TODO: print result for specific n
		print("NORMED: \r")
		#print(f"Normed UNIFORM pols n={n}: {unif_pol_normed}")
		#print(f"Normed NORMAL pols n={n}: {norm_pol_normed}")
		print("\r")
		print(f"Expected Value UNIFORM for n={n}: ") #we get identity matrix of dim=n, as <phi_j, phi_k> = \delta_jk
		print(unif_expect_normed)
		print(f"Expected Value NORMAL for n={n}: ")
		print(norm_expect_normed)
		print("\r")
		print("\r")
		

		#IF	WE WANT NON-NORMALIZED ONES
		#print("NOT NORMED: \r")
		#print(f"UNIFORM pols n={n}: {unif_pol}")
		#print(f"NORMAL pols n={n}: {norm_pol}")
		#print("\r")
		#print(f"Expected Value UNIFORM for n={n}: ") 
		#print(unif_expect)
		#print(f"Expected Value NORMAL for n={n}: ") #if normed=False, we get diag(|phi_1|, |phi-2|, ..., |phi_n|) instead of diag(1,...,1)
		#print(norm_expect)

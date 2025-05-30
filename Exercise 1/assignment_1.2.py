import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from utils.sampling import compute_rmse


def sample_normal(
    n_samples: int, mu_target: npt.NDArray, V_target: npt.NDArray, seed: int = 42
) -> npt.NDArray:
    # TODO: generate samples from multivariate normal distribution.
    # ====================================================================
    
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mu_target, V_target, size=(n_samples))

    # ====================================================================
    return samples


def compute_moments(samples: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # TODO: estimate mean and covariance of the samples.
    # ====================================================================
    mean = np.mean(samples, axis=0)
    covariance = np.zeros((samples.shape[1], samples.shape[1]))
    
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            covariance[i,j] = np.sum((samples[:, i] - mean[i]) * (samples[:, j] - mean[j]))
    
    covariance *= (1/(len(samples)-1))

    variance = np.var(samples, ddof=1, axis=0)
    rmse = np.sqrt(variance / samples.shape[0])
    
    # ====================================================================
    return mean, covariance, rmse

if __name__ == "__main__":
    
    
    # TODO: define the parameters of the simulation.
    # ====================================================================
    
    sample_size = [10, 100, 1000, 10000]
    miu_target = np.array([-0.4, 1.1])
    cov = np.array([[2, 0.4], [0.4, 1]])
    seed = 42

    # ====================================================================

    # TODO: estimate mean, covariance, and compute the required errors.
    # ====================================================================

    rmse_total = []
    mean_per_samples = []
    covariance_per_samples = []
    
    for i, s in enumerate(sample_size):
        samples = sample_normal(s, miu_target, cov, seed + i)
        mean_sample, covariance_sample, rmse = compute_moments(samples)
        print(f"\nFor sample size {sample_size[i]}: \n"
              f"mean_sample: {mean_sample[0]} \n" \
              f"covariance_sample: {covariance_sample[0]}")
        mean_per_samples.append(mean_sample)
        covariance_per_samples.append(covariance_sample)
        rmse_total.append(rmse)

    # ====================================================================

    # TODO: plot the results on the log-log scale.
    # ====================================================================
    rmse_total = np.array(rmse_total)
    mean_per_samples = np.array(mean_per_samples)
    covariance_per_samples = np.array(covariance_per_samples)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(sample_size, rmse_total)  # Plot the chart
    ax1.set_xscale('log')
    ax1.set_title("RMSE for mean estimator")
    ax1.set_ylabel("RMSE")
    ax1.legend(["RMSE for mean value 0", "RMSE for mean value 1"])

    ax2.set_xlabel("Number of Samples")
    mean_per_samples[:, 0] = np.abs(mean_per_samples[:, 0] - miu_target[0])
    covariance_per_samples[:, 0] = np.abs(covariance_per_samples[:, 0] - cov[0])
    mean_cov = np.concatenate((mean_per_samples[:, 0].reshape((len(sample_size), 1)), covariance_per_samples[:, 0]), axis=1)
    ax2.plot(sample_size, mean_cov)
    ax2.set_title("Absolute Error for each estimator")
    ax2.set_ylabel("Absolute Error")
    ax2.legend(["Error for mean value 0", "Error for cov value 0,0", "Error for cov value 0,1"])

    fig.tight_layout()
    fig.savefig('bonus_exercise_1/outputs/assignment_1.2.png', bbox_inches='tight')  # save_image 
    # ====================================================================

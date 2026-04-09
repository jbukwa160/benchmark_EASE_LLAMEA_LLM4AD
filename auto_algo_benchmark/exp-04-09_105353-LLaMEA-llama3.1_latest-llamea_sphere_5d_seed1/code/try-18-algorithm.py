import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim) * 0.1
    importance_sampling_ratio = 0.01

    for i in range(budget):
        # Importance sampling to focus on promising regions of search space
        if rng.random() < importance_sampling_ratio:
            x_new = np.zeros(dim)
            f_new = func(x_new)

        else:
            # Sample from a normal distribution with the current covariance matrix
            x_new = np.random.multivariate_normal(x_opt, cov_mat)

        # Clip candidates to [lb, ub]
        x_clipped = np.clip(x_new, lb, ub)

        # Evaluate the objective function at the new point
        f_new = func(x_clipped)

        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

            # Update the covariance matrix using importance sampling ratio
            cov_mat += importance_sampling_ratio * (x_clipped - x_opt) * (x_clipped - x_opt).T

        else:
            # Reduce the covariance matrix when the new point is worse than the current optimal point
            cov_mat *= 0.9
            if np.linalg.det(cov_mat) < 1e-6:
                cov_mat = np.eye(dim) * 0.1

    return x_opt
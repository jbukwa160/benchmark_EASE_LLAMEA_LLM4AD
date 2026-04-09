import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim) * 0.1
    importance_sampling_ratio = 0.01

    success_rate = 0

    for i in range(budget):
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
            success_rate += 1

            x_opt = x_clipped
            f_opt = f_new

            # Update the covariance matrix using importance sampling ratio and success rate
            cov_mat += (0.1 + 0.9 * (success_rate / (i + 1))) * (x_clipped - x_opt) * (x_clipped - x_opt).T

        else:
            success_rate = max(0, success_rate - 0.01)

    return x_opt
import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim) * 0.1
    importance_sampling_ratio = 0.01

    for i in range(budget):
        if rng.random() < importance_sampling_ratio:
            x_new = np.clip(np.random.uniform(lb, ub, size=dim), lb, ub)
            f_new = func(x_new)

        else:
            # Sample from a normal distribution with the current covariance matrix
            x_new = cov_mat @ np.random.multivariate_normal(np.zeros(dim), np.eye(dim)) + x_opt

        # Clip candidates to [lb, ub]
        x_clipped = np.clip(x_new, lb, ub)

        # Evaluate the objective function at the new point
        f_new = func(x_clipped)

        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

            # Update the covariance matrix using importance sampling ratio and a more effective way to update the covariance matrix
            cov_mat += importance_sampling_ratio * (x_clipped - x_opt).reshape(-1, 1) @ (x_clipped - x_opt).reshape(1, -1)

    return x_opt
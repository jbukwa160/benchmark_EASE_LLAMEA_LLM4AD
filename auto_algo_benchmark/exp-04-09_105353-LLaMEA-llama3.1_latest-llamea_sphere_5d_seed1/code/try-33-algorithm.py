import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim) * 0.1
    importance_sampling_ratio = 0.01

    for i in range(budget):
        if rng.random() < importance_sampling_ratio:
            x_new = np.array([lb + (ub - lb) * rng.uniform(0, 1) for _ in range(dim)])
        else:
            x_new = np.random.multivariate_normal(x_opt, cov_mat)

        x_clipped = np.clip(x_new, lb, ub)
        f_new = func(x_clipped)

        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

            # Update the covariance matrix using importance sampling ratio
            step_size = 0.01
            cov_mat += importance_sampling_ratio * step_size * (x_clipped - x_opt) * (x_clipped - x_opt).T / np.linalg.det(cov_mat)

    return x_opt
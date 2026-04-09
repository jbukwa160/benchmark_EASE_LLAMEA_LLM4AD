import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim) * 1e-6

    for i in range(budget):
        mean_new = x_opt + (ub - lb) * (np.random.rand(dim) - 0.5)
        cov_new = cov_mat
        x_new = np.random.multivariate_normal(mean_new, cov_new)

        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

        # Update covariance matrix with importance sampling
        w = (f_opt - f_new) / (f_opt + 1e-6)
        cov_mat += w * np.outer(x_new - x_opt, x_new - x_opt)

    return x_opt
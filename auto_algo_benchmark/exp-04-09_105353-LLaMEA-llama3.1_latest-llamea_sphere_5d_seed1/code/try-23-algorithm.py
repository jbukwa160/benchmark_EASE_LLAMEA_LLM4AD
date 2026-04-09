import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim) * 0.1
    importance_sampling_ratio = 0.01
    reduction_factor = 0.9  # reduce covariance matrix when new point is not better

    for i in range(budget):
        if rng.random() < importance_sampling_ratio:
            x_new = np.zeros(dim)
            f_new = func(x_new)

        else:
            x_new = np.random.multivariate_normal(x_opt, cov_mat)

        x_clipped = np.clip(x_new, lb, ub)
        f_new = func(x_clipped)

        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

            cov_mat += importance_sampling_ratio * (x_clipped - x_opt) * (x_clipped - x_opt).T
        else:
            # Reduce the covariance matrix when the new point is not better
            cov_mat *= reduction_factor

    return x_opt
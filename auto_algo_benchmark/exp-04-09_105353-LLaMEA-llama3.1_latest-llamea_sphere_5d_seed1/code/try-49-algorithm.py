import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    for i in range(budget):
        if i > 0:
            mean = (i * x_opt + (budget - i) * prev_x) / budget
            cov = ((budget - i) * cov_prev + i * np.outer(x_opt - mean, x_opt - mean)) / budget
            chol = np.linalg.cholesky(cov)
        else:
            chol = np.eye(dim)

        x_new = mean + (ub - lb) * chol @ rng.randn(dim)
        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            prev_x = x_opt
            cov_prev = np.outer(x_opt - mean, x_opt - mean)
            x_opt = x_clipped
            f_opt = f_new

    return x_opt
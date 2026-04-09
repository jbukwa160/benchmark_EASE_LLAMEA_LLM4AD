import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov = np.eye(dim) * (ub - lb)**2
    for i in range(budget):
        x_new = x_opt + (ub - lb) * np.random.multivariate_normal(np.zeros(dim), cov)
        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

        # Update covariance matrix based on new function value
        delta = x_clipped - x_opt
        cov += (f_new - f_opt) * np.outer(delta, delta)

    return x_opt
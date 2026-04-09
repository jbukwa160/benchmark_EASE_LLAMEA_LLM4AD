import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov = np.eye(dim) * 0.1
    weights = np.ones(budget)

    for i in range(budget):
        x_new = np.random.multivariate_normal(x_opt, cov)
        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

        weights[i] *= (f_opt - f_new) + 1e-6
        cov += 0.5 * (np.outer((x_clipped - x_opt), (x_clipped - x_opt)) - cov)

    return x_opt
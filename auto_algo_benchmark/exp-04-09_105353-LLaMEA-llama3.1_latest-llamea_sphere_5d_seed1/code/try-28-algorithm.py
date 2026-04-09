import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov = np.eye(dim) * 0.1
    for i in range(budget):
        x_new = x_opt + np.random.multivariate_normal(np.zeros(dim), cov)
        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

        cov += 0.01 * (x_new - x_opt) * (x_new - x_opt).T
    return x_opt
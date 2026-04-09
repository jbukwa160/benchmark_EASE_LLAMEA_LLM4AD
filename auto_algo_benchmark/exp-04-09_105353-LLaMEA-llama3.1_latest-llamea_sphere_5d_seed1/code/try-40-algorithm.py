import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4
    for i in range(budget):
        mean = x_opt + (ub - lb) / 2
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        x_clipped = np.clip(sample, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            cov_matrix += 0.1 * (x_clipped - x_opt)[:, None] @ (x_clipped - x_opt)[None, :]
            x_opt = x_clipped
            f_opt = f_new

    return x_opt
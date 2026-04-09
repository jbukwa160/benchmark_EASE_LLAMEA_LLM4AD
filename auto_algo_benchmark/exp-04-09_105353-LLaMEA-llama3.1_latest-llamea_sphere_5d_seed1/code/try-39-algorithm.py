import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4
    step_size = (ub - lb) / 2
    for i in range(budget):
        mean = x_opt + step_size
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        x_clipped = np.clip(sample, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

        cov_matrix += 0.1 * (x_opt - mean)[:, None] @ (x_opt - mean)[None, :]
        step_size *= 0.99 + np.random.rand() * 0.01

    return x_opt
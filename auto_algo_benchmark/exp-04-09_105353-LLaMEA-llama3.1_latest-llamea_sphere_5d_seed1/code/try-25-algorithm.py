import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4
    for i in range(budget):
        mean = x_opt + (ub - lb) / 2
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        importance_prob = func(sample)
        sample *= np.exp(-importance_prob)

        x_clipped = np.clip(mean + sample, lb, ub)
        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

    cov_matrix = (1 - 1 / (i + 2)) * cov_matrix + 1 / (i + 2) * np.dot(sample.T, sample)
    return x_opt
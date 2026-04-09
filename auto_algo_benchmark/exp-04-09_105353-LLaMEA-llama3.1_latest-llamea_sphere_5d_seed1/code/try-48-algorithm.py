import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4
    for i in range(budget):
        mean = x_opt + (ub - lb) / 2
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        
        # Use a more robust method to generate samples from the multivariate normal distribution
        sample = np.clip(np.round(sample * 1e6) / 1e6, lb, ub)

        f_new = func(sample)
        if f_new < f_opt:
            x_opt = sample
            f_opt = f_new

    return x_opt
import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4

    for i in range(budget):
        mean = x_opt + (ub - lb) / 2
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        
        # Incorporate importance sampling by using the current best solution as a reference point
        r = i / budget
        cov_matrix *= (1 - r) * (ub - lb)**2 / 4 + r
        
        x_clipped = np.clip(sample, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

    return x_opt
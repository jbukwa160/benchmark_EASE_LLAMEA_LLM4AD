import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4
    for i in range(budget):
        mean = x_opt + (ub - lb) / 2
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        
        # Importance sampling: focus on promising regions
        importance = np.exp(-f_opt) if f_opt != 0 else 1
        sample *= importance
        
        x_clipped = np.clip(sample, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

        # Update covariance matrix to focus on promising regions
        cov_matrix += (x_clipped - mean) * (x_clipped - mean).T / importance

    return x_opt
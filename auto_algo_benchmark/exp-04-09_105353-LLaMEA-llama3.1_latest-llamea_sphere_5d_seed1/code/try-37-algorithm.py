import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4
    importance_factors = np.exp(-f_opt) # Importance factors based on previous solution's fitness

    for i in range(budget):
        mean = x_opt + (ub - lb) / 2
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        importance_weight = importance_factors * np.exp(-np.sum(sample**2)) # Importance weight based on new solution's fitness and distance from current best

        x_clipped = np.clip(sample, lb, ub)

        f_new = func(x_clipped)
        weight_ratio = importance_weight / (importance_weight + 1e-8) # Avoid division by zero
        acceptance_prob = weight_ratio ** 2 # Acceptance probability based on importance weights

        if rng.random() < acceptance_prob:
            x_opt = x_clipped
            f_opt = f_new
            cov_matrix += np.outer((x_clipped - x_opt), (x_clipped - x_opt)) / (1 + i) # Update covariance matrix with new solution
            importance_factors *= weight_ratio ** 2

    return x_opt
import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    mean = x_opt
    cov = (ub - lb) * np.eye(dim)

    for i in range(budget):
        # Importance sampling: sample from a distribution centered at the current optimum with a small spread
        x_new = rng.normal(mean, 0.01)
        # Clip candidates to [lb, ub]
        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        
        if f_new < f_opt:
            mean = (i/(i+1))*mean + (1/(i+1))*x_clipped
            cov = (i/(i+1))*cov + (1/(i+1))*(np.outer(x_clipped - mean, x_clipped - mean))
            # Sample from a multivariate normal distribution with the updated mean and covariance matrix
            x_new = rng.multivariate_normal(mean, cov)
            f_opt = f_new

    return x_opt
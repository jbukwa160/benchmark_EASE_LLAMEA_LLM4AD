import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov = np.eye(dim) * 1e-5  # initialize covariance matrix
    for i in range(budget):
        x_new = x_opt + (ub - lb) * np.dot(np.linalg.cholesky(cov), (np.random.rand(dim) - 0.5))
        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

        cov += (x_new - x_opt).dot((x_new - x_opt).T)  # update covariance matrix

    return x_opt
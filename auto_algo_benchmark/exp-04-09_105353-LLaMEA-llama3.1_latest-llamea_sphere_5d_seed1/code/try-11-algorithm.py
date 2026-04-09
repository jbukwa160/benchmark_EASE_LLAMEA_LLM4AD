import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim)  # initialize covariance matrix
    w = np.ones(dim) / dim  # initialize weights (importance sampling)

    for i in range(budget):
        # propose new candidate using multivariate normal distribution with mean x_opt and covariance matrix cov_mat
        x_new = x_opt + np.dot(np.linalg.cholesky(cov_mat), np.random.randn(dim))

        # clip to bounds
        x_clipped = np.clip(x_new, lb, ub)

        # compute importance sampling weight for new candidate
        w_new = func(x_clipped) * (1 / (func(x_opt) + 0.01))  # add a small constant to avoid division by zero

        # update weights and covariance matrix using incremental EM algorithm
        w = (w * np.exp(-w_new)) / np.sum(w * np.exp(-w_new))
        cov_mat = (1 - 2 * w[None, :] @ w[:, None] + 2 * np.outer(w, w)) @ cov_mat

        # evaluate new candidate and update x_opt if better
        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

    return x_opt
import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = np.random.uniform(lb, ub, (1, dim))
    f_opt = func(x)
    x_opt = x.copy()

    for i in range(1, budget):
        mu = x_opt
        cov = 0.5 * np.eye(dim)

        # Adapt covariance matrix based on previous performance
        if i > 10:
            mean_diff = np.mean(x - x_opt, axis=0)
            cov += (mean_diff[:, None] @ mean_diff[None, :]) / dim

        s = np.random.multivariate_normal(mu.flatten(), cov)

        x_new = mu + s.reshape((-1, dim))
        f_new = func(x_new)

        if f_new < f_opt:
            f_opt = f_new
            x_opt = x_new.copy()

        # Clip candidates to [lb, ub]
        for j in range(dim):
            x_opt[j] = max(lb[j], min(ub[j], x_opt[j]))

    return f_opt, x_opt[-1]
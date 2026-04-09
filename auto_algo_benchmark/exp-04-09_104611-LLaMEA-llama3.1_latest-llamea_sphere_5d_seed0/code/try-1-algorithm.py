import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = np.random.uniform(lb, ub, (1, dim))
    f_opt = func(x)
    x_opt = x.copy()

    for i in range(1, budget):
        mean, cov = gp_mean_cov(x, rng)
        std = np.sqrt(np.diag(cov))
        mu = mean + std * np.random.randn(dim)
        sigma = np.exp(rng.uniform(-2, 0)) # scale the variance
        s = np.random.multivariate_normal(mu, sigma**2 * cov)

        x_new = x + s
        f_new = func(x_new)

        if f_new < f_opt:
            f_opt = f_new
            x_opt = x_new.copy()

        # Clip candidates to [lb, ub]
        for j in range(dim):
            x_opt[j] = max(lb[j], min(ub[j], x_opt[j]))
            x_new[j] = max(lb[j], min(ub[j], x_new[j]))

        x = np.vstack((x, x_new))

    return f_opt, x_opt[-1]
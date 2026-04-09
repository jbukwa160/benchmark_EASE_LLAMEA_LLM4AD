import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = np.random.uniform(lb, ub, (1, dim))
    f_opt = func(x)
    x_opt = x.copy()

    for i in range(1, budget):
        mean, cov = gp_mean_cov(x, rng)

        # CMA-ES update
        sigma = 0.5 / np.sqrt(np.log(i + 2)) * (ub - lb) / 3
        s = np.random.multivariate_normal(mean, sigma**2 * cov)
        x_new = x + s

        for j in range(dim):
            x_new[j] = max(lb[j], min(ub[j], x_new[j]))

        # GP regression update
        f_new = func(x_new)

        if f_new < f_opt:
            f_opt = f_new
            x_opt = x_new.copy()

        # Sample from the distribution of the best solution found so far
        mean, cov = gp_mean_cov(x_opt, rng)
        s = np.random.multivariate_normal(mean, cov)

        for j in range(dim):
            candidate = max(lb[j], min(ub[j], x_opt[j] + 0.1 * s[j]))
            if func(np.array([[candidate]])) < f_new:
                x_new[j] = candidate
                f_new = func(x_new)
                if f_new < f_opt:
                    f_opt = f_new
                    x_opt = x_new.copy()

        x = np.vstack((x, x_new))

    return f_opt, x_opt[-1]
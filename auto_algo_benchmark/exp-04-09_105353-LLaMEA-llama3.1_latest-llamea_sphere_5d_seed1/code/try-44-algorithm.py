import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    for i in range(budget):
        # Importance sampling to focus on promising regions of search space
        sigma = 0.1 + (ub - lb) / np.sqrt(i + 1)
        mean = x_opt + sigma * np.random.randn(dim)
        x_new = np.clip(mean, lb, ub)

        f_new = func(x_new)
        if f_new < f_opt:
            x_opt = x_new
            f_opt = f_new

    return x_opt
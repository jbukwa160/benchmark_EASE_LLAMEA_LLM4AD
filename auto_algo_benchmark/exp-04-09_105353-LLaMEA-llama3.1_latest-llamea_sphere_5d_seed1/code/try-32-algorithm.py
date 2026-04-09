import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    for i in range(budget):
        x_new = x_opt + (ub - lb) * (np.random.rand(dim) - 0.5)
        x_clipped = np.clip(x_new, lb, ub)

        importance = np.exp(-f_opt + func(x_clipped))
        probability = importance / np.sum(importance)

        if rng.uniform(0, 1) < probability:
            x_opt = x_clipped
            f_opt = func(x_clipped)

    return x_opt
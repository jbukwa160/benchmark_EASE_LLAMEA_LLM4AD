import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    for i in range(budget):
        # Importance sampling: select sample based on its likelihood of being better than the current best
        beta = 1.0 - (i + 1) / (budget + 1)
        x_new = x_opt + (ub - lb) * np.power(np.random.rand(dim), beta)

        # Clip candidate to bounds
        x_clipped = np.clip(x_new, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

    return x_opt
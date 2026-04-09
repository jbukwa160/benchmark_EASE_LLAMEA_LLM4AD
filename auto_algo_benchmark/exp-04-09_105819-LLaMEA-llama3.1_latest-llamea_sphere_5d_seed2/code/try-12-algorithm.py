import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    T0 = 100.0
    alpha = 0.999
    x_opt = np.random.uniform(size=dim)
    x_best = x_opt.copy()
    f_best = func(x_opt)

    for i in range(budget):
        x_new = x_opt + (ub - lb) * rng.rand(dim) - (ub - lb)/2
        x_new = np.clip(x_new, lb, ub)
        f_new = func(x_new)

        if f_new < f_best:
            x_best = x_new.copy()
            f_best = f_new

        delta = f_new - f_best
        if delta < 0 or rng.rand() < np.exp(-delta/T0):
            T0 *= alpha
            x_opt = x_new

    return x_best
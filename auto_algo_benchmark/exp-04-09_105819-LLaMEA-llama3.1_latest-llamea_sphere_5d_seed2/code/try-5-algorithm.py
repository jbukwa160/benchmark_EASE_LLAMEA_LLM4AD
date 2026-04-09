import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    T = 1000.0
    alpha = 0.99
    x_opt = np.random.uniform(size=dim)
    x_best = x_opt.copy()
    f_best = func(x_opt)

    for i in range(budget):
        if rng.rand() < 0.1: # restart with new initial solution every 10 iterations
            T *= 0.9
            x_opt = np.random.uniform(size=dim)
        
        x_new = x_opt + (ub - lb) * rng.randn(dim) - (ub - lb)/2
        x_new = np.clip(x_new, lb, ub)
        f_new = func(x_new)

        if f_new < f_best:
            x_best = x_new.copy()
            f_best = f_new

        delta = f_new - f_best
        if delta < 0 or rng.rand() < np.exp(-delta/T):
            x_opt = x_new

        T *= alpha

    return x_best
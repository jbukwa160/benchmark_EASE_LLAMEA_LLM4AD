import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    T = 1000.0
    alpha = 0.99
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
        if delta < 0 or rng.rand() < np.exp(-delta/T):
            x_opt = x_new

        T *= alpha

    # Local search phase: perform 5 iterations of local refinement around the current best solution
    for _ in range(5):
        x_search = x_best + (ub - lb) * rng.randn(dim) - (ub - lb)/2
        x_search = np.clip(x_search, lb, ub)
        f_search = func(x_search)

        if f_search < f_best:
            x_best = x_search.copy()
            f_best = f_search

    return x_best
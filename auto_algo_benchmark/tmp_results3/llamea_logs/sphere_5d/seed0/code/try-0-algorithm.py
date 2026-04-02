def algorithm(func, dim, lb, ub, budget, rng):
    import numpy as np
    x = np.zeros(dim)
    x = np.clip(x, lb, ub)
    return x
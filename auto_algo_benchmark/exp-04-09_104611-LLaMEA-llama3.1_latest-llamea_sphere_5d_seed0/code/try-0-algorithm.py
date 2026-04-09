import numpy as np

def algorithm(func, dim, lb, ub, budget, rng=np.random):
    x = rng.uniform(lb, ub, (1, dim))
    f_opt = np.inf
    x_opt = None

    for i in range(budget):
        s = np.sqrt((ub - lb) / dim)
        step = s * rng.normal(size=(1, dim))

        new_x = x + step
        new_x = np.clip(new_x, lb, ub)

        f = func(new_x)

        if f < f_opt:
            f_opt = f
            x_opt = new_x

        x = new_x

    return x_opt
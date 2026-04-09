import numpy as np

def algorithm(func, dim, lb, ub, budget, rng=np.random):
    x = rng.uniform(lb, ub, (1, dim))
    f_opt = np.inf
    x_opt = None
    lr = 1.0

    for i in range(budget):
        s = np.sqrt((ub - lb) / dim)
        step = s * rng.normal(size=(1, dim)) * lr

        new_x = x + step
        new_x = np.clip(new_x, lb, ub)

        f = func(new_x)

        if f < f_opt:
            f_opt = f
            x_opt = new_x

        # Adaptively adjust the learning rate based on the current best score.
        if i > 0 and rng.rand() < 0.1:
            lr *= np.exp((f_opt - f) / (i + 1))

        x = new_x

    return x_opt
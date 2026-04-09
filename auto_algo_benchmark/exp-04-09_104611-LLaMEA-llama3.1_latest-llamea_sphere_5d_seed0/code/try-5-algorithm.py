import numpy as np

def algorithm(func, dim, lb, ub, budget, rng=np.random):
    x = rng.uniform(lb, ub, (1, dim))
    f_opt = np.inf
    x_opt = None
    s = (ub - lb) / dim  # initial step size
    alpha = 0.5  # learning rate

    for i in range(budget):
        step = s * rng.normal(size=(1, dim))  # sample a new direction
        new_x = x + step
        new_x = np.clip(new_x, lb, ub)

        f = func(new_x)
        if f < f_opt:
            f_opt = f
            x_opt = new_x

        delta = (f - func(x)) / s  # estimate the gradient magnitude
        if delta > 0:  # update step size if improvement is observed
            s *= np.exp(alpha * delta)

        x = new_x

    return x_opt
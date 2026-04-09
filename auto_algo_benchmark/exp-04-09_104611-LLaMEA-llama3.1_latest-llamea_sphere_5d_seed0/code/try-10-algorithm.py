import numpy as np

def algorithm(func, dim, lb, ub, budget, rng=np.random):
    x = rng.uniform(lb, ub, (1, dim))
    f_opt = np.inf
    x_opt = None

    successes = 0
    total_steps = 0

    s_max = (ub - lb) / dim
    for i in range(budget):
        s = s_max * rng.normal(size=(1, dim))

        new_x = x + s
        new_x = np.clip(new_x, lb, ub)

        f = func(new_x)
        if f < f_opt:
            successes += 1
            total_steps += 1

            f_opt = f
            x_opt = new_x

        else:
            total_steps += 1

        s_ratio = successes / (total_steps + 1) if total_steps > 0 else 0
        s_max *= np.sqrt(s_ratio)

        x = new_x

    return x_opt
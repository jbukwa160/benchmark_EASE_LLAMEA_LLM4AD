import numpy as np

def algorithm(func, dim, lb, ub, budget, rng=np.random):
    x = rng.uniform(lb, ub, (1, dim))
    f_opt = np.inf
    x_opt = None

    step_history = []
    s = np.sqrt((ub - lb) / dim)
    step_size = s * 0.5

    for i in range(budget):
        step = rng.normal(size=(1, dim)) * step_size
        new_x = x + step
        new_x = np.clip(new_x, lb, ub)

        f = func(new_x)

        if f < f_opt:
            f_opt = f
            x_opt = new_x

        if i > 0 and rng.rand() < 0.1: # adapt step-size every 10 steps
            success_rate = len([s for s in step_history if s['f'] < f_opt]) / len(step_history)
            if success_rate > 0.5:
                step_size *= 1.1
            else:
                step_size /= 1.1

        x = new_x
        step_history.append({'x': x, 'f': f})

    return x_opt
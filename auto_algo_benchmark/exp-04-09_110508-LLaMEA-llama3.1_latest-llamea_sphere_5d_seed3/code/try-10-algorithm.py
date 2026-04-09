import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=dim)
    f_opt = np.inf
    x_opt = None
    restart_threshold = 0.1 * budget

    for i in range(budget - 1):
        v = rng.normal(size=dim)
        x_new = x + 0.1 * v

        # Clip new candidate to bounds
        x_new = np.clip(x_new, lb, ub)

        f_new = func(x_new)

        if f_new < f_opt:
            f_opt = f_new
            x_opt = x_new

        # Probability of accepting worse solution (adaptively adjusted)
        alpha = 0.1 / (i + 2)
        beta = np.exp(-f_new * (i + 2) / budget)

        if rng.uniform(0, 1) < np.exp(alpha * f_opt - f_new + beta):
            x = x_new

        # Adaptive restart
        if i > restart_threshold and f_new >= f_opt:
            x = rng.uniform(lb, ub, size=dim)
            f_opt = np.inf
            restart_threshold *= 2

    return f_opt, x_opt
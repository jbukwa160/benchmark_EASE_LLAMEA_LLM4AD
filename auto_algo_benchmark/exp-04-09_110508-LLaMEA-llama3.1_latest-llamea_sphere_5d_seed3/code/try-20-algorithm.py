import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=dim)
    f_opt = np.inf
    x_opt = None

    for i in range(budget // 10): # restart every 10 iterations
        x_restart = rng.uniform(lb, ub, size=dim)

        for j in range(9):
            v1 = rng.normal(size=dim)
            v2 = rng.normal(size=dim)
            v3 = rng.normal(size=dim)

            # Calculate the offspring candidates using the DE/rand/1/bin mutation strategy
            x_new1 = x_restart + 0.5 * (v1 - x_restart) + 0.5 * (v2 - x_restart)
            x_new2 = x_restart + 0.5 * (v1 - x_restart) + 0.3 * (v2 - x_restart) + 0.4 * (v3 - x_restart)

            # Clip new candidates to bounds
            x_new1 = np.clip(x_new1, lb, ub)
            x_new2 = np.clip(x_new2, lb, ub)

            f_new1 = func(x_new1)
            f_new2 = func(x_new2)

            if f_new1 < f_opt:
                f_opt = f_new1
                x_opt = x_new1

            if f_new2 < f_opt:
                f_opt = f_new2
                x_opt = x_new2

            # Probability of accepting worse solution (adaptively adjusted)
            alpha = 0.01 / (i + 2)
            beta = np.exp(-f_new1 * (i + 2) / budget)

            if rng.uniform(0, 1) < np.exp(alpha * f_opt - f_new1 + beta):
                x = x_new1

        # Update the best solution found in this restart
        f_opt_restart = np.inf
        for k in range(dim): # evaluate all points in the last iteration of each restart
            v = rng.normal(size=dim)
            x_new = x_restart + 0.1 * v

            # Clip new candidate to bounds
            x_new = np.clip(x_new, lb, ub)

            f_new = func(x_new)

            if f_new < f_opt_restart:
                f_opt_restart = f_new

        # Update the global best solution
        if f_opt_restart < f_opt:
            f_opt = f_opt_restart
            x_opt = x_restart

    return f_opt, x_opt
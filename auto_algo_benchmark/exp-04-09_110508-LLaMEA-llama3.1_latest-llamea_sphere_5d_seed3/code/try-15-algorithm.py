import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=dim)
    f_opt = np.inf
    x_opt = None

    for i in range(budget // 10): # restart every 10 iterations
        x_restart = rng.uniform(lb, ub, size=dim)

        for j in range(9):
            v = rng.normal(size=dim)
            x_new = x + 0.1 * v

            # Clip new candidate to bounds
            x_new = np.clip(x_new, lb, ub)

            f_new = func(x_new)

            if f_new < f_opt:
                f_opt = f_new
                x_opt = x_new

            # Probability of accepting worse solution (adaptively adjusted)
            alpha = 0.05 / np.sqrt(i + 2)  # reduced acceptance probability
            beta = np.exp(-f_new * (i + 2) / budget)

            if rng.uniform(0, 1) < np.exp(alpha * f_opt - f_new + beta):
                x = x_new

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
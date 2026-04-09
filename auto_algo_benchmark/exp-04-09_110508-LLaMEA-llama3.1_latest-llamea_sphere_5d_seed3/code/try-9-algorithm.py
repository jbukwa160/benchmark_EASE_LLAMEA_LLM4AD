import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=dim)
    f_opt = np.inf
    x_opt = None
    step_size = 0.1
    accepted_count = 0
    rejected_count = 0
    max_accepted_count = int(budget * 0.5)

    for i in range(budget - 1):
        v = rng.normal(size=dim)
        x_new = x + step_size * v

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
            accepted_count += 1
            rejected_count = 0
        else:
            rejected_count += 1

        # Update step-size based on ratio of accepted to total candidates
        if accepted_count > max_accepted_count or (i == budget - 2 and accepted_count >= max_accepted_count):
            step_size *= np.exp(accepted_count / (max_accepted_count + 1))
            accepted_count = 0
            rejected_count = 0

    return f_opt, x_opt
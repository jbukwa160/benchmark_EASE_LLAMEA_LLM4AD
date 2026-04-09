import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    pop_size = 50
    n_generations = int(budget / (pop_size * dim))
    sigma_init = 0.5
    sigma_mult = 1.2
    cr = 0.9
    f = np.inf
    x_opt = None

    for _ in range(n_generations):
        pop = rng.uniform(lb, ub, (pop_size, dim))

        for i in range(pop_size):
            u, v, w = rng.choice(range(pop_size), size=3, replace=False)
            while u == i or v == i or w == i:
                u, v, w = rng.choice(range(pop_size), size=3, replace=False)

            x_new = pop[u] + sigma_mult ** (_ / n_generations) * (pop[v] - 2 * pop[w]) + cr * (np.random.uniform(lb, ub, dim) - lb)
            x_new = np.clip(x_new, lb, ub)

            f_new = func(x_new)

            if f_new < f:
                f = f_new
                x_opt = x_new

    return f, x_opt
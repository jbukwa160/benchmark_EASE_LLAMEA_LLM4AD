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

            x_new = pop[u] + sigma_mult ** (_ / n_generations) * (pop[v] - 2 * pop[w])
            x_mut1 = np.clip(x_new + cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
            x_mut2 = np.clip(x_new - cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)

            f_new_mut1 = func(x_mut1)
            f_new_mut2 = func(x_mut2)

            if f_new_mut1 < f:
                f = f_new_mut1
                x_opt = x_mut1

            if f_new_mut2 < f:
                f = f_new_mut2
                x_opt = x_mut2

    return f, x_opt
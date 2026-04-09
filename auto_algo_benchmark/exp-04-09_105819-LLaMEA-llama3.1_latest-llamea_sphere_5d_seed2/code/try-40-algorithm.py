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
            u, v = rng.choice(range(pop_size), size=2, replace=False)
            while u == i or v == i:
                u, v = rng.choice(range(pop_size), size=2, replace=False)

            p_mut = 0.5 + (ub - lb) / (np.abs(func(pop[i])) ** 0.1)
            if rng.random() < p_mut:
                x_mut = np.clip(pop[i] + cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
                f_new = func(x_mut)
                if f_new < f:
                    f = f_new
                    x_opt = x_mut

            x_new1 = pop[u] + sigma_mult ** (_ / n_generations) * (pop[v] - 2 * pop[i])
            x_crossover1 = np.clip(0.5 * (x_new1 + pop[i]), lb, ub)
            f_new1 = func(x_crossover1)

            if f_new1 < f:
                f = f_new1
                x_opt = x_crossover1

    return f, x_opt
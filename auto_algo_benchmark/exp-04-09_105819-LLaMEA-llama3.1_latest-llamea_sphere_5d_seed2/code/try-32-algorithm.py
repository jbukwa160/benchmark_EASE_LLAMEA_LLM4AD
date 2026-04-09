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

            x_new1 = pop[u] + sigma_mult ** (_ / n_generations) * (pop[v] - 2 * pop[w])
            x_mut1 = np.clip(x_new1 + cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
            x_crossover1 = 0.5 * (x_mut1 + pop[i])

            x_new2 = pop[u] + sigma_mult ** (_ / n_generations) * (pop[v] - 2 * pop[w])
            x_mut2 = np.clip(x_new2 - cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
            x_crossover2 = 0.5 * (x_mut2 + pop[i])

            f_new1 = func(x_crossover1)
            f_new2 = func(x_crossover2)

            if f_new1 < f:
                f = f_new1
                x_opt = x_crossover1

            if f_new2 < f:
                f = f_new2
                x_opt = x_crossover2

    # Adaptive mutation strategy inspired by the "better" mutation strategy
    sigma_adap = 0.9 * sigma_init + (f / np.mean([func(x) for x in pop])) ** 2
    for i in range(pop_size):
        u, v = rng.choice(range(pop_size), size=2, replace=False)
        while u == i or v == i:
            u, v = rng.choice(range(pop_size), size=2, replace=False)

        x_mut3 = np.clip(pop[u] + sigma_adap * (pop[v] - pop[i]), lb, ub)
        f_new3 = func(x_mut3)

        if f_new3 < f:
            f = f_new3
            x_opt = x_mut3

    return f, x_opt
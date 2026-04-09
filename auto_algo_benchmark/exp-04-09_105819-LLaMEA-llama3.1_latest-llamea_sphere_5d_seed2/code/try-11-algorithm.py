import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    pop_size = 50
    n_generations = int(budget / (pop_size * dim))
    sigma_init = 0.5
    sigma_mult = 1.2
    cr = 0.9
    cp = 0.5  # crossover probability

    for _ in range(n_generations):
        pop = rng.uniform(lb, ub, (pop_size, dim))

        for i in range(pop_size):
            u, v, w = rng.choice(range(pop_size), size=3, replace=False)
            while u == i or v == i or w == i:
                u, v, w = rng.choice(range(pop_size), size=3, replace=False)

            # Adapt crossover probability based on current generation
            cp_adapted = min(max(cp * (1 - _ / n_generations), 0.01), 0.99)

            x_new = pop[i] + sigma_mult ** (_ / n_generations) * (pop[u] - 2 * pop[v]) + cp_adapted * (np.random.uniform(lb, ub, dim) - lb)
            x_new = np.clip(x_new, lb, ub)

            f_new = func(x_new)

            if f_new < func(pop[i]):
                pop[i] = x_new

    return min(func(pop), key=func), np.array([min(func(pop), key=func)])
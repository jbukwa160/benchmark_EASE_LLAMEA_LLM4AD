import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    pop_size = 50
    n_generations = int(budget / (pop_size * dim))
    sigma_init = 0.5
    cr = 0.9
    f = np.inf
    x_opt = None

    for _ in range(n_generations):
        pop = rng.uniform(lb, ub, (pop_size, dim))

        for i in range(pop_size):
            u, v, w, z = rng.choice(range(pop_size), size=4, replace=False)
            while u == i or v == i or w == i or z == i:
                u, v, w, z = rng.choice(range(pop_size), size=4, replace=False)

            sigma = sigma_init * (1 - _ / n_generations)
            x_new = pop[u] + 0.5 * (pop[v] - pop[w]) + 0.5 * (pop[z] - pop[i])
            x_new = np.clip(x_new, lb, ub)

            f_new = func(x_new)

            if f_new < f:
                f = f_new
                x_opt = x_new

    return f, x_opt
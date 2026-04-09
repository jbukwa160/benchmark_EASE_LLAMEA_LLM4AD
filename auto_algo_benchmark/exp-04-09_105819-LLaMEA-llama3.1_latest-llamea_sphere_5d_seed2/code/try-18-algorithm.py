import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    pop_size = 100
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

            x_new1 = pop[i] + sigma_mult ** (_ / n_generations) * (pop[u] - 2 * pop[v]) 
            for j in range(dim):
                if rng.uniform(0, 1) < cr:
                    x_new1[j] += np.random.uniform(-ub, ub)
            x_new1 = np.clip(x_new1, lb, ub)
            f_new1 = func(x_new1)

            x_new2 = pop[i] + sigma_mult ** (_ / n_generations) * (pop[v] - 2 * pop[w]) 
            for j in range(dim):
                if rng.uniform(0, 1) < cr:
                    x_new2[j] += np.random.uniform(-ub, ub)
            x_new2 = np.clip(x_new2, lb, ub)
            f_new2 = func(x_new2)

            x_new3 = pop[i] + sigma_mult ** (_ / n_generations) * (pop[u] + pop[v]) 
            for j in range(dim):
                if rng.uniform(0, 1) < cr:
                    x_new3[j] += np.random.uniform(-ub, ub)
            x_new3 = np.clip(x_new3, lb, ub)
            f_new3 = func(x_new3)

            if f_new1 < f:
                f = f_new1
                x_opt = x_new1
            elif f_new2 < f:
                f = f_new2
                x_opt = x_new2
            elif f_new3 < f:
                f = f_new3
                x_opt = x_new3

    return f, x_opt
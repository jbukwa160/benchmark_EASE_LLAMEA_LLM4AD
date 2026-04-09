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

            # Introduce a mutation strategy inspired by the "better" mutation
            x_mut1 = pop[u]
            for j in range(dim):
                if rng.uniform(0, 1) < cr:
                    x_mut1[j] += np.random.uniform(-ub, ub)
            x_mut1 = np.clip(x_mut1, lb, ub)
            f_mut1 = func(x_mut1)

            # Adaptive crossover operator
            for j in range(dim):
                if rng.uniform(0, 1) < cr:
                    x_new1[j] += (x_mut1[j] - pop[i][j]) * np.random.uniform(-ub, ub)
                    x_new2[j] += (x_mut1[j] - pop[i][j]) * np.random.uniform(-ub, ub)

            # Introduce a selection mechanism to favor better solutions
            f_best = min(f_new1, f_new2, f_mut1, f_new3)
            if f_best < f:
                f = f_best
                x_opt = x_new1 if f_new1 == f_best else (x_new2 if f_new2 == f_best else (x_mut1 if f_mut1 == f_best else x_new3))
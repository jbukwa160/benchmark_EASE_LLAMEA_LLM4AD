import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    pop_size = 50
    n_generations = int(budget / (pop_size * dim))
    sigma_init = 0.5
    sigma_mult = 1.2
    cr = 0.9
    f = np.inf
    x_opt = None
    scores = []

    for _ in range(n_generations):
        pop = rng.uniform(lb, ub, (pop_size, dim))
        score_pop = [func(individual) for individual in pop]

        for i in range(pop_size):
            u, v, w = rng.choice(range(pop_size), size=3, replace=False)
            while u == i or v == i or w == i:
                u, v, w = rng.choice(range(pop_size), size=3, replace=False)

            sigma = sigma_init * (sigma_mult ** (_ / n_generations))
            x_new1 = pop[u] + sigma * (pop[v] - 2 * pop[w])
            x_mut1 = np.clip(x_new1 + cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
            x_crossover1 = 0.5 * (x_mut1 + pop[i])

            x_new2 = pop[u] + sigma * (pop[v] - 2 * pop[w])
            x_mut2 = np.clip(x_new2 - cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
            x_crossover2 = 0.5 * (x_mut2 + pop[i])

            f_new1 = func(x_crossover1)
            f_new2 = func(x_crossover2)

            if f_new1 < score_pop[i]:
                score_pop[i] = f_new1
                scores.append((i, f_new1))
            if f_new2 < score_pop[i]:
                score_pop[i] = f_new2
                scores.append((i, f_new2))

        scores.sort(key=lambda x: x[1])
        sigma_adaptive = 0.5 * (sigma_init + sigma_mult ** (_ / n_generations)) + 0.5 * (sigma_init - sigma_mult ** (_ / n_generations)) * np.random.uniform(0, 1)
        sigma = sigma_adaptive

    scores.sort(key=lambda x: x[1])
    f_opt = func(pop[scores[-1][0]])
    x_opt = pop[scores[-1][0]]

    return f_opt, x_opt
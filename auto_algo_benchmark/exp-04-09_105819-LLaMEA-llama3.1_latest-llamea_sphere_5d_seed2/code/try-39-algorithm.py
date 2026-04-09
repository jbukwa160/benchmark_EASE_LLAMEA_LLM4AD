import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    pop_size = 50
    n_generations = int(budget / (pop_size * dim))
    sigma_init = 0.5
    sigma_mult = 1.2
    cr = 0.9
    f_best = np.inf
    x_opt_best = None

    for _ in range(n_generations):
        pop = rng.uniform(lb, ub, (pop_size, dim))

        scores = [func(x) for x in pop]
        indices = np.argsort(scores)
        best_indices = indices[:int(pop_size * 0.2)]
        worst_indices = indices[-int(pop_size * 0.2):]

        sigma = sigma_init * (sigma_mult ** (_ / n_generations))

        for i in range(pop_size):
            if i not in best_indices and i not in worst_indices:
                u, v, w = rng.choice(best_indices, size=3, replace=False)
                while u == i or v == i or w == i:
                    u, v, w = rng.choice(best_indices, size=3, replace=False)

                x_new1 = pop[u] + sigma * (pop[v] - 2 * pop[w])
                x_mut1 = np.clip(x_new1 + cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
                x_crossover1 = 0.5 * (x_mut1 + pop[i])

                f_new1 = func(x_crossover1)

                if f_new1 < scores[i]:
                    pop[i] = x_crossover1
                    scores[i] = f_new1

            u, v, w = rng.choice(worst_indices, size=3, replace=False)
            while u == i or v == i or w == i:
                u, v, w = rng.choice(worst_indices, size=3, replace=False)

            x_new2 = pop[u] + sigma * (pop[v] - 2 * pop[w])
            x_mut2 = np.clip(x_new2 - cr * (np.random.uniform(lb, ub, dim) - lb), lb, ub)
            x_crossover2 = 0.5 * (x_mut2 + pop[i])

            f_new2 = func(x_crossover2)

            if f_new2 < scores[i]:
                pop[i] = x_crossover2
                scores[i] = f_new2

        if np.min(scores) < f_best:
            f_best = np.min(scores)
            x_opt_best = pop[np.argmin(scores)]

    return f_best, x_opt_best
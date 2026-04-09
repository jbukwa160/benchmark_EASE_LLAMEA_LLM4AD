import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    pop_size = 1000
    mutation_prob = 0.01
    elite_size = int(pop_size * 0.2)
    sigma_init = 5.0
    sigma_decay = 1e-6
    cr = 0.5

    pop = np.random.uniform(lb, ub, (pop_size, dim))
    target = func(np.zeros(dim))

    for i in range(budget):
        fit = np.array([func(x) for x in pop])
        idx = np.argsort(fit)
        pop[:elite_size] = pop[idx[:elite_size]]
        sigma = max(sigma_init * 0.99999 ** i - sigma_decay, 1e-6)

        for j in range(elite_size, pop_size):
            v = rng.normal(0, sigma, dim)
            while np.any(pop[j] + v < lb) or np.any(pop[j] + v > ub):
                v = rng.normal(0, sigma, dim)
            pop[j] += v

        for j in range(elite_size, pop_size):
            if rng.random() < mutation_prob:
                k = rng.randint(0, dim - 1)
                v = rng.uniform(-sigma, sigma) + pop[np.random.choice(range(elite_size), p=np.exp(-fit[:elite_size] / target))][k]
                while np.any(pop[j][k] + v < lb[k]) or np.any(pop[j][k] + v > ub[k]):
                    v = rng.uniform(-sigma, sigma)
                pop[j, k] += v

        if rng.random() < cr:
            for j in range(elite_size, pop_size):
                idx1 = np.random.choice(range(elite_size))
                idx2 = np.random.choice(range(elite_size))
                idx3 = np.random.choice(range(elite_size))
                child = 0.5 * (pop[idx1] + pop[idx2]) + rng.normal(0, sigma, dim)
                while np.any(child < lb) or np.any(child > ub):
                    child = 0.5 * (pop[idx1] + pop[idx2]) + rng.normal(0, sigma, dim)
                pop[j] = child

        if fit[idx[-1]] < target:
            best_x = pop[idx[-1]]
            best_f = fit[idx[-1]]
            target = best_f

    return best_f, best_x
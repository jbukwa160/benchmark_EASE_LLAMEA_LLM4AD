import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=dim)
    f_opt = np.inf
    x_opt = None

    for i in range(budget // 10): # restart every 10 iterations
        x_restart = rng.uniform(lb, ub, size=dim)

        population_size = 50 # set a moderate population size
        population = [rng.normal(size=dim) + x_restart for _ in range(population_size)]

        for j in range(9):
            v1 = rng.normal(size=dim)
            v2 = rng.normal(size=dim)
            v3 = rng.normal(size=dim)

            # Calculate the offspring candidates using the DE/rand/1/bin mutation strategy
            x_new1 = x_restart + 0.5 * (v1 - x_restart) + 0.5 * (v2 - x_restart)
            x_new2 = x_restart + 0.5 * (v1 - x_restart) + 0.3 * (v2 - x_restart) + 0.4 * (v3 - x_restart)

            # Clip new candidates to bounds
            x_new1 = np.clip(x_new1, lb, ub)
            x_new2 = np.clip(x_new2, lb, ub)

            f_new1 = func(x_new1)
            f_new2 = func(x_new2)

            population.append((x_new1, f_new1))
            population.append((x_new2, f_new2))

        # Sort the population by fitness
        population.sort(key=lambda x: x[1])

        # Remove duplicates and update the best solution found in this restart
        unique_population = [x for i, x in enumerate(population) if all(np.not_equal(x, y)) or (i == 0 and np.equal(x, y))]
        f_opt_restart = min(unique_population, key=lambda x: x[1])[1]
        x_opt_restart = min(unique_population, key=lambda x: x[1])[0]

        # Update the global best solution
        if f_opt_restart < f_opt:
            f_opt = f_opt_restart
            x_opt = x_opt_restart

    return f_opt, x_opt
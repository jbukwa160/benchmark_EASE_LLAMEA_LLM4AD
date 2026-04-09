import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    def differential_evolution(x):
        trial_x = x + 0.5 * (np.random.uniform(-1, 1, size=dim) * x)
        return func(trial_x)

    stagnation_period = 10
    restart_freq = 3

    x = rng.uniform(lb, ub, size=(budget, dim))
    f = [func(x[i]) for i in range(budget)]

    best_idx = np.argmin(f)
    
    stagnation_count = 0
    
    for _ in range(100): # maximum number of generations
        new_best_idx = 0
        for i in range(budget):
            idx2 = np.random.choice([j for j in range(budget) if j != best_idx], 3, replace=False)
            trial_x = x[new_best_idx].copy() + 0.5 * (x[idx2] - x[best_idx])
            trial_f = differential_evolution(trial_x)
            
            if trial_f < f[best_idx]:
                stagnation_count = 0
                new_best_idx = i
                x[i] = trial_x
                f[i] = trial_f
        
        best_idx = np.argmin(f)

        # adaptive restart mechanism
        if stagnation_count >= stagnation_period and _ % restart_freq == 0:
            x = rng.uniform(lb, ub, size=(budget, dim))
            f = [func(x[i]) for i in range(budget)]
            stagnation_count = 0
        
        stagnation_count += 1
    
    return f[best_idx], x[best_idx]
import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    def differential_evolution(x):
        trial_x = x + 0.5 * (np.random.uniform(-1, 1, size=dim) * x)
        return func(trial_x)

    x = rng.uniform(lb, ub, size=(budget, dim))
    f = [func(x[i]) for i in range(budget)]

    best_idx = np.argmin(f)
    
    stagnation = 0
    max_stagnation = budget // 2
    
    for _ in range(10): # number of restarts
        new_best_idx = 0
        for i in range(budget):
            idx2 = np.random.choice([j for j in range(budget) if j != best_idx], 3, replace=False)
            trial_x = x[new_best_idx].copy() + 0.5 * (x[idx2] - x[best_idx])
            trial_f = differential_evolution(trial_x)
            
            if trial_f < f[best_idx]:
                stagnation = 0
                new_best_idx = i
                x[i] = trial_x
                f[i] = trial_f
        
        if np.argmin(f) == best_idx:
            stagnation += 1
        
        if stagnation >= max_stagnation:
            x[rng.choice(budget, size=budget)] = rng.uniform(lb, ub, size=(budget, dim))
            f = [func(x[i]) for i in range(budget)]
            stagnation = 0
            best_idx = np.argmin(f)
        
        best_idx = np.argmin(f)

    return x[np.argmin(f)], x[np.argmin(f)]
import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    def differential_evolution(x):
        trial_x = x + 0.5 * (np.random.uniform(-1, 1, size=dim) * x)
        return func(trial_x)

    max_stagnation = budget // 2
    stagnation_count = [0] * budget

    x = rng.uniform(lb, ub, size=(budget, dim))
    f = [func(x[i]) for i in range(budget)]

    best_idx = np.argmin(f)
    
    for _ in range(10): # number of restarts
        new_best_idx = 0
        for i in range(budget):
            idx2 = np.random.choice([j for j in range(budget) if j != best_idx], 3, replace=False)
            trial_x = x[new_best_idx].copy() + 0.5 * (x[idx2] - x[best_idx])
            trial_f = differential_evolution(trial_x)
            
            if trial_f < f[best_idx]:
                stagnation_count[best_idx] = 0
                new_best_idx = i
                x[i] = trial_x
                f[i] = trial_f
                stagnation_count[i] = 1
                
            elif np.random.rand() < 0.2:
                stagnation_count[new_best_idx] += 1
            
        for i in range(budget):
            if stagnation_count[i] > max_stagnation:
                x[i] = rng.uniform(lb, ub, size=dim)
                f[i] = func(x[i])
                stagnation_count[i] = 0
        
        best_idx = np.argmin(f)

    return f[best_idx], x[best_idx]
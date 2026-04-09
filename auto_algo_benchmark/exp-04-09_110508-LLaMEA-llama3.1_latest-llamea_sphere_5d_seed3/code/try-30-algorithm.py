import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    def differential_evolution(x):
        trial_x = x + 0.5 * (np.random.uniform(-1, 1, size=dim) * x)
        return func(trial_x)

    x = rng.uniform(lb, ub, size=(budget*10, dim))
    f = [func(x[i]) for i in range(budget*10)]

    best_idx = np.argmin(f)
    
    stagnation_count = 0
    for _ in range(100): # number of generations
        new_best_idx = 0
        for i in range(budget):
            idx2 = np.random.choice([j for j in range(int(budget*10)) if j != best_idx], 3, replace=False)
            trial_x = x[new_best_idx].copy() + 0.5 * (x[idx2] - x[best_idx])
            trial_f = differential_evolution(trial_x)
            
            if trial_f < f[best_idx]:
                new_best_idx = i
                x[i*10+budget] = trial_x
                f[i*10+budget] = trial_f
        
        # Check for stagnation
        if np.mean(np.array(f)[int(budget):]) > np.mean(np.array(f)[:int(budget)]):
            stagnation_count += 1
            if stagnation_count >= budget//2:
                x = rng.uniform(lb, ub, size=(budget*10, dim))
                f = [func(x[i]) for i in range(budget*10)]
                best_idx = np.argmin(f)
                stagnation_count = 0
        
        else:
            stagnation_count = 0
            best_idx = np.argmin(np.array(f)[int(budget):])
        
        # Store best solution found so far
        if f[best_idx] < f[int(budget)]:
            x[int(budget*9)] = x[best_idx]
            f[int(budget*9)] = f[best_idx]

    return f[budget], x[budget]
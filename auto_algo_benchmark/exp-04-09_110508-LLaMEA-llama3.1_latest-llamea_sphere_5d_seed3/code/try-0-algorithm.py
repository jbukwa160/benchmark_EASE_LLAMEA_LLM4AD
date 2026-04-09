import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=(budget, dim))
    f = [func(x[i]) for i in range(budget)]
    
    best_idx = 0
    for i in range(1, budget):
        idx2 = np.random.choice([j for j in range(i) if j != best_idx], 3, replace=False)
        idx3 = np.random.choice([k for k in range(budget) if k not in [best_idx] + idx2], 1)[0]
        
        trial_x = x[idx2].mean(axis=0) + 0.5 * (x[idx3] - x[best_idx])
        trial_f = func(trial_x)
        
        if trial_f < f[best_idx]:
            best_idx = i
            x[best_idx] = trial_x
            f[best_idx] = trial_f
    
    return f[best_idx], x[best_idx]
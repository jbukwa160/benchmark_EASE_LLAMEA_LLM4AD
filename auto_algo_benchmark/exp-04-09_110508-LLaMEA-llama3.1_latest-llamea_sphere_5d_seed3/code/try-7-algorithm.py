import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=(budget, dim))
    f = [func(x[i]) for i in range(budget)]
    
    best_idx = 0
    F = 1.0 # initial mutation factor
    CR = 0.5 # initial crossover rate
    
    for _ in range(budget):
        idx2 = np.random.choice([j for j in range(len(f)) if j != best_idx], 3, replace=False)
        idx3 = np.random.choice([k for k in range(len(f)) if k not in [best_idx] + idx2], 1)[0]
        
        trial_x = x[idx2].mean(axis=0) + F * (x[idx3] - x[best_idx])
        trial_f = func(trial_x)
        
        if np.random.rand() < CR or trial_f < f[best_idx]:
            best_idx = len(f)
            x[best_idx] = trial_x
            f.append(trial_f)
        else:
            F *= 0.9 # decrease mutation factor
            
    return min(f), x[np.argmin(f)]
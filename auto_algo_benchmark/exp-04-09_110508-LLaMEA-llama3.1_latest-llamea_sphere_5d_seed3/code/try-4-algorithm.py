import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=(budget, dim))
    f = [func(x[i]) for i in range(budget)]
    
    best_idx = 0
    for _ in range(int(0.2 * budget)):
        idx2 = np.random.choice([j for j in range(len(f)) if j != best_idx], 3, replace=False)
        idx3 = np.random.choice([k for k in range(len(f)) if k not in [best_idx] + idx2], 1)[0]
        
        mutation_x = x[best_idx] + 0.5 * (x[idx3] - x[best_idx]) + rng.uniform(-1, 1, size=dim)
        trial_f = func(mutation_x)
        
        if trial_f < f[best_idx]:
            best_idx = len(f)
            x[best_idx] = mutation_x
            f.append(trial_f)
    
    return np.min(np.array(f)), np.argmin(np.array(f))
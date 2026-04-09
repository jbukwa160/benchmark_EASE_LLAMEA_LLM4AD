import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x = rng.uniform(lb, ub, size=(budget, dim))
    f = [func(x[i]) for i in range(budget)]
    
    best_idx = 0
    adapt_step_size = 1.0
    
    for _ in range(100):  # Adapt step size every 100 iterations
        for i in range(1, budget):
            idx2 = np.random.choice([j for j in range(i) if j != best_idx], 3, replace=False)
            idx3 = np.random.choice([k for k in range(budget) if k not in [best_idx] + idx2], 1)[0]
            
            trial_x = x[idx2].mean(axis=0) + adapt_step_size * (x[idx3] - x[best_idx])
            trial_f = func(trial_x)
            
            if trial_f < f[best_idx]:
                best_idx = i
                x[best_idx] = trial_x
                f[best_idx] = trial_f
            
            # Adapt step size based on success rate of last 10 individuals
            adapt_step_size *= 1.001 if np.mean([f[j] for j in range(budget) if j != best_idx and f[j] < f[best_idx]]) > 0.5 else 0.99
        
        x = rng.uniform(lb, ub, size=(budget, dim))
        f = [func(x[i]) for i in range(budget)]
    
    return f[best_idx], x[best_idx]
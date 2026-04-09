import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = float('inf')
    
    cov = np.eye(dim) * 0.5
    for i in range(budget):
        dist_x = np.random.normal(x_opt, np.sqrt(np.diag(cov)), (1000, dim))
        weights = np.exp(-func(dist_x).mean())
        weights /= weights.sum()
        
        # Importance sampling to focus on promising regions of search space
        x_cand = rng.choice(dim, p=weights)
        x_cand = dist_x[np.arange(1000), np.arange(dim), x_cand].reshape((dim,))
        x_cand = np.clip(x_cand, lb, ub)
        
        f = func(x_cand)
        if f < f_opt:
            f_opt = f
            x_opt = x_cand
            
        delta_x = x_cand - x_opt
        cov = (1 - 0.5 / (i + 2)) * cov + (0.5 / (i + 2)) * delta_x * delta_x.T
        
    return x_opt
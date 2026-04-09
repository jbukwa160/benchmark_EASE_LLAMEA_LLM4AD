import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = float('inf')
    
    cov = np.eye(dim) * 0.5
    for i in range(budget):
        x_cand = rng.multivariate_normal(x_opt, cov, 1)[0]
        
        # Clip candidates to [lb, ub]
        x_cand = np.clip(x_cand, lb, ub)
        
        f = func(x_cand)
        if f < f_opt:
            f_opt = f
            x_opt = x_cand
            
        delta_x = x_cand - x_opt
        cov = (1 - 0.5 / (i + 2)) * cov + (0.5 / (i + 2)) * delta_x * delta_x.T
        
    return x_opt
import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = float('inf')
    
    # Use historical samples (only one available) to initialize the covariance matrix
    cov = np.eye(dim) * 0.5 + np.outer(
        [1, -2, 4, -8, 16], 
        [1, -2, 4, -8, 16]
    )
    
    for i in range(budget):
        x_cand = rng.multivariate_normal(x_opt, cov, 1)[0]
        
        # Clip candidates to [lb, ub]
        x_cand = np.clip(x_cand, lb, ub)
        
        f = func(x_cand)
        if f < f_opt:
            f_opt = f
            x_opt = x_cand
            
        delta_x = x_cand - x_opt
        cov += delta_x * delta_x.T
        
    return x_opt
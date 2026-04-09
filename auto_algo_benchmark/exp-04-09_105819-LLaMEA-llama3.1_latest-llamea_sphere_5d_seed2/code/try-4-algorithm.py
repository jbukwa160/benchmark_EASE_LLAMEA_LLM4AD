import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.array([0.]*dim)
    f_opt = float('inf')
    
    X = np.zeros((budget, dim))
    Y = np.zeros(budget)

    for i in range(1, budget+1):
        if i < 10: # burn-in phase
            X[i-1] = np.random.uniform(lb, ub, size=dim)
        else:
            mu = np.mean(Y[:i])
            sigma = np.std(Y[:i])
            cov = np.cov(X[:i-1].T) # re-estimate the covariance matrix at each iteration
            inv_cov = np.linalg.inv(cov)
            K = np.zeros((dim,))
            for j in range(dim):
                K[j] = cov @ (X[i-2,j]**2 + sigma**2 - X[:i-1,j]**2)
            
            alpha = inv_cov @ cov / (np.eye(i-1) + inv_cov @ cov)
            mu_prime = np.mean(Y[:i]) + K.T @ Y[:i] @ alpha
            f_opt_prime = mu_prime - 0.5 * np.dot(K, np.dot(alpha, K))
            
            x_new = rng.normal(x_opt, 0.1) # sample around the current optimum
            
            Y[i-1] = func(np.clip(x_new, lb, ub)) # evaluate the new candidate
            if Y[i-1] < f_opt:
                f_opt = Y[i-1]
                x_opt = np.array(x_new)

    return f_opt, x_opt
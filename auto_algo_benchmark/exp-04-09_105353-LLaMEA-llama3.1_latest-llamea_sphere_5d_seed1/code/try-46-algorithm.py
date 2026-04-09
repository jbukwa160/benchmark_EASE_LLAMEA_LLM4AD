import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_matrix = np.eye(dim) * (ub - lb)**2 / 4
    for i in range(budget):
        mean = x_opt + (ub - lb) / 2

        # Importance sampling: focus on promising regions with higher probability
        dist = np.linalg.cholesky(cov_matrix)
        sample = mean + np.dot(dist, rng.standard_normal((dim,)))
        importance = np.exp(-func(sample))
        prob = importance / (importance.sum() + 1e-8)

        # Draw a new point based on the importance sampling distribution
        idx = np.random.choice(dim, p=prob)
        sample[idx] += np.sqrt(np.log(budget / i) / (dim - 1)) * rng.standard_normal()

        x_clipped = np.clip(sample, lb, ub)

        f_new = func(x_clipped)
        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

    return x_opt
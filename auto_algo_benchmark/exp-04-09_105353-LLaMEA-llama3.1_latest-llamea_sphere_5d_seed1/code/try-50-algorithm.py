import numpy as np

def algorithm(func, dim, lb, ub, budget, rng):
    x_opt = np.zeros(dim)
    f_opt = func(x_opt)

    cov_mat = np.eye(dim) * 0.1
    importance_sampling_ratio = 0.01
    trust_radius = 1.0  # Initial trust region radius

    for i in range(budget):
        if rng.random() < importance_sampling_ratio:
            x_new = np.zeros(dim)
            f_new = func(x_new)

        else:
            # Sample from a normal distribution with the current covariance matrix
            x_new = np.random.multivariate_normal(x_opt, cov_mat)

        # Clip candidates to [lb, ub]
        x_clipped = np.clip(x_new, lb, ub)

        # Evaluate the objective function at the new point
        f_new = func(x_clipped)

        if f_new < f_opt:
            x_opt = x_clipped
            f_opt = f_new

            # Update the covariance matrix using importance sampling ratio
            cov_mat += importance_sampling_ratio * (x_clipped - x_opt) * (x_clipped - x_opt).T

        # Adapt trust region radius based on improvement and current radius
        if f_new < f_opt / 2:
            trust_radius *= 0.9  # Reduce radius for more exploration
        elif f_new > f_opt * 1.5:
            trust_radius *= 1.1  # Increase radius for more exploitation

        # Apply trust region constraint to new sample
        if np.linalg.norm(x_clipped - x_opt) > trust_radius:
            continue  # Reject candidate outside trust region

    return x_opt
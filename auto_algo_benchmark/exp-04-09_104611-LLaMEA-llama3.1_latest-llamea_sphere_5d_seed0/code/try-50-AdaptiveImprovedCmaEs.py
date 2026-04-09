import numpy as np

class AdaptiveImprovedCmaEs:
    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.mu = 0.5 ** (1 / dim) * np.random.uniform(-1, 1, size=(dim,))
        self.sigma = 0.1
        self.cov = np.eye(dim)
        self.step_size = 0.01
        self.learning_rate = 0.995

    def __call__(self, func):
        for i in range(self.budget):
            if self.x_opt is not None:
                x = np.clip(self.x_opt + np.random.multivariate_normal(np.zeros(self.dim), self.sigma * (np.eye(self.dim) + self.cov)), -5.0, 5.0)
            else:
                x = np.clip(np.random.multivariate_normal(self.mu, self.sigma * (np.eye(self.dim) + self.cov)), -5.0, 5.0)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            # Adaptive step-size update with exponential learning rate schedule and improvement-based adaptation, adaptive covariance matrix diagonal and reusing the best solution found so far.
            improvement = (f - self.f_opt) / abs(f) if f != 0 else 1e-6
            self.step_size *= np.exp(min(self.learning_rate ** i * improvement, 1))
            self.cov = (1 - 2 / (self.budget + 1)) * self.cov + (1 / (self.budget + 1)) * np.outer((x - self.mu), (x - self.mu))
            # Adaptive covariance matrix diagonal
            self.cov[range(self.dim), range(self.dim)] = np.maximum(self.cov[range(self.dim), range(self.dim)], 0.01)
            # Reuse the best solution found so far to improve convergence speed.
            if f < self.f_opt:
                self.mu = (1 - 2 / (self.budget + 1)) * self.mu + (1 / (self.budget + 1)) * x

        return self.f_opt, self.x_opt

def algorithm(func, dim, lb, ub, budget, rng):
    es = AdaptiveImprovedCmaEs(budget=budget, dim=dim)
    return es(func)

# Example usage:
def sphere(x):
    return np.sum(np.square(x))

print(algorithm(sphere, 5, -5.0, 5.0, 10000, None))
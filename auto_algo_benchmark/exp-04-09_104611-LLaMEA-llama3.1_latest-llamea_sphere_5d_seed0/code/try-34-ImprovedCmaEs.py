import numpy as np

class ImprovedCmaEs:
    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.mu = 0.5 ** (1 / dim) * np.random.uniform(-1, 1, size=(dim,))
        self.sigma = 0.1
        self.cov = np.eye(dim)
        self.step_size = 0.01
        self.learning_rate = 0.5
        self.adaptive_diagonal = True

    def __call__(self, func):
        for i in range(self.budget):
            x = np.clip(np.random.multivariate_normal(self.mu, self.sigma * (np.eye(self.dim) + self.cov)), -5.0, 5.0)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

            # Adaptive step-size update for the covariance matrix and learning rate schedule
            self.step_size *= np.exp(1 - (f / max(self.f_opt, 1e-6))) * min(max(self.learning_rate ** i, 1e-3), 0.99)
            if self.adaptive_diagonal:
                self.cov = (1 - 2 / (self.budget + 1)) * self.cov + (1 / (self.budget + 1)) * np.outer((x - self.mu), (x - self.mu))
                self.cov = np.diag(np.clip(self.cov.diagonal(), a_min=1e-6, a_max=np.inf))

            # Reuse the best solution found so far
            if i > 0 and f < self.f_opt / (i + 1):
                self.mu = x

        return self.f_opt, self.x_opt

def algorithm(func, dim, lb, ub, budget, rng):
    es = ImprovedCmaEs(budget=budget, dim=dim)
    return es(func)

# Example usage:
def sphere(x):
    return np.sum(np.square(x))

print(algorithm(sphere, 5, -5.0, 5.0, 10000, None))
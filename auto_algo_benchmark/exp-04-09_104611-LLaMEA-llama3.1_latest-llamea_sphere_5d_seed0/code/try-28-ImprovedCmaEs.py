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
        self.learning_rate = 0.9

    def __call__(self, func):
        for i in range(self.budget):
            x = np.clip(np.random.multivariate_normal(self.mu, self.sigma * (np.eye(self.dim) + self.cov)), -5.0, 5.0)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            # Adaptive step-size update for the covariance matrix and learning rate schedule with adaptive learning rate
            adaptation_rate = (1 / (self.budget + 1)) * np.exp(-i / self.budget)
            self.step_size *= np.exp(1 - (f / self.f_opt) ** adaptation_rate) * min(self.learning_rate, 0.99 - i / 10000)  # added dynamic learning rate adaptation
            self.cov = (1 - 2 / (self.budget + 1)) * self.cov + (1 / (self.budget + 1)) * np.outer((x - self.mu), (x - self.mu))
            self.mu = self.mu + (1 / (self.budget + 1)) * np.dot(self.cov, (x - self.mu))

        return self.f_opt, self.x_opt

def algorithm(func, dim, lb, ub, budget, rng):
    es = ImprovedCmaEs(budget=budget, dim=dim)
    return es(func)

# Example usage:
def sphere(x):
    return np.sum(np.square(x))

print(algorithm(sphere, 5, -5.0, 5.0, 10000, None))
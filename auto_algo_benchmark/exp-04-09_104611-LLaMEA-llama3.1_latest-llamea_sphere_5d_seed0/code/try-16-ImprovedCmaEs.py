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
        self.lr_schedule = [1] * (self.budget // 2) + [0.1] * (self.budget // 4) + [0.01] * (self.budget // 10)

    def __call__(self, func):
        for i in range(self.budget):
            x = np.clip(np.random.multivariate_normal(self.mu, self.sigma * (np.eye(self.dim) + self.cov)), -5.0, 5.0)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            step_size = self.lr_schedule[i] * np.exp(-i / (self.budget // 2))
            self.cov = (1 - 2 / (self.budget + 1)) * self.cov + (1 / (self.budget + 1)) * np.outer((x - self.mu), (x - self.mu))
            self.mu = self.mu + step_size * np.dot(self.cov, (x - self.mu))

        return self.f_opt, self.x_opt

def algorithm(func, dim, lb, ub, budget, rng):
    es = ImprovedCmaEs(budget=budget, dim=dim)
    return es(func)

# Example usage:
def sphere(x):
    return np.sum(np.square(x))

print(algorithm(sphere, 5, -5.0, 5.0, 10000, None))
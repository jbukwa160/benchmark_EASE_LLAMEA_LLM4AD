import numpy as np

class ImprovedCmaEsDynamicPopulation:
    def __init__(self, budget=10000, dim=10):
        self.budget = int(budget)
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.mu = 0.5 ** (1 / dim) * np.random.uniform(-1, 1, size=(dim,))
        self.sigma = 0.1
        self.cov = np.eye(dim)
        self.step_size = 0.01
        self.learning_rate = 0.95
        self.population_size = 10

    def __call__(self, func):
        for i in range(self.budget):
            if i % (self.budget // 10) == 0:
                # Dynamic population size adjustment
                self.population_size = int(10 + (i / (self.budget // 10)) * 90)
            x = np.clip(np.random.multivariate_normal(self.mu, self.sigma * (np.eye(self.dim) + self.cov)), -5.0, 5.0)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            # Adaptive step-size update with learning rate schedule
            self.step_size *= np.exp((1 - (f / self.f_opt)) * self.learning_rate ** i)
            self.cov = (1 - 2 / (self.budget + 1)) * self.cov + (1 / (self.budget + 1)) * np.outer((x - self.mu), (x - self.mu))
            # Adaptive covariance matrix diagonal
            self.cov[range(self.dim), range(self.dim)] = np.maximum(self.cov[range(self.dim), range(self.dim)], 0.01)

        return self.f_opt, self.x_opt

def algorithm(func, dim, lb, ub, budget, rng):
    es = ImprovedCmaEsDynamicPopulation(budget=budget, dim=dim)
    return es(func)

# Example usage:
def sphere(x):
    return np.sum(np.square(x))

print(algorithm(sphere, 5, -5.0, 5.0, 10000, None))
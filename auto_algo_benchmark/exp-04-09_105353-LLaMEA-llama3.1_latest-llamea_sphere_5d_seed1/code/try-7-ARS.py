import numpy as np

class ARS:
    def __init__(self, dim, lb=-5.0, ub=5.0, budget=10000):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.budget = budget
        self.x_opt = None
        self.f_opt = np.inf

    def __call__(self, func):
        X = np.zeros((self.budget, self.dim))
        Y = np.zeros(self.budget)
        C = np.eye(self.dim)

        for i in range(1, self.budget+1):
            x_new = self.lb + (self.ub - self.lb) * np.random.normal(size=self.dim)
            for d in range(self.dim):
                if self.x_opt is not None:
                    x_new[d] = max(self.lb, min(x_new[d], self.ub))
                    C[d,d] += (x_new[d] - X[i-1,d])**2
                    x_new[d] += np.random.normal(0, 1/np.sqrt(C[d,d]))

            X[i-1] = x_new
            Y[i-1] = func(x_new)

            if i > 1:
                for d in range(self.dim):
                    C[d,d] /= (i-1)
                C += np.eye(self.dim) / (self.budget * self.dim)

            if Y[i-1] < self.f_opt:
                self.f_opt = Y[i-1]
                self.x_opt = X[i-1]

        return self.f_opt, self.x_opt
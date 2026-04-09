import numpy as np

class BO:
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

        # Select the initial point for the GP
        x_init = np.array([0.0] * self.dim)
        if self.x_opt is not None:
            x_init = self.x_opt

        mean_init, var_init = self.gp().predict(np.array([[x_init]]), full_cov=False)

        for i in range(1, self.budget+1):
            mean, var = self.gp.predict(np.array([[np.mean(X[:i-1], axis=0)]]), full_cov=False)
            std = np.sqrt(var)

            x_new = self.lb + (self.ub - self.lb) * np.random.beta(1, 1)
            for d in range(self.dim):
                if self.x_opt is not None:
                    x_new[d] = max(self.lb, min(x_new[d], self.ub))
                    x_new[d] += np.random.uniform(-std[d], std[d]) + (x_new[d] - mean_init[d])

            X[i-1] = x_new
            Y[i-1] = func(x_new)

            if i > 1:
                self.gp.fit(X[:i-1], Y[:i-1])
            if Y[i-1] < self.f_opt:
                self.f_opt = Y[i-1]
                self.x_opt = X[i-1]

        return self.f_opt, self.x_opt

    def gp(self):
        import numpy as np
        from scipy.stats import norm

        class GP:
            def __init__(self, mean_func=np.zeros(0)):
                self.mean_func = mean_func
                self.L = np.eye(self.dim)
                self.sigma_f = 1.0

            def predict(self, x_pred, full_cov=True):
                Kxx = np.linalg.multi_dot([np.diag(np.ones(self.dim)), self.L, np.diag(np.ones(self.dim))])
                Kxp = np.linalg.multi_dot([self.L, (x_pred - np.mean(X[:i-1], axis=0)).reshape(-1, 1), np.ones((1, self.dim))])

                var = self.sigma_f**2 * (1 + Kxx)
                mean = self.mean_func(x_pred) + Kxp / var
                return mean, var

            def fit(self, X, Y):
                L_inv = np.linalg.inv(np.eye(self.dim) + 1/(self.sigma_f**2) * np.dot(X.T, X))
                self.L = np.dot(L_inv, np.linalg.inv(np.eye(self.dim) - 1/(self.sigma_f**2) * np.dot(X.T, X)))
                self.sigma_f *= np.sqrt(1 / (1 + 1/self.sigma_f**2 * np.sum(np.diag(np.dot(X.T, X)))))

        return GP()

def algorithm(func, dim, lb, ub, budget, rng):
    bo = BO(dim, lb=lb, ub=ub, budget=budget)
    f_best, x_best = bo(func)
    return x_best
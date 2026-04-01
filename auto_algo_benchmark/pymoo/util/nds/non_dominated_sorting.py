from __future__ import annotations
import numpy as np

class NonDominatedSorting:
    def do(self, F, only_non_dominated_front=True):
        F = np.asarray(F, dtype=float)
        n = len(F)
        if n == 0:
            return []
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        front = [i for i in range(n) if not dominated[i]]
        return front if only_non_dominated_front else [front]

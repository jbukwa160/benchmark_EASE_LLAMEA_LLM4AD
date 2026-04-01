from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


PENALTY_BEST_F = 1e12


class OverBudgetError(RuntimeError):
    pass


@dataclass(slots=True)
class BenchmarkTask:
    name: str
    dim: int
    budget: int
    lower_bound: float
    upper_bound: float
    objective: Callable[[np.ndarray], float]
    eval_seeds: list[int]


class BudgetedObjective:
    def __init__(self, objective: Callable[[np.ndarray], float], dim: int, budget: int, lower_bound: float, upper_bound: float):
        self.objective = objective
        self.dim = int(dim)
        self.budget = int(budget)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.used_budget = 0
        self.best_x: np.ndarray | None = None
        self.best_f = float("inf")

    def __call__(self, x) -> float:
        if self.used_budget >= self.budget:
            raise OverBudgetError(f"Budget exceeded: {self.used_budget} >= {self.budget}")

        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"Expected a {self.dim}D point, got shape {arr.shape}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=self.upper_bound, neginf=self.lower_bound)
        arr = np.clip(arr, self.lower_bound, self.upper_bound)

        self.used_budget += 1
        value = float(self.objective(arr))
        if not math.isfinite(value):
            value = PENALTY_BEST_F
        if value < self.best_f:
            self.best_f = value
            self.best_x = arr.copy()
        return value


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def rastrigin(x: np.ndarray) -> float:
    n = x.size
    return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    if x.size < 2:
        return float((1.0 - x[0]) ** 2)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def mixed(x: np.ndarray) -> float:
    return float(0.5 * sphere(x) + 0.3 * rastrigin(x) + 0.2 * np.sum(np.abs(x)))


OBJECTIVES: dict[str, Callable[[np.ndarray], float]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "mixed": mixed,
}


def task_from_name(name: str, defaults: dict) -> BenchmarkTask:
    parts = name.split("_")
    if len(parts) < 2 or not parts[-1].endswith("d"):
        raise ValueError(f"Unsupported task name: {name}")
    dim = int(parts[-1][:-1])
    family = "_".join(parts[:-1])
    if family not in OBJECTIVES:
        raise ValueError(f"Unknown task family: {family}")
    return BenchmarkTask(
        name=name,
        dim=dim,
        budget=int(defaults.get("budget", 80)),
        lower_bound=float(defaults.get("lower_bound", -5.0)),
        upper_bound=float(defaults.get("upper_bound", 5.0)),
        objective=OBJECTIVES[family],
        eval_seeds=[int(x) for x in defaults.get("eval_seeds", [11, 17, 23])],
    )


def score_from_best_f(best_f: float) -> float:
    if best_f is None or not math.isfinite(best_f):
        return -12.0
    clipped = max(float(best_f), 1e-12)
    return float(-math.log10(clipped))

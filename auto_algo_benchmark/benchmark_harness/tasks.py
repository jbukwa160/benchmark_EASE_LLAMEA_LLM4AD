"""
tasks.py — Benchmark task definitions (sphere, rastrigin, rosenbrock in 5D).

Each task exposes:
  - name: str
  - dimension: int
  - lower_bound: float
  - upper_bound: float
  - evaluate(x: np.ndarray) -> float   (lower is better, so we negate for maximisation frameworks)
  - description: str  (natural-language for LLM prompt)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkTask:
    name: str
    dimension: int
    lower_bound: float
    upper_bound: float
    _fn: Callable = field(repr=False)
    description: str = ""

    def evaluate(self, x: np.ndarray) -> float:
        """Return raw objective value (to be minimised)."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"Expected shape ({self.dimension},), got {x.shape}")
        return float(self._fn(x))

    def score(self, x: np.ndarray) -> float:
        """Return negated value — higher is better (for frameworks that maximise)."""
        return -self.evaluate(x)


# ---------------------------------------------------------------------------
# Raw objective functions
# ---------------------------------------------------------------------------

def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def _rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, Callable[[int, float, float], BenchmarkTask]] = {
    "sphere_5d": lambda lb, ub: BenchmarkTask(
        name="sphere_5d",
        dimension=5,
        lower_bound=lb,
        upper_bound=ub,
        _fn=_sphere,
        description=(
            "Minimise the 5-dimensional Sphere function: f(x) = sum(x_i^2). "
            f"All variables are bounded in [{lb}, {ub}]. "
            "The global minimum is 0 at x = (0, 0, 0, 0, 0)."
        ),
    ),
    "rastrigin_5d": lambda lb, ub: BenchmarkTask(
        name="rastrigin_5d",
        dimension=5,
        lower_bound=lb,
        upper_bound=ub,
        _fn=_rastrigin,
        description=(
            "Minimise the 5-dimensional Rastrigin function: "
            "f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i)). "
            f"All variables are bounded in [{lb}, {ub}]. "
            "The global minimum is 0 at x = (0, 0, 0, 0, 0). "
            "This is a highly multimodal landscape."
        ),
    ),
    "rosenbrock_5d": lambda lb, ub: BenchmarkTask(
        name="rosenbrock_5d",
        dimension=5,
        lower_bound=lb,
        upper_bound=ub,
        _fn=_rosenbrock,
        description=(
            "Minimise the 5-dimensional Rosenbrock (banana) function: "
            "f(x) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2). "
            f"All variables are bounded in [{lb}, {ub}]. "
            "The global minimum is 0 at x = (1, 1, 1, 1, 1)."
        ),
    ),
}


def get_task(name: str, lower_bound: float = -5.0, upper_bound: float = 5.0) -> BenchmarkTask:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name](lower_bound, upper_bound)

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    objective: str
    dim: int
    budget: int
    lower_bound: float
    upper_bound: float
    eval_seeds: tuple[int, ...]


def builtin_task_specs(task_defaults: dict[str, Any]) -> dict[str, BenchmarkTask]:
    budget = int(task_defaults.get("budget", 80))
    lower_bound = float(task_defaults.get("lower_bound", -5.0))
    upper_bound = float(task_defaults.get("upper_bound", 5.0))
    eval_seeds = tuple(int(x) for x in task_defaults.get("eval_seeds", [11, 29, 47]))
    return {
        "sphere_5d": BenchmarkTask("sphere_5d", "sphere", 5, budget, lower_bound, upper_bound, eval_seeds),
        "rastrigin_5d": BenchmarkTask("rastrigin_5d", "rastrigin", 5, budget, lower_bound, upper_bound, eval_seeds),
        "rosenbrock_5d": BenchmarkTask("rosenbrock_5d", "rosenbrock", 5, budget, lower_bound, upper_bound, eval_seeds),
        "mixed_5d": BenchmarkTask("mixed_5d", "mixed", 5, budget, lower_bound, upper_bound, eval_seeds)
    }


def make_objective(name: str):
    if name == "sphere":
        return lambda x: float(np.sum(np.square(x)))
    if name == "rastrigin":
        return lambda x: float(10 * len(x) + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x)))
    if name == "rosenbrock":
        return lambda x: float(np.sum(100.0 * np.square(x[1:] - np.square(x[:-1])) + np.square(1 - x[:-1])))
    if name == "ackley":
        def ackley(x):
            x = np.asarray(x, dtype=float)
            a = 20.0
            b = 0.2
            c = 2 * np.pi
            n = x.size
            sum_sq = np.sum(x ** 2)
            sum_cos = np.sum(np.cos(c * x))
            return float(-a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e)
        return ackley
    raise ValueError(f"Unknown objective: {name}")


def objective_names_for_task(task: BenchmarkTask) -> list[str]:
    if task.objective == "mixed":
        return ["sphere", "rastrigin", "rosenbrock"]
    return [task.objective]


def score_from_best_f(best_f: float) -> float:
    return -math.log10(max(float(best_f), 1e-12))


def validate_python_code(code: str) -> None:
    tree = ast.parse(code)
    allowed_roots = {"numpy", "np", "math","random"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_roots:
                    raise ValueError(f"Disallowed import: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ValueError("Relative imports are not allowed")
            root = node.module.split(".")[0]
            if root not in allowed_roots:
                raise ValueError(f"Disallowed import-from: {node.module}")


class BudgetedObjective:
    def __init__(self, fn, budget: int):
        self.fn = fn
        self.budget = int(budget)
        self.calls = 0

    def __call__(self, x):
        if self.calls >= self.budget:
            raise RuntimeError("Evaluation budget exceeded")
        self.calls += 1
        x = np.asarray(x, dtype=float)
        return float(self.fn(x))


def _coerce_result(result: Any, fallback_best_f: float, history: list[float]) -> tuple[float, list[float]]:
    if isinstance(result, dict):
        best_f = result.get("best_f", fallback_best_f)
        raw_history = result.get("history", history)
        try:
            best_f = float(best_f)
        except Exception:
            best_f = fallback_best_f
        hist = []
        for item in list(raw_history)[:]:
            try:
                hist.append(float(item))
            except Exception:
                continue
        if not hist:
            hist = history
        return best_f, hist

    if isinstance(result, (tuple, list)):
        if len(result) >= 2:
            try:
                best_f = float(result[1])
            except Exception:
                best_f = fallback_best_f
            hist = history
            if len(result) >= 3 and isinstance(result[2], (list, tuple)):
                maybe_hist = []
                for item in result[2]:
                    try:
                        maybe_hist.append(float(item))
                    except Exception:
                        continue
                if maybe_hist:
                    hist = maybe_hist
            return best_f, hist

    try:
        return float(result), history
    except Exception:
        return fallback_best_f, history


def evaluate_solver_callable(solver, task: BenchmarkTask) -> dict[str, Any]:
    per_problem = []
    all_scores = []
    objective_means = []

    for objective_name in objective_names_for_task(task):
        fn = make_objective(objective_name)
        seed_best_fs = []
        for seed in task.eval_seeds:
            rng = np.random.default_rng(seed)
            wrapped = BudgetedObjective(fn, task.budget)
            best_f = float("inf")
            hist = []
            try:
                result = solver(
                    wrapped,
                    task.budget,
                    task.dim,
                    task.lower_bound,
                    task.upper_bound,
                    int(seed),
                )
                best_f, hist = _coerce_result(result, best_f, hist)
            except Exception:
                best_f = float("inf")
                hist = []
            if not np.isfinite(best_f):
                best_f = 1e12
            if not hist:
                hist = [best_f]
            best_f = min(best_f, min(hist))
            seed_best_fs.append(best_f)
            all_scores.append(score_from_best_f(best_f))
        problem_mean = float(np.mean(seed_best_fs))
        objective_means.append(problem_mean)
        per_problem.append({
            "objective": objective_name,
            "mean_best_f": problem_mean,
            "score": float(np.mean([score_from_best_f(x) for x in seed_best_fs]))
        })

    return {
        "fitness": float(np.mean(all_scores)),
        "raw_objective_mean": float(np.mean(objective_means)),
        "per_problem": per_problem,
    }


def llamea_task_prompt(task: BenchmarkTask) -> str:
    objectives = ", ".join(objective_names_for_task(task))
    return f"""
You are designing a Python optimizer for continuous black-box minimization.

Write Python code that defines exactly one top-level function named `solve` with this signature:

```python
def solve(objective, budget, dim, lower_bound, upper_bound, seed):
    ...
```

Rules:
- Use only numpy, math and optionally random.
- The goal is to MINIMIZE the objective.
- Never call `objective` more than `budget` times.
- Respect the scalar box bounds [`lower_bound`, `upper_bound`] in every candidate vector.
- Use the integer `seed` for deterministic randomness.
- Return either:
  1. a dict with keys `best_x`, `best_f`, `history`, or
  2. a tuple `(best_x, best_f, history)`.
- `history` should be the best-so-far objective values over time.
- Keep the code self-contained.

The optimizer will be evaluated on: {objectives}.
Dimension: {task.dim}
Budget per run: {task.budget}
The search space is continuous.
Higher benchmark score is achieved by obtaining lower final objective values.
""".strip()


def llm4ad_template_program() -> str:
    return """
import math
import numpy as np

def solve(objective, budget, dim, lower_bound, upper_bound, seed):
    rng = np.random.default_rng(seed)
    best_x = rng.uniform(lower_bound, upper_bound, size=dim)
    best_f = float(objective(best_x))
    history = [best_f]
    for _ in range(1, budget):
        x = rng.uniform(lower_bound, upper_bound, size=dim)
        fx = float(objective(x))
        if fx < best_f:
            best_x = x.copy()
            best_f = fx
        history.append(best_f)
    return {
        "best_x": best_x.tolist(),
        "best_f": float(best_f),
        "history": history,
    }
""".strip()


def llm4ad_task_description(task: BenchmarkTask) -> str:
    objectives = ", ".join(objective_names_for_task(task))
    return f"""
Improve the body of the Python function `solve(objective, budget, dim, lower_bound, upper_bound, seed)`.

The function must implement a continuous black-box optimizer that minimizes the objective value.
Use only numpy, math and optionally random.
Do not exceed the evaluation budget.
Return a dict with keys `best_x`, `best_f`, and `history`.

This function will be evaluated on: {objectives}
Dimension: {task.dim}
Budget: {task.budget}
""".strip()

from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


INVALID_OBJECTIVE = 1e12
_ALLOWED_IMPORT_ROOTS = {"numpy", "np", "math", "random"}
_DISALLOWED_CALL_NAMES = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "__import__",
}
_DISALLOWED_ATTRIBUTE_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "socket",
    "shutil",
    "ctypes",
    "multiprocessing",
    "threading",
    "concurrent",
    "requests",
    "urllib",
    "importlib",
    "builtins",
}
_DISALLOWED_ATTRIBUTE_NAMES = {
    "system",
    "popen",
    "Popen",
    "run",
    "kill",
    "terminate",
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "rename",
    "replace",
    "mkdir",
    "makedirs",
    "spawn",
    "fork",
}

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "False": False,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "sum": sum,
    "True": True,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
    "__import__": __import__,
}


class SkipCurrentGeneration(RuntimeError):
    pass


def _read_skip_request_count() -> int:
    path_str = os.environ.get("BENCHMARK_SKIP_SIGNAL_FILE", "").strip()
    if not path_str:
        return 0
    path = Path(path_str)
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    for key in ("skip_count", "generation"):
        try:
            return int(payload.get(key, 0) or 0)
        except Exception:
            continue
    return 0


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
        "mixed_5d": BenchmarkTask("mixed_5d", "mixed", 5, budget, lower_bound, upper_bound, eval_seeds),
    }


def make_objective(name: str):
    if name == "sphere":
        def sphere(x):
            x = np.asarray(x, dtype=float)
            with np.errstate(all="ignore"):
                return float(np.sum(np.square(x)))
        return sphere

    if name == "rastrigin":
        def rastrigin(x):
            x = np.asarray(x, dtype=float)
            with np.errstate(all="ignore"):
                return float(10 * x.size + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x)))
        return rastrigin

    if name == "rosenbrock":
        def rosenbrock(x):
            x = np.asarray(x, dtype=float)
            with np.errstate(all="ignore"):
                return float(np.sum(100.0 * np.square(x[1:] - np.square(x[:-1])) + np.square(1 - x[:-1])))
        return rosenbrock

    if name == "ackley":
        def ackley(x):
            x = np.asarray(x, dtype=float)
            a = 20.0
            b = 0.2
            c = 2 * np.pi
            n = max(int(x.size), 1)
            with np.errstate(all="ignore"):
                sum_sq = np.sum(np.square(x))
                sum_cos = np.sum(np.cos(c * x))
                value = -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e
            return float(value)
        return ackley

    raise ValueError(f"Unknown objective: {name}")


def objective_names_for_task(task: BenchmarkTask) -> list[str]:
    if task.objective == "mixed":
        return ["sphere", "rastrigin", "rosenbrock"]
    return [task.objective]


def score_from_best_f(best_f: float) -> float:
    return -math.log10(max(float(best_f), 1e-12))


def get_safe_exec_globals() -> dict[str, Any]:
    return {
        "np": np,
        "numpy": np,
        "math": math,
        "__builtins__": dict(_SAFE_BUILTINS),
    }


def validate_python_code(code: str) -> None:
    tree = ast.parse(code)
    has_solve = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"Disallowed import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ValueError("Relative imports are not allowed")
            root = node.module.split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Disallowed import-from: {node.module}")

        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            has_solve = True

        if isinstance(node, ast.Name) and node.id in _DISALLOWED_CALL_NAMES:
            raise ValueError(f"Disallowed builtin usage: {node.id}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALL_NAMES:
                raise ValueError(f"Disallowed call: {node.func.id}")
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in _DISALLOWED_ATTRIBUTE_NAMES:
                    raise ValueError(f"Disallowed attribute call: {node.func.attr}")
                if isinstance(node.func.value, ast.Name) and node.func.value.id in _DISALLOWED_ATTRIBUTE_ROOTS:
                    raise ValueError(f"Disallowed module usage: {node.func.value.id}.{node.func.attr}")

        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in _DISALLOWED_ATTRIBUTE_ROOTS:
                raise ValueError(f"Disallowed module usage: {node.value.id}.{node.attr}")

    if not has_solve:
        raise ValueError("Generated code does not define a top-level solve()")


class BudgetedObjective:
    def __init__(self, fn, budget: int, dim: int, lower_bound: float, upper_bound: float, skip_baseline: int | None = None):
        self.fn = fn
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.skip_baseline = int(skip_baseline or 0)
        self.calls = 0
        self.best_f = float("inf")
        self.best_x: list[float] | None = None
        self.best_history: list[float] = []

    def __call__(self, x):
        if _read_skip_request_count() > self.skip_baseline:
            raise SkipCurrentGeneration("Skipped current generation from terminal")
        if self.calls >= self.budget:
            raise RuntimeError("Evaluation budget exceeded")
        self.calls += 1

        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != self.dim:
            raise ValueError(f"Expected candidate dimension {self.dim}, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError("Candidate vector contains NaN or Inf")

        arr = np.clip(arr, self.lower_bound, self.upper_bound)
        with np.errstate(all="ignore"):
            value = float(self.fn(arr))
        if not np.isfinite(value):
            raise FloatingPointError("Objective returned NaN or Inf")

        if value < self.best_f:
            self.best_f = value
            self.best_x = arr.astype(float).tolist()
        self.best_history.append(self.best_f)
        return value


def _coerce_result(result: Any) -> tuple[float | None, list[float], list[float] | None]:
    returned_best_f: float | None = None
    returned_history: list[float] = []
    returned_best_x: list[float] | None = None

    if isinstance(result, dict):
        best_f = result.get("best_f")
        best_x = result.get("best_x")
        raw_history = result.get("history", [])

        try:
            returned_best_f = float(best_f)
            if not math.isfinite(returned_best_f):
                returned_best_f = None
        except Exception:
            returned_best_f = None

        if best_x is not None:
            try:
                arr = np.asarray(best_x, dtype=float).reshape(-1)
                if np.all(np.isfinite(arr)):
                    returned_best_x = arr.astype(float).tolist()
            except Exception:
                returned_best_x = None

        try:
            for item in list(raw_history):
                value = float(item)
                if math.isfinite(value):
                    returned_history.append(value)
        except Exception:
            returned_history = []
        return returned_best_f, returned_history, returned_best_x

    if isinstance(result, (tuple, list)):
        if len(result) >= 2:
            try:
                returned_best_f = float(result[1])
                if not math.isfinite(returned_best_f):
                    returned_best_f = None
            except Exception:
                returned_best_f = None

            if len(result) >= 1:
                try:
                    arr = np.asarray(result[0], dtype=float).reshape(-1)
                    if np.all(np.isfinite(arr)):
                        returned_best_x = arr.astype(float).tolist()
                except Exception:
                    returned_best_x = None

            if len(result) >= 3 and isinstance(result[2], (list, tuple, np.ndarray)):
                for item in result[2]:
                    try:
                        value = float(item)
                    except Exception:
                        continue
                    if math.isfinite(value):
                        returned_history.append(value)
        return returned_best_f, returned_history, returned_best_x

    try:
        returned_best_f = float(result)
        if not math.isfinite(returned_best_f):
            returned_best_f = None
    except Exception:
        returned_best_f = None
    return returned_best_f, returned_history, returned_best_x


def evaluate_solver_callable(solver, task: BenchmarkTask, skip_baseline: int | None = None) -> dict[str, Any]:
    per_problem = []
    all_scores: list[float] = []
    objective_means: list[float] = []
    total_calls = 0

    for objective_name in objective_names_for_task(task):
        fn = make_objective(objective_name)
        seed_best_fs: list[float] = []
        seed_calls: list[int] = []

        for seed in task.eval_seeds:
            wrapped = BudgetedObjective(fn, task.budget, task.dim, task.lower_bound, task.upper_bound, skip_baseline=skip_baseline)

            try:
                result = solver(
                    wrapped,
                    task.budget,
                    task.dim,
                    task.lower_bound,
                    task.upper_bound,
                    int(seed),
                )
            except Exception as exc:
                raise RuntimeError(f"{objective_name}/seed {seed}: {type(exc).__name__}: {exc}") from exc

            returned_best_f, returned_history, returned_best_x = _coerce_result(result)
            if wrapped.calls <= 0:
                raise ValueError(f"{objective_name}/seed {seed}: solver never called the objective")
            if not math.isfinite(wrapped.best_f):
                raise FloatingPointError(f"{objective_name}/seed {seed}: no finite objective value was observed")

            observed_best_f = float(wrapped.best_f)
            best_f = observed_best_f
            consistency_gap = None

            if returned_best_x is not None:
                arr = np.asarray(returned_best_x, dtype=float).reshape(-1)
                if arr.size == task.dim and np.all(np.isfinite(arr)):
                    arr = np.clip(arr, task.lower_bound, task.upper_bound)
                    with np.errstate(all="ignore"):
                        returned_x_value = float(fn(arr))
                    if math.isfinite(returned_x_value):
                        best_f = min(best_f, returned_x_value)
                        consistency_gap = abs(returned_x_value - observed_best_f)

            if returned_best_f is not None:
                if consistency_gap is None:
                    consistency_gap = abs(returned_best_f - observed_best_f)
                else:
                    consistency_gap = min(consistency_gap, abs(returned_best_f - observed_best_f))

            if consistency_gap is not None and consistency_gap > 1e-6 * max(1.0, abs(observed_best_f)):
                raise ValueError(
                    f"{objective_name}/seed {seed}: returned best_f/history is inconsistent with observed objective evaluations"
                )

            seed_best_fs.append(best_f)
            seed_calls.append(wrapped.calls)
            total_calls += wrapped.calls
            all_scores.append(score_from_best_f(best_f))

        problem_mean = float(np.mean(seed_best_fs))
        objective_means.append(problem_mean)
        per_problem.append(
            {
                "objective": objective_name,
                "mean_best_f": problem_mean,
                "score": float(np.mean([score_from_best_f(x) for x in seed_best_fs])),
                "mean_objective_calls": float(np.mean(seed_calls)) if seed_calls else 0.0,
            }
        )

    return {
        "fitness": float(np.mean(all_scores)),
        "raw_objective_mean": float(np.mean(objective_means)),
        "per_problem": per_problem,
        "objective_calls_total": int(total_calls),
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
- Use only numpy, math, and optionally random.
- The goal is to MINIMIZE the objective.
- Never call `objective` more than `budget` times.
- Respect the scalar box bounds [`lower_bound`, `upper_bound`] in every candidate vector.
- Use the integer `seed` for deterministic randomness.
- Return either:
  1. a dict with keys `best_x`, `best_f`, `history`, or
  2. a tuple `(best_x, best_f, history)`.
- `best_f` and `history` must come only from real objective evaluations. Never fabricate them.
- `history` should be the best-so-far objective values over time.
- Keep the code self-contained.
- Do not use scipy or any other external library.
- Avoid numerically unstable operations, unsafe matrix inversions, and divisions by values that may be zero.
- Clip any proposed candidate back into the box bounds before calling `objective`.
- Do not output explanations, only valid Python code.

The optimizer will be evaluated on: {objectives}.
Dimension: {task.dim}
Budget per run: {task.budget}
The search space is continuous.
Higher benchmark score is achieved by obtaining lower final objective values.
""".strip()


def llm4ad_template_program() -> str:
    return """
import math
import random
import numpy as np

def solve(objective, budget, dim, lower_bound, upper_bound, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    def clip(x):
        return np.clip(np.asarray(x, dtype=float), lower_bound, upper_bound)

    best_x = clip(rng.uniform(lower_bound, upper_bound, size=dim))
    best_f = float(objective(best_x))
    history = [best_f]

    step = max((upper_bound - lower_bound) * 0.2, 1e-3)
    while len(history) < budget:
        local_trials = min(8, budget - len(history))
        improved = False
        for _ in range(local_trials):
            candidate = clip(best_x + rng.normal(0.0, step, size=dim))
            value = float(objective(candidate))
            if value < best_f:
                best_x = candidate
                best_f = value
                improved = True
            history.append(best_f)
            if len(history) >= budget:
                break
        if not improved:
            step *= 0.7
            if step < 1e-4:
                step = max((upper_bound - lower_bound) * 0.05, 1e-4)
                restart = clip(rng.uniform(lower_bound, upper_bound, size=dim))
                value = float(objective(restart))
                if value < best_f:
                    best_x = restart
                    best_f = value
                if len(history) < budget:
                    history.append(best_f)
        else:
            step = min(step * 1.05, max((upper_bound - lower_bound) * 0.5, 1e-3))

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
Use only numpy, math, and optionally random.
Do not use scipy or any other external library.
Do not exceed the evaluation budget.
Always clip candidate vectors to the box bounds before calling the objective.
Return a dict with keys `best_x`, `best_f`, and `history`.
`best_f` and `history` must be derived only from real objective evaluations.
Avoid unstable covariance updates, unsafe matrix inversions, and divisions by values that may be zero.

This function will be evaluated on: {objectives}
Dimension: {task.dim}
Budget: {task.budget}
""".strip()

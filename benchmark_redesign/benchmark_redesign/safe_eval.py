from __future__ import annotations

import ast
import builtins
import math
import random
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .tasks import BenchmarkTask, BudgetedObjective, PENALTY_BEST_F, score_from_best_f


ALLOWED_IMPORT_ROOTS = {
    "math",
    "random",
    "numpy",
    "np",
}


@dataclass(slots=True)
class EvalResult:
    status: str
    score: float
    best_f: float
    mean_best_f: float
    details: dict[str, Any]
    error: str = ""


class CodeValidationError(RuntimeError):
    pass


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in BANNED_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import of '{name}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "sum",
        "tuple",
        "zip",
    ]
}
SAFE_BUILTINS["__import__"] = _restricted_import


BANNED_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
}


BANNED_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
    "http",
    "urllib",
    "pickle",
    "joblib",
    "multiprocessing",
    "threading",
}


def validate_python_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(f"Syntax error: {exc}") from exc

    top_level_solver_found = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                modules = [node.module.split(".")[0]]
            for module in modules:
                if module in BANNED_IMPORT_ROOTS:
                    raise CodeValidationError(f"Import of '{module}' is not allowed")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                raise CodeValidationError(f"Call to '{node.func.id}' is not allowed")
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            top_level_solver_found = True

    if not top_level_solver_found:
        raise CodeValidationError("Expected a top-level function named 'solve'")


def load_solver_callable(code: str) -> Callable:
    validate_python_code(code)
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "numpy": np,
        "math": math,
        "random": random,
    }
    exec(code, namespace, namespace)
    solve = namespace.get("solve")
    if not callable(solve):
        raise CodeValidationError("Generated code did not define callable solve()")
    return solve


def _normalize_solver_output(result: Any, objective: BudgetedObjective) -> tuple[np.ndarray | None, float]:
    if isinstance(result, dict):
        best_x = result.get("best_x")
        best_f = result.get("best_f", objective.best_f)
        return (None if best_x is None else np.asarray(best_x, dtype=float)), float(best_f)
    if isinstance(result, tuple) and len(result) >= 2:
        return np.asarray(result[0], dtype=float), float(result[1])
    if isinstance(result, (float, int, np.floating)):
        return objective.best_x, float(result)
    return objective.best_x, float(objective.best_f)


def evaluate_solver_callable(code: str, task: BenchmarkTask, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    objective = BudgetedObjective(
        objective=task.objective,
        dim=task.dim,
        budget=task.budget,
        lower_bound=task.lower_bound,
        upper_bound=task.upper_bound,
    )
    solve = load_solver_callable(code)

    best_x = None
    best_f = PENALTY_BEST_F
    try:
        result = solve(
            objective,
            task.budget,
            task.dim,
            task.lower_bound,
            task.upper_bound,
            rng,
        )
        best_x, best_f = _normalize_solver_output(result, objective)
    except Exception as exc:
        return {
            "seed": seed,
            "status": "failed",
            "best_f": PENALTY_BEST_F,
            "score": score_from_best_f(PENALTY_BEST_F),
            "used_budget": objective.used_budget,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }

    if best_x is not None and objective.used_budget < task.budget:
        try:
            best_f = min(best_f, objective(best_x))
        except Exception:
            pass

    if not math.isfinite(best_f):
        best_f = PENALTY_BEST_F

    return {
        "seed": seed,
        "status": "ok",
        "best_f": float(best_f),
        "score": score_from_best_f(best_f),
        "used_budget": objective.used_budget,
        "best_x": None if best_x is None else np.asarray(best_x, dtype=float).tolist(),
    }


def evaluate_code_on_task(code: str, task: BenchmarkTask) -> EvalResult:
    per_seed = [evaluate_solver_callable(code, task, seed) for seed in task.eval_seeds]
    ok = [x for x in per_seed if x["status"] == "ok"]
    if not ok:
        return EvalResult(
            status="failed",
            score=score_from_best_f(PENALTY_BEST_F),
            best_f=PENALTY_BEST_F,
            mean_best_f=PENALTY_BEST_F,
            details={"per_seed": per_seed},
            error="All evaluation seeds failed",
        )

    mean_best_f = float(np.mean([x["best_f"] for x in ok]))
    best_f = float(np.min([x["best_f"] for x in ok]))
    return EvalResult(
        status="ok",
        score=score_from_best_f(mean_best_f),
        best_f=best_f,
        mean_best_f=mean_best_f,
        details={"per_seed": per_seed},
        error="",
    )

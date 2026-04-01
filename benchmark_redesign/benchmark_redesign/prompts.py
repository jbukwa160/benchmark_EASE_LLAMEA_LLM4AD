from __future__ import annotations

from .tasks import BenchmarkTask


def llm_system_prompt(task: BenchmarkTask) -> str:
    return (
        "You are an expert numerical optimization researcher writing robust Python code. "
        "Return only Python code."
    )


def llm_user_prompt(task: BenchmarkTask) -> str:
    return f"""
Write exactly one top-level Python function named `solve` with this signature:

```python
def solve(func, budget, dim, lower_bound, upper_bound, rng):
    ...
    return best_x, best_f
```

Task:
- Minimize a black-box {task.name} objective in {task.dim} dimensions.
- The search domain is [{task.lower_bound}, {task.upper_bound}] for every dimension.
- You may call `func(x)` at most `budget` times.
- `func(x)` returns a scalar objective value to minimize.
- `rng` is a NumPy random generator.
- Use only Python standard library, `math`, `random`, and `numpy`.
- Do not import os, sys, subprocess, requests, pathlib, pickle, multiprocessing, or threading.
- Always keep candidates inside bounds.
- Be defensive: handle small budgets and numerical instability.
- Return the best point found and its best objective value.

Output rules:
- Return only a Python code block or raw Python code.
- The function name must be exactly `solve`.
- No explanations before or after the code.
""".strip()


def llm4ad_task_description(task: BenchmarkTask) -> str:
    return llm_user_prompt(task)


def llm4ad_template_program() -> str:
    return """
import math
import random
import numpy as np


def solve(func, budget, dim, lower_bound, upper_bound, rng):
    \"\"\"Minimize the objective and return (best_x, best_f).\"\"\"
    best_x = np.zeros(dim, dtype=float)
    best_f = float(func(best_x))
    return best_x, best_f
""".strip() + "\n"

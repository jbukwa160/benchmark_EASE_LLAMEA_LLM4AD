"""
utils.py — Shared utilities: logging, result I/O, stats, validation, safe execution.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PENALTY_OBJECTIVE = 1e12


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

class ResultStore:
    """Append-friendly JSON-lines result file with lightweight resume helpers."""

    def __init__(self, output_dir: str, append: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.append = append
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def result_path(self, framework: str, task: str) -> Path:
        return self.output_dir / f"{framework}_{task}.jsonl"

    def write(self, framework: str, task: str, record: dict[str, Any]):
        path = self.result_path(framework, task)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self, framework: str, task: str) -> list[dict[str, Any]]:
        path = self.result_path(framework, task)
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def has_completed_record(self, framework: str, task: str, seed: int) -> bool:
        return any(r.get("seed") == seed and r.get("success") for r in self.read_all(framework, task))


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Safe generated-code validation
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT_PREFIXES = {
    "threading",
    "_thread",
    "multiprocessing",
    "subprocess",
    "socket",
    "asyncio",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "telnetlib",
    "paramiko",
    "pathlib",
    "shutil",
    "tempfile",
    "pickle",
    "dill",
    "cloudpickle",
    "joblib",
    "concurrent",
    "signal",
    "resource",
    "ctypes",
}

_FORBIDDEN_CALL_NAMES = {
    "exec",
    "eval",
    "compile",
    "open",
    "input",
    "breakpoint",
    "__import__",
}

_FORBIDDEN_ATTR_PATHS = {
    "os.system",
    "os.popen",
    "os.spawnl",
    "os.spawnlp",
    "os.spawnv",
    "os.spawnvp",
    "os.startfile",
    "sys.exit",
    "subprocess.Popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "threading.Thread",
    "multiprocessing.Process",
    "multiprocessing.Pool",
    "concurrent.futures.ThreadPoolExecutor",
    "concurrent.futures.ProcessPoolExecutor",
    "pathlib.Path.open",
}


class CodeValidationError(ValueError):
    pass


class _GeneratedCodeValidator(ast.NodeVisitor):
    def _attribute_path(self, node: ast.AST) -> str:
        parts: list[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _FORBIDDEN_IMPORT_PREFIXES:
                raise CodeValidationError(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        root = module.split(".")[0]
        if root in _FORBIDDEN_IMPORT_PREFIXES:
            raise CodeValidationError(f"Forbidden import: {module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
            raise CodeValidationError(f"Forbidden call: {node.func.id}()")
        attr_path = self._attribute_path(node.func)
        if attr_path in _FORBIDDEN_ATTR_PATHS:
            raise CodeValidationError(f"Forbidden call: {attr_path}()")
        self.generic_visit(node)


def validate_generated_code(source: str) -> tuple[bool, str]:
    if not source or not source.strip():
        return False, "Generated code is empty."
    if len(source) > 120_000:
        return False, "Generated code is too large."

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc.msg} (line {exc.lineno})"

    try:
        _GeneratedCodeValidator().visit(tree)
    except CodeValidationError as exc:
        return False, str(exc)

    fn_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_defs:
        return False, "No top-level function definition found."

    return True, "ok"


# ---------------------------------------------------------------------------
# Safe candidate evaluation in a fresh subprocess
# ---------------------------------------------------------------------------

_EVAL_SUBPROCESS_SCRIPT = textwrap.dedent(
    r"""
    import json
    import math
    import sys
    import warnings
    import numpy as np

    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")

    PENALTY = 1e12

    def sphere(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x ** 2))

    def rastrigin(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

    def rosenbrock(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    TASKS = {
        "sphere_5d": (5, sphere),
        "rastrigin_5d": (5, rastrigin),
        "rosenbrock_5d": (5, rosenbrock),
    }

    def main():
        payload = json.loads(sys.stdin.read())
        code = payload["code"]
        task_name = payload["task_name"]
        lb = float(payload["lower_bound"])
        ub = float(payload["upper_bound"])
        budget = int(payload["budget"])
        eval_seeds = list(payload["eval_seeds"])

        dim, task_fn = TASKS[task_name]
        namespace = {"np": np}
        try:
            exec(code, namespace)
        except Exception as exc:
            print(json.dumps({"mean_objective": PENALTY, "scores": [], "message": f"compile error: {exc}"}))
            return

        fn = namespace.get("algorithm") or namespace.get("optimise") or namespace.get("optimize")
        if fn is None:
            callables = [v for v in namespace.values() if callable(v) and not isinstance(v, type)]
            fn = callables[-1] if callables else None
        if fn is None:
            print(json.dumps({"mean_objective": PENALTY, "scores": [], "message": "no callable found"}))
            return

        scores = []
        failures = 0
        for seed in eval_seeds:
            rng = np.random.default_rng(int(seed))
            try:
                result = fn(task_fn, dim, lb, ub, budget, rng)
                if isinstance(result, tuple) and result:
                    result = result[0]
                arr = np.asarray(result, dtype=float)
                if arr.shape == ():
                    val = float(arr)
                else:
                    arr = arr.reshape(-1)
                    if arr.shape != (dim,):
                        raise ValueError(f"expected shape {(dim,)}, got {arr.shape}")
                    arr = np.clip(arr, lb, ub)
                    val = float(task_fn(arr))
                if not math.isfinite(val):
                    val = PENALTY
                    failures += 1
            except Exception:
                val = PENALTY
                failures += 1
            scores.append(val)

        mean_objective = float(np.mean(scores)) if scores else PENALTY
        print(json.dumps({
            "mean_objective": mean_objective,
            "scores": scores,
            "message": f"failures={failures}/{len(eval_seeds)}",
        }))

    if __name__ == "__main__":
        main()
    """
)


def evaluate_candidate_code(
    code: str,
    *,
    task_name: str,
    lower_bound: float,
    upper_bound: float,
    budget: int,
    eval_seeds: list[int],
    timeout_seconds: int | float,
) -> tuple[float, str, str | None]:
    """Return (mean_objective, feedback, error). Lower is better."""

    ok, reason = validate_generated_code(code)
    if not ok:
        return PENALTY_OBJECTIVE, reason, reason

    payload = {
        "code": code,
        "task_name": task_name,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "budget": budget,
        "eval_seeds": eval_seeds,
    }

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _EVAL_SUBPROCESS_SCRIPT],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        msg = f"Evaluation timed out after {timeout_seconds}s"
        return PENALTY_OBJECTIVE, msg, msg
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Evaluation subprocess failed: {exc}"
        return PENALTY_OBJECTIVE, msg, msg

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        msg = stderr or stdout or f"Evaluation subprocess exited with code {completed.returncode}"
        return PENALTY_OBJECTIVE, msg, msg

    try:
        data = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except Exception:
        msg = stderr or stdout or "Evaluation subprocess produced invalid output"
        return PENALTY_OBJECTIVE, msg, msg

    mean_objective = float(data.get("mean_objective", PENALTY_OBJECTIVE))
    if not math.isfinite(mean_objective):
        mean_objective = PENALTY_OBJECTIVE
    feedback = str(data.get("message") or f"mean_obj={mean_objective:.6f}")
    error = None if mean_objective < PENALTY_OBJECTIVE else feedback
    return mean_objective, feedback, error


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), 15)
            except Exception:
                proc.terminate()
            time.sleep(1)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except Exception:
                    proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def child_creation_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}

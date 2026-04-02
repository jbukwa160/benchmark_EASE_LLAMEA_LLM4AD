"""Adapter for LLaMEA with robust repo discovery and Windows-safe pickle handling."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib import request as _urllib_request

from .base import BaseAdapter
from ..tasks import BenchmarkTask
from ..utils import PENALTY_OBJECTIVE, evaluate_candidate_code, get_logger, timestamp

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Repo discovery helpers
# ---------------------------------------------------------------------------

def _candidate_repo_dirs(config_dir: Path, configured: str | None, package_dir: str, framework_names: list[str]) -> list[Path]:
    raw_candidates: list[Path] = []
    if configured:
        p = Path(configured)
        raw_candidates.append(p if p.is_absolute() else (config_dir / p))

    parent = config_dir.parent
    for name in framework_names:
        raw_candidates.extend([
            parent / name,
            parent / f"{name}-main",
            config_dir / name,
            config_dir / f"{name}-main",
        ])

    try:
        for p in parent.iterdir():
            if p.is_dir() and p.name.lower().startswith(tuple(n.lower() for n in framework_names)):
                raw_candidates.append(p)
    except Exception:
        pass

    seen: set[str] = set()
    resolved: list[Path] = []
    for p in raw_candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(rp)

    valid: list[Path] = []
    for p in resolved:
        if (p / package_dir / "__init__.py").exists():
            valid.append(p)
        elif p.name == package_dir and (p / "__init__.py").exists():
            valid.append(p.parent)
    return valid


# ---------------------------------------------------------------------------
# Picklable evaluator and LLM helpers
# ---------------------------------------------------------------------------

class _LLaMEAEvaluator:
    def __init__(self, task_name: str, lower_bound: float, upper_bound: float, budget: int, eval_seeds: list[int], timeout_seconds: int):
        self.task_name = task_name
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.budget = int(budget)
        self.eval_seeds = list(eval_seeds)
        self.timeout_seconds = int(timeout_seconds)

    def __call__(self, individual, _logger=None):
        mean_objective, feedback, err = evaluate_candidate_code(
            individual.code,
            task_name=self.task_name,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            budget=self.budget,
            eval_seeds=self.eval_seeds,
            timeout_seconds=self.timeout_seconds,
        )
        individual.set_scores(-mean_objective)
        individual.feedback = feedback
        if err:
            individual.error = err
        if hasattr(individual, "add_metadata"):
            individual.add_metadata("mean_objective", mean_objective)
        return individual


def _reconstruct_ollama_llm(base_module: str, base_name: str, endpoint: str, model: str, timeout: int, temperature: float):
    mod = importlib.import_module(base_module)
    base_cls = getattr(mod, base_name)
    return _make_ollama_llm(
        base_cls,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        temperature=temperature,
    )


def _make_ollama_llm(base_cls, *, endpoint: str, model: str, timeout: int, temperature: float):
    init_kw: dict[str, Any] = {}
    try:
        base_init_params = inspect.signature(base_cls.__init__).parameters
    except Exception:
        base_init_params = {}
    for name, val in [
        ("api_key", "ollama"),
        ("model", model),
        ("base_url", endpoint),
        ("do_auto_trim", False),
        ("debug_mode", False),
    ]:
        if name in base_init_params:
            init_kw[name] = val

    class OllamaLLM(base_cls):  # type: ignore[misc,valid-type]
        def __init__(self):
            try:
                base_cls.__init__(self, **init_kw)
            except TypeError:
                try:
                    base_cls.__init__(self)
                except TypeError:
                    pass
            self.model = model
            self._endpoint = endpoint
            self._model = model
            self._timeout = timeout
            self._temperature = temperature

        def __reduce__(self):
            return (
                _reconstruct_ollama_llm,
                (base_cls.__module__, base_cls.__name__, self._endpoint, self._model, self._timeout, self._temperature),
            )

        def query(self, session_messages, **kwargs) -> str:
            return self._http(session_messages)

        def get_response(self, prompt, **kwargs) -> str:
            return self._http([{"role": "user", "content": str(prompt)}])

        def chat(self, messages, **kwargs) -> str:
            return self._http(messages)

        def __call__(self, prompt, **kwargs) -> str:
            return self._http([{"role": "user", "content": str(prompt)}])

        def _http(self, messages, retries: int = 4) -> str:
            payload = json.dumps({
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                "stream": False,
            }).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer ollama",
            }
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                req = _urllib_request.Request(self._endpoint, data=payload, headers=headers, method="POST")
                try:
                    with _urllib_request.urlopen(req, timeout=self._timeout) as resp:
                        raw = json.loads(resp.read().decode("utf-8"))
                    return raw["choices"][0]["message"]["content"]
                except Exception as exc:
                    last_exc = exc
                    if attempt < retries:
                        time.sleep(2 * attempt)
            raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_exc}")

    OllamaLLM.__name__ = "OllamaLLM"
    OllamaLLM.__qualname__ = "OllamaLLM"
    OllamaLLM.__module__ = __name__
    return OllamaLLM()


def _llamea_noop_pickle_archive(self, *args, **kwargs):
    return None


def _patch_llamea_code_extraction():
    try:
        llm_mod = importlib.import_module("llamea.llm")
        base_cls = getattr(llm_mod, "LLM", None)
        no_code_exc = getattr(llm_mod, "NoCodeException", Exception)
        if base_cls is None or getattr(base_cls, "_auto_algo_plain_code_patch", False):
            return
        original = base_cls.extract_algorithm_code

        def patched(self, message):
            try:
                return original(self, message)
            except no_code_exc:
                msg = (message or "").strip()
                if ("def algorithm" in msg or "class " in msg or "import numpy" in msg) and "```" not in msg:
                    return msg
                raise

        base_cls.extract_algorithm_code = patched
        base_cls._auto_algo_plain_code_patch = True
    except Exception:
        pass


class LLaMEAAdapter(BaseAdapter):
    name = "llamea"

    def is_enabled(self) -> bool:
        return self.framework_cfg.get("enabled", False)

    def _repo_path(self) -> Path:
        config_dir = Path(self.global_cfg.get("_config_dir", Path.cwd()))
        configured = self.framework_cfg.get("repo_path", "../LLaMEA")
        candidates = _candidate_repo_dirs(config_dir, configured, "llamea", ["LLaMEA", "llamea"])
        if candidates:
            chosen = candidates[0]
            if str(chosen) != str(Path(configured)):
                logger.debug("Resolved LLaMEA repo path to %s", chosen)
            return chosen
        p = Path(configured)
        return p if p.is_absolute() else (config_dir / p).resolve()

    def _add_repo_to_path(self) -> None:
        p = self._repo_path()
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
            logger.debug("Added LLaMEA repo to sys.path: %s", p)

    def _build_evaluator(self, task: BenchmarkTask):
        return _LLaMEAEvaluator(
            task_name=task.name,
            lower_bound=task.lower_bound,
            upper_bound=task.upper_bound,
            budget=int(self.task_defaults.get("budget", 1500)),
            eval_seeds=list(self.task_defaults.get("eval_seeds", list(range(11, 21)))),
            timeout_seconds=int(self.framework_cfg.get("candidate_timeout_sec", 180)),
        )

    def run(self, task: BenchmarkTask, seed: int) -> dict[str, Any]:
        self._add_repo_to_path()
        t0 = time.perf_counter()
        result: dict[str, Any] = {
            "framework": self.name,
            "task": task.name,
            "seed": seed,
            "best_value": None,
            "success": False,
            "error": None,
            "elapsed_sec": None,
            "timestamp": timestamp(),
            "extra": {},
        }

        original_pickle_archive = None
        LLaMEA = None
        try:
            import numpy as np
            from llamea import LLaMEA  # type: ignore
            _patch_llamea_code_extraction()

            LLaMEABaseLLM = None
            for mod_path in ("llamea.llm", "llamea"):
                try:
                    mod = importlib.import_module(mod_path)
                    LLaMEABaseLLM = getattr(mod, "LLM", None)
                    if LLaMEABaseLLM is not None:
                        break
                except Exception:
                    continue
            if LLaMEABaseLLM is None:
                class LLaMEABaseLLM:  # type: ignore[no-redef]
                    def __init__(self, **kwargs):
                        pass

            random.seed(seed)
            np.random.seed(seed)

            cfg = self.framework_cfg
            ollama_v1 = self.ollama_base_url.rstrip("/") + "/v1/chat/completions"
            llm = _make_ollama_llm(
                LLaMEABaseLLM,
                endpoint=ollama_v1,
                model=self.ollama_model,
                timeout=int(cfg.get("llm_timeout_sec", 180)),
                temperature=float(cfg.get("temperature", 0.7)),
            )

            role_prompt = (
                "You are an expert in black-box continuous optimisation. "
                "Write a single sequential Python function only. "
                "Do not use threads, multiprocessing, subprocesses, files, "
                "network calls, eval, exec, or imports beyond numpy.\n\n"
            )
            task_prompt = (
                f"Task: {task.description}\n\n"
                "Return Python code only. Implement:\n"
                "def algorithm(func, dim, lb, ub, budget, rng):\n"
                "    ...\n"
                "The function must return the best 1-D numpy array found.\n"
                "Clip candidates to [lb, ub].\n"
            )

            ctor_params = inspect.signature(LLaMEA.__init__).parameters
            parallel_backend = cfg.get("parallel_backend", "threading")
            if int(cfg.get("max_workers", 1)) <= 1 and parallel_backend == "loky":
                parallel_backend = "threading"

            want: dict[str, Any] = dict(
                f=self._build_evaluator(task),
                llm=llm,
                budget=int(cfg.get("search_budget", 40)),
                n_parents=int(cfg.get("n_parents", 4)),
                n_offspring=int(cfg.get("n_offspring", 8)),
                role_prompt=role_prompt,
                task_prompt=task_prompt,
                experiment_name=f"llamea_{task.name}_seed{seed}",
                elitism=bool(cfg.get("elitism", True)),
                eval_timeout=int(cfg.get("eval_timeout", 600)),
                max_workers=int(cfg.get("max_workers", 1)),
                minimization=False,
                log=bool(cfg.get("framework_log", True)),
                parallel_backend=parallel_backend,
                parent_selection=cfg.get("parent_selection", "random"),
                tournament_size=int(cfg.get("tournament_size", 3)),
            )
            filtered = {k: v for k, v in want.items() if k in ctor_params}

            original_pickle_archive = getattr(LLaMEA, "pickle_archive", None)
            if original_pickle_archive is not None:
                LLaMEA.pickle_archive = _llamea_noop_pickle_archive  # type: ignore[assignment]
            optimizer = LLaMEA(**filtered)
            best_solution = optimizer.run()

            if best_solution is None:
                raise RuntimeError("LLaMEA.run() returned None.")

            bi = float(best_solution.fitness)
            bv = float(-bi) if math.isfinite(bi) else PENALTY_OBJECTIVE

            if not math.isfinite(bi) or bv >= PENALTY_OBJECTIVE:
                result["error"] = getattr(best_solution, "error", "") or "No valid solution produced."
            else:
                result["best_value"] = bv
                result["success"] = True

            result["extra"] = {
                "best_internal_score": bi,
                "best_name": getattr(best_solution, "name", "")[:200],
                "best_feedback": getattr(best_solution, "feedback", "")[:500],
                "best_error": getattr(best_solution, "error", "")[:500],
                "repo_path": str(self._repo_path()),
            }

        except Exception:
            tb = traceback.format_exc()
            logger.error("LLaMEA run failed:\n%s", tb)
            result["error"] = tb
        finally:
            if LLaMEA is not None and original_pickle_archive is not None:
                try:
                    LLaMEA.pickle_archive = original_pickle_archive  # type: ignore[assignment]
                except Exception:
                    pass

        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return result

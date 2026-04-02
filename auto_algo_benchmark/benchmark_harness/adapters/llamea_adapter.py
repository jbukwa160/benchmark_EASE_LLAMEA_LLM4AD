"""Adapter for LLaMEA — fixes pickling of the LLM object in pickle_archive."""

from __future__ import annotations

import inspect
import json
import math
import pickle
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib import request as _urllib_request

from .base import BaseAdapter
from ..config import resolve_repo_path
from ..tasks import BenchmarkTask
from ..utils import PENALTY_OBJECTIVE, evaluate_candidate_code, get_logger, timestamp

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Global registry so pickle can reconstruct OllamaLLM instances by ID
# ---------------------------------------------------------------------------
# When pickle serialises an OllamaLLM it stores (_reconstruct_ollama_llm, (id,))
# and the id maps back to the constructor args held here.
_OLLAMA_LLM_REGISTRY: dict[int, dict] = {}
_OLLAMA_LLM_COUNTER = 0


def _reconstruct_ollama_llm(registry_id: int):
    """Called by pickle to rebuild an OllamaLLM after unpickling."""
    args = _OLLAMA_LLM_REGISTRY.get(registry_id)
    if args is None:
        raise pickle.UnpicklingError(
            f"OllamaLLM registry id {registry_id} not found. "
            "This usually means pickling happened in a different process."
        )
    return _build_ollama_llm_instance(**args)


def _build_ollama_llm_instance(*, LLaMEABaseLLM_module: str,
                                LLaMEABaseLLM_name: str,
                                endpoint: str, model: str,
                                timeout: int, temperature: float,
                                registry_id: int):
    """Reconstruct the OllamaLLM from serialisable parts."""
    import importlib
    mod = importlib.import_module(LLaMEABaseLLM_module)
    LLaMEABaseLLM = getattr(mod, LLaMEABaseLLM_name)
    return _make_ollama_llm(
        LLaMEABaseLLM, endpoint=endpoint, model=model,
        timeout=timeout, temperature=temperature,
        _registry_id=registry_id,
    )


def _make_ollama_llm(LLaMEABaseLLM, *, endpoint: str, model: str,
                     timeout: int, temperature: float,
                     _registry_id: int | None = None):
    """
    Create an OllamaLLM that:
      1. Inherits from LLaMEABaseLLM so LLaMEA's type-checks pass.
      2. Is fully picklable via __reduce__ (needed for LLaMEA's pickle_archive).
      3. Routes all LLM calls to Ollama over HTTP.
    """
    global _OLLAMA_LLM_COUNTER

    # Register this instance so pickle can reconstruct it
    if _registry_id is None:
        _OLLAMA_LLM_COUNTER += 1
        _registry_id = _OLLAMA_LLM_COUNTER

    base_init_params = inspect.signature(LLaMEABaseLLM.__init__).parameters
    init_kw: dict[str, Any] = {}
    for name, val in [
        ("api_key",      "ollama"),
        ("model",        model),
        ("base_url",     endpoint),
        ("do_auto_trim", False),
        ("debug_mode",   False),
    ]:
        if name in base_init_params:
            init_kw[name] = val

    # Store constructor args in registry for pickle reconstruction
    _OLLAMA_LLM_REGISTRY[_registry_id] = dict(
        LLaMEABaseLLM_module = LLaMEABaseLLM.__module__,
        LLaMEABaseLLM_name   = LLaMEABaseLLM.__name__,
        endpoint    = endpoint,
        model       = model,
        timeout     = timeout,
        temperature = temperature,
        registry_id = _registry_id,
    )

    class OllamaLLM(LLaMEABaseLLM):  # type: ignore[misc,valid-type]

        def __init__(self):
            try:
                LLaMEABaseLLM.__init__(self, **init_kw)
            except TypeError:
                try:
                    LLaMEABaseLLM.__init__(self)
                except TypeError:
                    pass
            self.model        = model
            self._endpoint    = endpoint
            self._model       = model
            self._timeout     = timeout
            self._temperature = temperature
            self._registry_id = _registry_id

        # ---- Pickle support -----------------------------------------------
        def __reduce__(self):
            return (_reconstruct_ollama_llm, (self._registry_id,))

        def __getstate__(self):
            return {"_registry_id": self._registry_id}

        def __setstate__(self, state):
            args = _OLLAMA_LLM_REGISTRY.get(state["_registry_id"], {})
            self._registry_id = state["_registry_id"]
            self._endpoint    = args.get("endpoint",    "")
            self._model       = args.get("model",       "")
            self._timeout     = args.get("timeout",     180)
            self._temperature = args.get("temperature", 0.7)
            self.model        = self._model

        # ---- LLaMEA LLM interface -----------------------------------------
        def query(self, session_messages, **kwargs) -> str:
            return self._http(session_messages)

        def get_response(self, prompt, **kwargs) -> str:
            return self._http([{"role": "user", "content": str(prompt)}])

        def chat(self, messages, **kwargs) -> str:
            return self._http(messages)

        def __call__(self, prompt, **kwargs) -> str:
            return self._http([{"role": "user", "content": str(prompt)}])

        @staticmethod
        def _normalize_response(text: str) -> str:
            text = str(text or "").strip()
            if "```" not in text:
                text = f"```python\n{text}\n```"
            return text

        # ---- HTTP core -------------------------------------------------------
        def _http(self, messages, retries: int = 4) -> str:
            payload = json.dumps({
                "model":       self._model,
                "messages":    messages,
                "temperature": self._temperature,
                "stream":      False,
            }).encode("utf-8")
            headers = {
                "Content-Type":  "application/json",
                "Authorization": "Bearer ollama",
            }
            last_exc: Exception | None = None
            for attempt in range(1, retries + 1):
                req = _urllib_request.Request(
                    self._endpoint, data=payload,
                    headers=headers, method="POST"
                )
                try:
                    with _urllib_request.urlopen(req, timeout=self._timeout) as resp:
                        raw = json.loads(resp.read().decode("utf-8"))
                    content = raw["choices"][0]["message"]["content"]
                    return self._normalize_response(content)
                except Exception as exc:
                    last_exc = exc
                    if attempt < retries:
                        time.sleep(2 * attempt)
            raise RuntimeError(
                f"Ollama request failed after {retries} attempts: {last_exc}"
            )

    OllamaLLM.__name__     = "OllamaLLM"
    OllamaLLM.__qualname__ = "OllamaLLM"
    OllamaLLM.__module__   = __name__

    return OllamaLLM()


class _LLaMEAEvaluator:
    """Picklable top-level evaluator used by LLaMEA."""

    def __init__(self, *, task_name: str, lower_bound: float, upper_bound: float,
                 budget: int, eval_seeds: list[int], timeout_seconds: int):
        self.task_name = task_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.budget = budget
        self.eval_seeds = list(eval_seeds)
        self.timeout_seconds = timeout_seconds

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
        individual.set_scores(-mean_objective, feedback=feedback)
        if err:
            individual.error = err
        individual.add_metadata("mean_objective", mean_objective)
        return individual


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------



def _disabled_pickle_archive(self, *args, **kwargs):
    """Disable LLaMEA warm-start archive snapshots during benchmark runs."""
    return None


def _stable_create_log_dir_method(self, name=""):
    """Create one stable log directory per task/seed instead of exp-* folders."""
    log_dir = Path(getattr(self.__class__, "_auto_algo_log_dir"))
    if log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
    (log_dir / "configspace").mkdir(parents=True, exist_ok=True)
    (log_dir / "code").mkdir(parents=True, exist_ok=True)
    return str(log_dir)


class LLaMEAAdapter(BaseAdapter):
    name = "llamea"

    def is_enabled(self) -> bool:
        return self.framework_cfg.get("enabled", False)

    def _repo_path(self) -> Path:
        config_dir = Path(self.global_cfg.get("_config_dir", Path.cwd()))
        repo = self.framework_cfg.get("repo_path", "../LLaMEA")
        return resolve_repo_path(config_dir, str(repo), "llamea")

    def _add_repo_to_path(self) -> None:
        p = self._repo_path()
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
            logger.debug("Added LLaMEA repo to sys.path: %s", p)

    def _build_evaluator(self, task: BenchmarkTask):
        """Evaluator compatible with current LLaMEA API and safe to pickle."""
        eval_seeds = list(self.task_defaults.get("eval_seeds", list(range(11, 21))))
        budget = int(self.task_defaults.get("budget", 1500))
        timeout_seconds = int(self.framework_cfg.get("candidate_timeout_sec", 180))
        return _LLaMEAEvaluator(
            task_name=task.name,
            lower_bound=task.lower_bound,
            upper_bound=task.upper_bound,
            budget=budget,
            eval_seeds=eval_seeds,
            timeout_seconds=timeout_seconds,
        )


    def _llamea_log_dir(self, task: BenchmarkTask, seed: int) -> Path:
        return (
            Path(self.global_cfg.get("_output_dir_resolved", "benchmark_results"))
            / "llamea_logs" / task.name / f"seed{seed}"
        )

    @staticmethod
    def _patch_experiment_logger(log_dir: Path):
        import llamea.loggers as llamea_loggers  # type: ignore

        logger_cls = llamea_loggers.ExperimentLogger
        original_create_log_dir = logger_cls.create_log_dir
        logger_cls._auto_algo_log_dir = str(log_dir)
        logger_cls.create_log_dir = _stable_create_log_dir_method
        return logger_cls, original_create_log_dir

    @staticmethod
    def _patch_pickle_archive(llamea_cls):
        """
        Disable LLaMEA archive pickling for the whole class before constructing
        the optimizer. LLaMEA calls self.pickle_archive() inside __init__, so
        patching after instantiation is too late.
        """
        original = getattr(llamea_cls, "pickle_archive", None)
        if original is None:
            return None
        setattr(llamea_cls, "pickle_archive", _disabled_pickle_archive)
        return llamea_cls, original

    def run(self, task: BenchmarkTask, seed: int) -> dict[str, Any]:
        self._add_repo_to_path()
        t0 = time.perf_counter()
        result: dict[str, Any] = {
            "framework":    self.name,
            "task":         task.name,
            "seed":         seed,
            "best_value":   None,
            "success":      False,
            "error":        None,
            "elapsed_sec":  None,
            "timestamp":    timestamp(),
            "extra":        {},
        }

        try:
            import numpy as np
            from llamea import LLaMEA  # type: ignore

            # Locate LLM base class
            LLaMEABaseLLM = None
            for mod_path in ("llamea.llm", "llamea"):
                try:
                    mod = __import__(mod_path, fromlist=["LLM"])
                    LLaMEABaseLLM = getattr(mod, "LLM", None)
                    if LLaMEABaseLLM is not None:
                        break
                except ImportError:
                    pass
            if LLaMEABaseLLM is None:
                class LLaMEABaseLLM:  # type: ignore[no-redef]
                    def __init__(self, **kwargs): pass

            random.seed(seed)
            np.random.seed(seed)

            cfg        = self.framework_cfg
            framework_log = bool(cfg.get("framework_log", True))
            log_dir = self._llamea_log_dir(task, seed)
            ollama_v1  = self.ollama_base_url.rstrip("/") + "/v1/chat/completions"

            llm = _make_ollama_llm(
                LLaMEABaseLLM,
                endpoint    = ollama_v1,
                model       = self.ollama_model,
                timeout     = int(cfg.get("llm_timeout_sec", 180)),
                temperature = float(cfg.get("temperature", 0.7)),
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
            max_workers = int(cfg.get("max_workers", 1))
            parallel_backend = cfg.get("parallel_backend", "threading" if max_workers <= 1 else "loky")

            logger_patch = None
            if framework_log:
                logger_patch = self._patch_experiment_logger(log_dir)

            want: dict[str, Any] = dict(
                f                = self._build_evaluator(task),
                llm              = llm,
                budget           = int(cfg.get("search_budget", 40)),
                n_parents        = int(cfg.get("n_parents", 4)),
                n_offspring      = int(cfg.get("n_offspring", 8)),
                role_prompt      = role_prompt,
                task_prompt      = task_prompt,
                experiment_name  = f"llamea_{task.name}_seed{seed}",
                elitism          = bool(cfg.get("elitism", True)),
                eval_timeout     = int(cfg.get("eval_timeout", 600)),
                max_workers      = max_workers,
                minimization     = False,
                log              = framework_log,
                parallel_backend = parallel_backend,
                parent_selection = cfg.get("parent_selection", "random"),
                tournament_size  = int(cfg.get("tournament_size", 3)),
            )
            filtered  = {k: v for k, v in want.items() if k in ctor_params}
            pickle_patch = self._patch_pickle_archive(LLaMEA)
            if pickle_patch is not None:
                optimizer_cls, original_pickle_archive = pickle_patch
            else:
                optimizer_cls = None
                original_pickle_archive = None

            optimizer = LLaMEA(**filtered)

            if logger_patch is not None:
                _, original_create_log_dir = logger_patch
                try:
                    best_solution = optimizer.run()
                finally:
                    import llamea.loggers as llamea_loggers  # type: ignore
                    llamea_loggers.ExperimentLogger.create_log_dir = original_create_log_dir
                    if hasattr(llamea_loggers.ExperimentLogger, "_auto_algo_log_dir"):
                        delattr(llamea_loggers.ExperimentLogger, "_auto_algo_log_dir")
                    if optimizer_cls is not None and original_pickle_archive is not None:
                        setattr(optimizer_cls, "pickle_archive", original_pickle_archive)
            else:
                try:
                    best_solution = optimizer.run()
                finally:
                    if optimizer_cls is not None and original_pickle_archive is not None:
                        setattr(optimizer_cls, "pickle_archive", original_pickle_archive)

            if best_solution is None:
                raise RuntimeError("LLaMEA.run() returned None.")

            bi = float(best_solution.fitness)
            bv = float(-bi) if math.isfinite(bi) else PENALTY_OBJECTIVE

            if not math.isfinite(bi) or bv >= PENALTY_OBJECTIVE:
                result["error"]   = (
                    getattr(best_solution, "error", "")
                    or "No valid solution produced."
                )
                result["success"] = False
            else:
                result["best_value"] = bv
                result["success"]    = True

            result["extra"] = {
                "best_internal_score": bi,
                "best_name":     getattr(best_solution, "name",     "")[:200],
                "best_feedback": getattr(best_solution, "feedback", "")[:500],
                "best_error":    getattr(best_solution, "error",    "")[:500],
            }
            if framework_log:
                result["extra"]["log_dir"] = str(log_dir)

        except ImportError as exc:
            msg = f"LLaMEA import failed: {exc}. Check repo_path in config."
            logger.error(msg)
            result["error"] = msg
        except Exception:
            tb = traceback.format_exc()
            logger.error("LLaMEA run failed:\n%s", tb)
            result["error"] = tb

        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return result

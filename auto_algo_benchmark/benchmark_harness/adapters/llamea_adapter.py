"""Adapter for the current LLaMEA repository, hardened for unattended runs."""

from __future__ import annotations

import inspect
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib import error, request

from .base import BaseAdapter
from ..tasks import BenchmarkTask
from ..utils import PENALTY_OBJECTIVE, evaluate_candidate_code, get_logger, timestamp

logger = get_logger(__name__)


class LLaMEAAdapter(BaseAdapter):
    name = "llamea"

    def is_enabled(self) -> bool:
        return self.framework_cfg.get("enabled", False)

    def _repo_path(self) -> Path:
        config_dir = Path(self.global_cfg.get("_config_dir", Path.cwd()))
        repo = Path(self.framework_cfg.get("repo_path", "../LLaMEA"))
        return repo if repo.is_absolute() else (config_dir / repo).resolve()

    def _add_repo_to_path(self) -> None:
        p = self._repo_path()
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
            logger.debug("Added LLaMEA repo to sys.path: %s", p)

    def _build_evaluator(self, task: BenchmarkTask):
        eval_seeds = list(self.task_defaults.get("eval_seeds", list(range(11, 21))))
        budget = int(self.task_defaults.get("budget", 1500))
        timeout_seconds = int(self.framework_cfg.get("candidate_timeout_sec", 180))

        def evaluate_solution(individual, _logger=None):
            mean_objective, feedback, err = evaluate_candidate_code(
                individual.code,
                task_name=task.name,
                lower_bound=task.lower_bound,
                upper_bound=task.upper_bound,
                budget=budget,
                eval_seeds=eval_seeds,
                timeout_seconds=timeout_seconds,
            )
            individual.set_scores(-mean_objective, feedback=feedback)
            if err:
                individual.error = err
            individual.add_metadata("mean_objective", mean_objective)
            return individual

        return evaluate_solution

    def _build_legacy_evaluator(self, task: BenchmarkTask):
        eval_seeds = list(self.task_defaults.get("eval_seeds", list(range(11, 21))))
        budget = int(self.task_defaults.get("budget", 1500))
        timeout_seconds = int(self.framework_cfg.get("candidate_timeout_sec", 180))

        def evaluate_algorithm(solution_code: str):
            mean_objective, feedback, err = evaluate_candidate_code(
                solution_code,
                task_name=task.name,
                lower_bound=task.lower_bound,
                upper_bound=task.upper_bound,
                budget=budget,
                eval_seeds=eval_seeds,
                timeout_seconds=timeout_seconds,
            )
            score = float(-mean_objective) if math.isfinite(mean_objective) else float(-PENALTY_OBJECTIVE)
            message = feedback or ""
            if err:
                message = (message + "\n" if message else "") + f"ERROR: {err}"
            return score, message[:2000]

        return evaluate_algorithm

    def _build_llm(self, LLaMEABaseLLM, cfg: dict):
        """
        Build an Ollama-backed LLM object compatible with the installed LLaMEA version.

        Key fix: we never call super().__init__(api_key=..., model=..., base_url=...)
        because those kwargs don't exist on the current LLaMEA LLM base class.
        Instead we inspect the actual signature and only pass what it accepts,
        then monkey-patch model/endpoint onto self afterwards.
        """
        ollama_endpoint = self.ollama_base_url.rstrip("/") + "/v1/chat/completions"
        ollama_model    = self.ollama_model
        llm_timeout     = int(cfg.get("llm_timeout_sec", 180))
        temperature     = float(cfg.get("temperature", 0.7))

        base_init_params = inspect.signature(LLaMEABaseLLM.__init__).parameters

        class _OllamaLLM(LLaMEABaseLLM):
            """Thin Ollama shim — pure urllib, zero extra deps."""

            def __init__(self):
                # Only pass kwargs that the base class actually accepts
                init_kw = {}
                for name, default in [
                    ("do_auto_trim", False),
                    ("debug_mode",   False),
                    ("api_key",      "ollama"),
                    ("model",        ollama_model),
                    ("base_url",     ollama_endpoint),
                ]:
                    if name in base_init_params:
                        init_kw[name] = default
                try:
                    super().__init__(**init_kw)
                except TypeError:
                    try:
                        super().__init__()
                    except TypeError:
                        pass
                # Ensure these attributes always exist regardless of base class
                self.model        = ollama_model
                self._ep          = ollama_endpoint
                self._to          = llm_timeout
                self._temp        = temperature

            # Current LLaMEA interface
            def query(self, session_messages, **kwargs):
                return self._http(session_messages)

            # Older / BLADE interface
            def get_response(self, prompt, **kwargs):
                return self._http([{"role": "user", "content": str(prompt)}])

            def chat(self, messages, **kwargs):
                return self._http(messages)

            # Some versions call __call__
            def __call__(self, prompt, **kwargs):
                return self._http([{"role": "user", "content": str(prompt)}])

            def _http(self, messages, retries: int = 4) -> str:
                payload = json.dumps({
                    "model":       ollama_model,
                    "messages":    messages,
                    "temperature": temperature,
                    "stream":      False,
                }).encode("utf-8")
                headers = {
                    "Content-Type":  "application/json",
                    "Authorization": "Bearer ollama",
                }
                last_exc: Exception | None = None
                for attempt in range(1, retries + 1):
                    req = request.Request(
                        ollama_endpoint, data=payload,
                        headers=headers, method="POST"
                    )
                    try:
                        with request.urlopen(req, timeout=llm_timeout) as resp:
                            raw = json.loads(resp.read().decode("utf-8"))
                        return raw["choices"][0]["message"]["content"]
                    except Exception as exc:
                        last_exc = exc
                        if attempt < retries:
                            time.sleep(2 * attempt)
                raise RuntimeError(
                    f"Ollama request failed after {retries} attempts: {last_exc}"
                )

        return _OllamaLLM()

    def run(self, task: BenchmarkTask, seed: int) -> dict[str, Any]:
        self._add_repo_to_path()
        t0 = time.perf_counter()
        result: dict[str, Any] = {
            "framework": self.name,
            "task":      task.name,
            "seed":      seed,
            "best_value": None,
            "success":   False,
            "error":     None,
            "elapsed_sec": None,
            "timestamp": timestamp(),
            "extra":     {},
        }

        try:
            import numpy as np
            from llamea import LLaMEA  # type: ignore

            # Locate the LLM base class — path differs across LLaMEA versions
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
                # Absolute fallback — create a trivial base so _OllamaLLM still works
                class LLaMEABaseLLM:  # type: ignore[no-redef]
                    def __init__(self, **kwargs): pass

            random.seed(seed)
            np.random.seed(seed)

            cfg        = self.framework_cfg
            llm        = self._build_llm(LLaMEABaseLLM, cfg)
            ctor_params = inspect.signature(LLaMEA.__init__).parameters
            logger.debug("LLaMEA.__init__ params: %s", list(ctor_params.keys()))

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

            # ----------------------------------------------------------------
            # Detect API variant by inspecting LLaMEA.__init__ parameters
            # ----------------------------------------------------------------
            has_llm_param = "llm" in ctor_params
            has_f_param   = "f"   in ctor_params

            if has_llm_param and has_f_param:
                # ---- Current main-branch: LLaMEA(f, llm, budget, ...) ----
                want = dict(
                    f            = self._build_evaluator(task),
                    llm          = llm,
                    budget       = int(cfg.get("search_budget", 40)),
                    n_parents    = int(cfg.get("n_parents", 4)),
                    n_offspring  = int(cfg.get("n_offspring", 8)),
                    role_prompt  = role_prompt,
                    task_prompt  = task_prompt,
                    experiment_name = f"llamea_{task.name}_seed{seed}",
                    elitism      = bool(cfg.get("elitism", True)),
                    eval_timeout = int(cfg.get("eval_timeout", 600)),
                    max_workers  = int(cfg.get("max_workers", 1)),
                    minimization = False,
                    log          = bool(cfg.get("framework_log", True)),
                    # Optional — only injected if constructor accepts them
                    parallel_backend = cfg.get("parallel_backend", "loky"),
                    parent_selection = cfg.get("parent_selection", "random"),
                    tournament_size  = int(cfg.get("tournament_size", 3)),
                    elitist          = bool(cfg.get("elitism", True)),
                )
                optimizer    = LLaMEA(**{k: v for k, v in want.items() if k in ctor_params})
                best_solution = optimizer.run()
                if best_solution is None:
                    raise RuntimeError("LLaMEA.run() returned None.")

                bi  = float(best_solution.fitness)
                bv  = float(-bi) if math.isfinite(bi) else PENALTY_OBJECTIVE
                if not math.isfinite(bi) or bv >= PENALTY_OBJECTIVE:
                    result["error"]   = getattr(best_solution, "error", "") or "No valid solution."
                    result["success"] = False
                else:
                    result["best_value"] = bv
                    result["success"]    = True
                result["extra"] = {
                    "api_variant":        "current",
                    "best_internal_score": bi,
                    "best_name":    getattr(best_solution, "name",     "")[:200],
                    "best_feedback": getattr(best_solution, "feedback", "")[:500],
                }

            elif has_llm_param and not has_f_param:
                # ---- BLADE-style: LLaMEA(llm, budget=..., n_parents=...) ----
                # Evaluation function is NOT passed to __init__
                want = dict(
                    llm         = llm,
                    budget      = int(cfg.get("search_budget", 40)),
                    n_parents   = int(cfg.get("n_parents", 4)),
                    n_offspring = int(cfg.get("n_offspring", 8)),
                    elitism     = bool(cfg.get("elitism", True)),
                    name        = f"llamea_{task.name}_seed{seed}",
                    role_prompt = role_prompt,
                    task_prompt = task_prompt,
                )
                optimizer   = LLaMEA(**{k: v for k, v in want.items() if k in ctor_params})
                legacy_eval = self._build_legacy_evaluator(task)
                run_params  = inspect.signature(optimizer.run).parameters
                run_output  = optimizer.run(legacy_eval) if len(run_params) > 1 else optimizer.run()
                bi, bv = _extract_score(run_output)
                if bv is None or bv >= PENALTY_OBJECTIVE:
                    result["error"]   = "BLADE LLaMEA produced no valid solution."
                    result["success"] = False
                else:
                    result["best_value"] = bv
                    result["success"]    = True
                result["extra"] = {"api_variant": "blade", "best_internal_score": bi}

            else:
                # ---- Legacy: LLaMEA(model, api_key, base_url, budget, ...) ----
                base_url = self.ollama_base_url.rstrip("/") + "/v1"
                want = dict(
                    model       = self.ollama_model,
                    api_key     = "ollama",
                    base_url    = base_url,
                    budget      = int(cfg.get("search_budget", 40)),
                    n_parents   = int(cfg.get("n_parents", 4)),
                    n_offspring = int(cfg.get("n_offspring", 8)),
                    role_prompt = role_prompt + task_prompt,
                    task_prompt = task_prompt,
                    experiment_name = f"llamea_{task.name}_seed{seed}",
                    eval_timeout    = int(cfg.get("eval_timeout", 600)),
                    max_workers     = int(cfg.get("max_workers", 1)),
                    elitist         = bool(cfg.get("elitism", True)),
                    elitism         = bool(cfg.get("elitism", True)),
                    parallel_backend = cfg.get("parallel_backend", "loky"),
                    parent_selection = cfg.get("parent_selection", "random"),
                    tournament_size  = int(cfg.get("tournament_size", 3)),
                )
                optimizer   = LLaMEA(**{k: v for k, v in want.items() if k in ctor_params})
                legacy_eval = self._build_legacy_evaluator(task)
                run_params  = inspect.signature(optimizer.run).parameters
                run_output  = optimizer.run(legacy_eval) if len(run_params) > 1 else optimizer.run()
                bi, bv = _extract_score(run_output)
                if bv is None or bv >= PENALTY_OBJECTIVE:
                    result["error"]   = "Legacy LLaMEA produced no valid solution."
                    result["success"] = False
                else:
                    result["best_value"] = bv
                    result["success"]    = True
                result["extra"] = {"api_variant": "legacy", "best_internal_score": bi}

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


# ---------------------------------------------------------------------------

def _extract_score(run_output) -> tuple[float | None, float | None]:
    """Return (internal_score, best_value) from whatever run() returned."""
    best_score = None
    if isinstance(run_output, tuple):
        if len(run_output) >= 2:
            best_score = run_output[1]
    elif hasattr(run_output, "fitness"):
        best_score = getattr(run_output, "fitness")

    if best_score is None:
        return None, None
    try:
        bi = float(best_score)
    except (TypeError, ValueError):
        return None, None

    bv = float(-bi) if math.isfinite(bi) else PENALTY_OBJECTIVE
    return bi, bv

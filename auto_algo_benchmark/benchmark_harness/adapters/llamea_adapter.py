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

        def evaluate_solution(individual, _logger):
            mean_objective, feedback, error = evaluate_candidate_code(
                individual.code,
                task_name=task.name,
                lower_bound=task.lower_bound,
                upper_bound=task.upper_bound,
                budget=budget,
                eval_seeds=eval_seeds,
                timeout_seconds=timeout_seconds,
            )
            # LLaMEA maximises unless minimization=True. We use negative objective.
            individual.set_scores(-mean_objective, feedback=feedback)
            if error:
                individual.error = error
            individual.add_metadata("mean_objective", mean_objective)
            return individual

        return evaluate_solution

    def _build_legacy_evaluator(self, task: BenchmarkTask):
        eval_seeds = list(self.task_defaults.get("eval_seeds", list(range(11, 21))))
        budget = int(self.task_defaults.get("budget", 1500))
        timeout_seconds = int(self.framework_cfg.get("candidate_timeout_sec", 180))

        def evaluate_algorithm(solution_code: str):
            mean_objective, feedback, error = evaluate_candidate_code(
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
            if error:
                message = (message + "\n" if message else "") + f"ERROR: {error}"
            return score, message[:2000]

        return evaluate_algorithm

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

        try:
            import numpy as np
            from llamea import LLaMEA  # type: ignore
            from llamea.llm import LLM as LLaMEABaseLLM  # type: ignore

            random.seed(seed)
            np.random.seed(seed)

            cfg = self.framework_cfg
            ctor_params = inspect.signature(LLaMEA.__init__).parameters
            uses_current_api = "llm" in ctor_params and "f" in ctor_params

            if uses_current_api:
                base_url = self.ollama_base_url.rstrip("/") + "/v1/chat/completions"

                class _OllamaHTTPAdapter(LLaMEABaseLLM):
                    def __init__(self, *, model: str, endpoint: str, timeout: int, temperature: float = 0.7):
                        super().__init__(api_key="ollama", model=model, base_url=endpoint)
                        self.endpoint = endpoint
                        self.timeout = timeout
                        self.temperature = temperature

                    def query(self, session_messages, max_retries: int = 4, default_delay: int = 2):
                        payload = {
                            "model": self.model,
                            "messages": session_messages,
                            "temperature": self.temperature,
                            "stream": False,
                        }
                        data = json.dumps(payload).encode("utf-8")
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": "Bearer ollama",
                        }
                        last_error = None
                        for attempt in range(1, max_retries + 1):
                            req = request.Request(self.endpoint, data=data, headers=headers, method="POST")
                            try:
                                with request.urlopen(req, timeout=self.timeout) as resp:
                                    raw = json.loads(resp.read().decode("utf-8"))
                                return raw["choices"][0]["message"]["content"]
                            except Exception as exc:  # pragma: no cover - network dependent
                                last_error = exc
                                if attempt == max_retries:
                                    break
                                time.sleep(default_delay * attempt)
                        raise RuntimeError(f"Ollama request failed: {last_error}")

                llm = _OllamaHTTPAdapter(
                    model=self.ollama_model,
                    endpoint=base_url,
                    timeout=int(cfg.get("llm_timeout_sec", 180)),
                    temperature=float(cfg.get("temperature", 0.7)),
                )

                role_prompt = (
                    "You are an expert in black-box continuous optimisation. "
                    "Write a single sequential Python function only. "
                    "Do not use threads, multiprocessing, subprocesses, files, network calls, eval, exec, or imports beyond numpy.\n\n"
                )
                task_prompt = (
                    f"Task: {task.description}\n\n"
                    "Return Python code only. Implement:\n"
                    "def algorithm(func, dim, lb, ub, budget, rng):\n"
                    "    ...\n"
                    "The function must return the best 1-D numpy array found.\n"
                    "Keep it numerically stable. Clip candidate vectors to [lb, ub].\n"
                )

                current_kwargs = {
                    "f": self._build_evaluator(task),
                    "llm": llm,
                    "budget": int(cfg.get("search_budget", 40)),
                    "n_parents": int(cfg.get("n_parents", 4)),
                    "n_offspring": int(cfg.get("n_offspring", 8)),
                    "role_prompt": role_prompt,
                    "task_prompt": task_prompt,
                    "experiment_name": f"llamea_{task.name}_seed{seed}",
                    "elitism": bool(cfg.get("elitism", True)),
                    "eval_timeout": int(cfg.get("eval_timeout", 600)),
                    "max_workers": int(cfg.get("max_workers", 1)),
                    "parallel_backend": cfg.get("parallel_backend", "loky"),
                    "minimization": False,
                    "log": bool(cfg.get("framework_log", True)),
                    "parent_selection": cfg.get("parent_selection", "random"),
                    "tournament_size": int(cfg.get("tournament_size", 3)),
                }
                optimizer = LLaMEA(**{k: v for k, v in current_kwargs.items() if k in ctor_params})
                best_solution = optimizer.run()
                if best_solution is None:
                    raise RuntimeError("LLaMEA returned no solution.")

                best_internal = float(best_solution.fitness)
                best_value = float(-best_internal) if math.isfinite(best_internal) else PENALTY_OBJECTIVE
                best_error = getattr(best_solution, "error", "") or "Candidate evaluation failed or no valid solution was produced."
                if (not math.isfinite(best_internal)) or best_value >= PENALTY_OBJECTIVE:
                    result["error"] = best_error
                    result["success"] = False
                else:
                    result["best_value"] = best_value
                    result["success"] = True

                result["extra"] = {
                    "api_variant": "current",
                    "best_internal_score": best_internal,
                    "best_name": getattr(best_solution, "name", ""),
                    "best_feedback": getattr(best_solution, "feedback", "")[:1000],
                    "best_error": getattr(best_solution, "error", "")[:1000],
                }
            else:
                base_url = self.ollama_base_url.rstrip("/") + "/v1"
                legacy_kwargs = {
                    "model": self.ollama_model,
                    "api_key": "ollama",
                    "base_url": base_url,
                    "budget": int(cfg.get("search_budget", 40)),
                    "n_parents": int(cfg.get("n_parents", 4)),
                    "n_offspring": int(cfg.get("n_offspring", 8)),
                    "role_prompt": (
                        "You are an expert in black-box continuous optimisation. "
                        "Write a single sequential Python function only. "
                        "Do not use threads, multiprocessing, subprocesses, files, network calls, eval, exec, or imports beyond numpy.\n\n"
                        f"Task: {task.description}\n\n"
                    ),
                    "task_prompt": f"Implement an optimisation algorithm for: {task.description}",
                    "experiment_name": f"llamea_{task.name}_seed{seed}",
                    "elitist": bool(cfg.get("elitism", True)),
                    "elitism": bool(cfg.get("elitism", True)),
                    "eval_timeout": int(cfg.get("eval_timeout", 600)),
                    "max_workers": int(cfg.get("max_workers", 1)),
                    "parallel_backend": cfg.get("parallel_backend", "loky"),
                    "parent_selection": cfg.get("parent_selection", "random"),
                    "tournament_size": int(cfg.get("tournament_size", 3)),
                }
                optimizer = LLaMEA(**{k: v for k, v in legacy_kwargs.items() if k in ctor_params})
                legacy_eval = self._build_legacy_evaluator(task)
                run_params = inspect.signature(optimizer.run).parameters
                if len(run_params) >= 1:
                    run_output = optimizer.run(legacy_eval)
                else:
                    run_output = optimizer.run()

                best_score = None
                extra = {"api_variant": "legacy"}
                if isinstance(run_output, tuple):
                    if len(run_output) >= 2:
                        best_score = run_output[1]
                    if len(run_output) >= 3:
                        extra["log_summary"] = str(run_output[2])[:1000]
                elif hasattr(run_output, "fitness"):
                    best_score = getattr(run_output, "fitness")
                    extra["best_name"] = getattr(run_output, "name", "")
                    extra["best_feedback"] = getattr(run_output, "feedback", "")[:1000]
                if best_score is None:
                    raise RuntimeError("Legacy LLaMEA run finished without returning a best score.")

                best_internal = float(best_score)
                best_value = float(-best_internal) if math.isfinite(best_internal) else PENALTY_OBJECTIVE
                if (not math.isfinite(best_internal)) or best_value >= PENALTY_OBJECTIVE:
                    result["error"] = extra.get("log_summary") or "Legacy LLaMEA produced no valid solution."
                    result["success"] = False
                else:
                    result["best_value"] = best_value
                    result["success"] = True
                extra["best_internal_score"] = best_internal
                result["extra"] = extra

        except ImportError as exc:
            msg = (
                f"LLaMEA import failed: {exc}. Make sure repo_path points to the current LLaMEA repo."
            )
            logger.error(msg)
            result["error"] = msg
        except Exception:
            tb = traceback.format_exc()
            logger.error("LLaMEA run failed:\n%s", tb)
            result["error"] = tb

        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return result

"""Adapter for the current LLaMEA repository, hardened for unattended runs."""

from __future__ import annotations

import json
import math
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
            from llamea import LLaMEA  # type: ignore
            from llamea.llm import LLM as LLaMEABaseLLM  # type: ignore

            cfg = self.framework_cfg
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

            optimizer = LLaMEA(
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
                parallel_backend=cfg.get("parallel_backend", "loky"),
                minimization=False,
                log=bool(cfg.get("framework_log", True)),
                parent_selection=cfg.get("parent_selection", "random"),
                tournament_size=int(cfg.get("tournament_size", 3)),
            )

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
                "best_internal_score": best_internal,
                "best_name": getattr(best_solution, "name", ""),
                "best_feedback": getattr(best_solution, "feedback", "")[:1000],
                "best_error": getattr(best_solution, "error", "")[:1000],
            }

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

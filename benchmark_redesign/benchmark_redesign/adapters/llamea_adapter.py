from __future__ import annotations

import ast
import re
import sys
import time
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..ollama_client import OllamaConfig, RobustOllamaClient
from ..prompts import llm_system_prompt, llm_user_prompt
from ..safe_eval import evaluate_code_on_task
from ..tasks import task_from_name, score_from_best_f, PENALTY_BEST_F


class LLAMEAAdapter(FrameworkAdapter):
    def __init__(self, framework_cfg: dict[str, Any], global_cfg: dict[str, Any]):
        super().__init__("llamea", framework_cfg, global_cfg)

    @staticmethod
    def _extract_code(text: str) -> str:
        text = text.strip()
        fence = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        return (fence.group(1) if fence else text).strip()

    @staticmethod
    def _extract_name_from_code(code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "solve"
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return node.name
        return "solve"

    def run(self, task_name: str, seed: int) -> dict[str, Any]:
        repo = Path(self.framework_cfg["repo"]).expanduser().resolve()
        sys.path.insert(0, str(repo))

        from llamea import LLaMEA  # type: ignore
        from llamea.llm import LLM as LLaMEABaseLLM  # type: ignore
        from llamea.solution import Solution  # type: ignore

        task = task_from_name(task_name, self.global_cfg["task_defaults"])
        ollama_cfg = OllamaConfig(
            model=self.global_cfg["ollama"]["model"],
            base_url=self.global_cfg["ollama"]["base_url"],
            connect_timeout_seconds=float(self.global_cfg["ollama"].get("connect_timeout_seconds", 10)),
            read_timeout_seconds=float(self.global_cfg["ollama"].get("read_timeout_seconds", 90)),
            hard_timeout_seconds=float(self.framework_cfg.get("llm_timeout_seconds", self.global_cfg["ollama"].get("hard_timeout_seconds", 120))),
            retries=int(self.global_cfg["ollama"].get("retries", 1)),
            temperature=float(self.framework_cfg.get("temperature", 0.2)),
        )
        client = RobustOllamaClient(ollama_cfg)

        class PatchedOllamaLLM(LLaMEABaseLLM):
            def __init__(self):
                super().__init__(api_key="", model=ollama_cfg.model, base_url=ollama_cfg.base_url)

            def query(self, session_messages: list):
                return client.chat(session_messages, hard_timeout_seconds=ollama_cfg.hard_timeout_seconds)

            def sample_solution(self, session_messages, parent_ids=None, HPO=False, base_code=None, diff_mode=False):
                if parent_ids is None:
                    parent_ids = []
                raw_message = self.query(session_messages)
                code = LLAMEAAdapter._extract_code(raw_message)
                name = LLAMEAAdapter._extract_name_from_code(code)
                return Solution(code=code, name=name, description="", parent_ids=parent_ids)

        def evaluate_solution(solution, _logger=None):
            result = evaluate_code_on_task(solution.code, task)
            solution.add_metadata("evaluation", result.details)
            solution.add_metadata("task_name", task.name)
            if result.status == "ok":
                solution.set_scores(result.score, f"mean_best_f={result.mean_best_f:.6g}")
            else:
                solution.set_scores(score_from_best_f(PENALTY_BEST_F), result.error)
            return solution

        llm = PatchedOllamaLLM()
        algo = LLaMEA(
            f=evaluate_solution,
            llm=llm,
            n_parents=int(self.framework_cfg.get("n_parents", 1)),
            n_offspring=int(self.framework_cfg.get("n_offspring", 1)),
            budget=int(self.framework_cfg.get("evolution_budget", 10)),
            eval_timeout=int(self.framework_cfg.get("evaluation_timeout_seconds", 20)),
            max_workers=int(self.framework_cfg.get("max_workers", 1)),
            parallel_backend=str(self.framework_cfg.get("parallel_backend", "loky")),
            task_prompt=llm_user_prompt(task),
            role_prompt=llm_system_prompt(task),
            example_prompt="",
            log=False,
            minimization=False,
            elitism=True,
        )

        start = time.time()
        best = algo.run()
        duration = time.time() - start
        best_eval = best.get_metadata("evaluation") if hasattr(best, "get_metadata") else None
        mean_best_f = None
        if isinstance(best_eval, dict):
            per_seed = [x for x in best_eval.get("per_seed", []) if x.get("status") == "ok"]
            if per_seed:
                mean_best_f = sum(float(x["best_f"]) for x in per_seed) / len(per_seed)

        return {
            "framework": self.framework_name,
            "task": task.name,
            "seed": seed,
            "status": "ok" if best is not None else "failed",
            "score": float(best.fitness) if best is not None else score_from_best_f(PENALTY_BEST_F),
            "best_f": float(mean_best_f if mean_best_f is not None else PENALTY_BEST_F),
            "duration_seconds": duration,
            "best_code": getattr(best, "code", ""),
            "best_name": getattr(best, "name", "solve"),
            "run_history_size": len(getattr(algo, "run_history", [])),
        }

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..ollama_client import OllamaConfig, RobustOllamaClient
from ..prompts import llm4ad_task_description, llm4ad_template_program
from ..safe_eval import evaluate_code_on_task
from ..tasks import task_from_name, score_from_best_f, PENALTY_BEST_F


class LLM4ADEoHAdapter(FrameworkAdapter):
    def __init__(self, framework_cfg: dict[str, Any], global_cfg: dict[str, Any]):
        super().__init__("llm4ad_eoh", framework_cfg, global_cfg)

    def run(self, task_name: str, seed: int) -> dict[str, Any]:
        repo = Path(self.framework_cfg["repo"]).expanduser().resolve()
        sys.path.insert(0, str(repo))

        from llm4ad.base import Evaluation, LLM, TextFunctionProgramConverter  # type: ignore
        from llm4ad.method.eoh import EoH  # type: ignore

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

        class PatchedLLM(LLM):
            def __init__(self):
                super().__init__(do_auto_trim=False, debug_mode=False)

            def draw_sample(self, prompt: str, *args, **kwargs) -> str:
                return client.chat(
                    [{"role": "user", "content": prompt}],
                    hard_timeout_seconds=ollama_cfg.hard_timeout_seconds,
                )

            def close(self):
                return None

        class OptimizerEvaluation(Evaluation):
            def __init__(self):
                super().__init__(
                    template_program=llm4ad_template_program(),
                    task_description=llm4ad_task_description(task),
                    use_numba_accelerate=False,
                    use_protected_div=False,
                    timeout_seconds=int(self_outer.framework_cfg.get("evaluation_timeout_seconds", 20)),
                )

            def evaluate_program(self, program_str: str, callable_func: callable):
                try:
                    result = evaluate_code_on_task(program_str, task)
                except Exception:
                    return score_from_best_f(PENALTY_BEST_F)
                if result.status != "ok":
                    return score_from_best_f(PENALTY_BEST_F)
                return float(result.score)

        self_outer = self
        llm = PatchedLLM()
        evaluator = OptimizerEvaluation()
        method = EoH(
            llm=llm,
            evaluation=evaluator,
            profiler=None,
            max_generations=int(self.framework_cfg.get("max_generations", 8)),
            max_sample_nums=int(self.framework_cfg.get("max_sample_nums", 16)),
            pop_size=int(self.framework_cfg.get("pop_size", 4)),
            selection_num=int(self.framework_cfg.get("selection_num", 2)),
            use_e2_operator=bool(self.framework_cfg.get("use_e2_operator", True)),
            use_m1_operator=bool(self.framework_cfg.get("use_m1_operator", True)),
            use_m2_operator=bool(self.framework_cfg.get("use_m2_operator", True)),
            num_samplers=int(self.framework_cfg.get("num_samplers", 1)),
            num_evaluators=int(self.framework_cfg.get("num_evaluators", 1)),
            debug_mode=False,
            multi_thread_or_process_eval=str(self.framework_cfg.get("multi_thread_or_process_eval", "thread")),
        )

        start = time.time()
        method.run()
        duration = time.time() - start

        best = None
        population = getattr(method, "_population", None)
        if population is not None and getattr(population, "population", None):
            ranked = sorted(population.population, key=lambda item: getattr(item, "score", float("-inf")), reverse=True)
            if ranked:
                best = ranked[0]

        best_code = ""
        best_eval = None
        best_f = PENALTY_BEST_F
        if best is not None:
            try:
                program = TextFunctionProgramConverter.function_to_program(best, llm4ad_template_program())
                best_code = str(program) if program is not None else str(best)
            except Exception:
                best_code = str(best)
        if best_code:
            try:
                best_eval = evaluate_code_on_task(best_code, task)
                if best_eval.status == "ok":
                    best_f = best_eval.mean_best_f
            except Exception:
                best_eval = None

        return {
            "framework": self.framework_name,
            "task": task.name,
            "seed": seed,
            "status": "ok" if best is not None else "failed",
            "score": float(getattr(best, "score", score_from_best_f(PENALTY_BEST_F))),
            "best_f": float(best_f),
            "duration_seconds": duration,
            "best_code": best_code,
            "best_name": getattr(best, "name", "solve") if best is not None else "solve",
            "generation": getattr(population, "generation", 0) if population is not None else 0,
            "population_size": len(getattr(population, "population", [])) if population is not None else 0,
        }

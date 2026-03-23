from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..tasks import BenchmarkTask, evaluate_solver_callable, llm4ad_task_description, llm4ad_template_program, score_from_best_f
from ..utils import ResourceMonitor, RunSummary, ensure_dir, import_from_repo, safe_float


PENALTY_SCORE = score_from_best_f(1e12)


class LLM4ADAdapter(FrameworkAdapter):
    def __init__(self, framework_cfg: dict[str, Any], global_cfg: dict[str, Any], output_dir: str | Path):
        super().__init__("llm4ad", framework_cfg, global_cfg, output_dir)

    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict[str, Any]]]:
        import_from_repo(self.framework_cfg["repo_path"], "llm4ad")

        from llm4ad.base import Evaluation  # type: ignore
        from llm4ad.method.eoh import EoH, EoHProfiler  # type: ignore
        from llm4ad.tools.llm.local_ollama import LocalOllamaLLM  # type: ignore

        run_dir = ensure_dir(self.output_dir / "artifacts" / "llm4ad" / task.name / f"seed_{seed}")

        timeout_seconds = int(self.framework_cfg.get("timeout_seconds", 1200))
        max_sample_nums = int(self.framework_cfg.get("max_sample_nums", 12))
        max_generations = int(self.framework_cfg.get("max_generations", 6))
        pop_size = int(self.framework_cfg.get("pop_size", 4))
        selection_num = int(self.framework_cfg.get("selection_num", 2))
        num_samplers = int(self.framework_cfg.get("num_samplers", 1))
        num_evaluators = int(self.framework_cfg.get("num_evaluators", 1))
        model_name = self.global_cfg["ollama"]["model"]
        base_url = self.global_cfg["ollama"]["base_url"]
        safe_evaluate = bool(self.framework_cfg.get("safe_evaluate", True))
        daemon_eval_process = bool(self.framework_cfg.get("daemon_eval_process", False))
        fork_proc = self.framework_cfg.get("fork_proc", "auto")

        lock = threading.Lock()
        eval_records: list[dict[str, Any]] = []

        class ContinuousOptimizationEvaluation(Evaluation):
            def __init__(self):
                super().__init__(
                    template_program=llm4ad_template_program(),
                    task_description=llm4ad_task_description(task),
                    timeout_seconds=timeout_seconds,
                    random_seed=seed,
                    safe_evaluate=safe_evaluate,
                    daemon_eval_process=daemon_eval_process,
                    fork_proc=fork_proc,
                )

            def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
                try:
                    result = evaluate_solver_callable(callable_func, task)
                    fitness = float(result["fitness"])
                    raw_objective = float(result["raw_objective_mean"])
                    fail_reason = ""
                except Exception as exc:
                    fitness = float(PENALTY_SCORE)
                    raw_objective = 1e12
                    fail_reason = f"{type(exc).__name__}: {exc}"
                with lock:
                    eval_records.append(
                        {
                            "candidate_score": fitness,
                            "raw_objective_mean": raw_objective,
                            "is_valid_candidate": fitness > PENALTY_SCORE + 1e-9,
                            "fail_reason": fail_reason,
                        }
                    )
                return fitness

        progress_rows: list[dict[str, Any]] = []
        best_score = None
        best_raw = None
        status = "success"
        notes = ""
        candidates_evaluated = 0
        runtime_sec = 0.0
        peak_rss_mb = None

        try:
            llm = LocalOllamaLLM(model_name=model_name, base_url=base_url)
            evaluation = ContinuousOptimizationEvaluation()
            profiler = EoHProfiler(log_dir=str(run_dir), log_style="simple", create_random_path=False)

            with ResourceMonitor() as monitor:
                method = EoH(
                    llm=llm,
                    evaluation=evaluation,
                    profiler=profiler,
                    max_sample_nums=max_sample_nums,
                    max_generations=max_generations,
                    pop_size=pop_size,
                    selection_num=selection_num,
                    num_samplers=num_samplers,
                    num_evaluators=num_evaluators,
                    debug_mode=False,
                )
                method.run()
            runtime_sec = monitor.runtime_sec
            peak_rss_mb = monitor.peak_rss_mb

            with (Path(run_dir) / "eval_records.json").open("w", encoding="utf-8") as f:
                json.dump(eval_records, f, indent=2)

            running_best = float("-inf")
            for idx, item in enumerate(eval_records, start=1):
                score = safe_float(item.get("candidate_score"))
                if score is None:
                    score = float(PENALTY_SCORE)
                running_best = max(running_best, score)
                progress_rows.append(
                    {
                        "framework": "llm4ad",
                        "benchmark": task.name,
                        "seed": seed,
                        "sample_index": idx,
                        "elapsed_sec": None,
                        "candidate_score": score,
                        "best_so_far": running_best,
                        "is_valid_candidate": bool(item.get("is_valid_candidate", False)),
                        "fail_reason": str(item.get("fail_reason", "") or ""),
                    }
                )
            candidates_evaluated = len(eval_records)
            if eval_records:
                best_item = max(eval_records, key=lambda x: safe_float(x.get("candidate_score")) if safe_float(x.get("candidate_score")) is not None else float(PENALTY_SCORE))
                best_score = safe_float(best_item.get("candidate_score"))
                best_raw = safe_float(best_item.get("raw_objective_mean"))
            if best_score is None or best_score <= PENALTY_SCORE + 1e-9:
                status = "no_valid_candidate"
                notes = "All generated candidates fell back to the penalty baseline."
        except Exception as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        summary = RunSummary(
            framework="llm4ad",
            benchmark=task.name,
            seed=seed,
            status=status,
            best_search_score=best_score,
            raw_objective_mean=best_raw,
            runtime_sec=runtime_sec,
            peak_rss_mb=peak_rss_mb,
            candidates_evaluated=candidates_evaluated,
            artifact_dir=str(Path(run_dir).resolve()),
            notes=notes,
        )
        return summary, progress_rows
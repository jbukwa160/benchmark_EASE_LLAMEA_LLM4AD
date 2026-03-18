from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..tasks import BenchmarkTask, evaluate_solver_callable, llm4ad_task_description, llm4ad_template_program
from ..utils import ResourceMonitor, RunSummary, case_insensitive_get, ensure_dir, import_from_repo, safe_float


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

        class ContinuousOptimizationEvaluation(Evaluation):
            def __init__(self):
                super().__init__(
                    template_program=llm4ad_template_program(),
                    task_description=llm4ad_task_description(task),
                    timeout_seconds=timeout_seconds,
                    random_seed=None,
                    safe_evaluate=True,
                )

            def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
                result = evaluate_solver_callable(callable_func, task)
                return result["fitness"]

        progress_rows: list[dict[str, Any]] = []
        best_score = None
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

            samples_dir = Path(run_dir) / "samples"
            history_items: list[dict[str, Any]] = []
            if samples_dir.exists():
                for path in sorted(samples_dir.glob("samples_*.json")):
                    if path.name == "samples_best.json":
                        continue
                    with path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, list):
                        history_items.extend(payload)

            history_items.sort(key=lambda x: int(case_insensitive_get(x, "sample_order", 0)))
            running_best = float("-inf")
            for item in history_items:
                idx = int(case_insensitive_get(item, "sample_order", 0))
                score = safe_float(case_insensitive_get(item, "score"))
                if score is None:
                    score = float("-inf")
                running_best = max(running_best, score)
                progress_rows.append({
                    "framework": "llm4ad",
                    "benchmark": task.name,
                    "seed": seed,
                    "sample_index": idx,
                    "elapsed_sec": None,
                    "candidate_score": score,
                    "best_so_far": running_best,
                })
            candidates_evaluated = len(history_items)
            if progress_rows:
                best_score = progress_rows[-1]["best_so_far"]
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
            raw_objective_mean=None,
            runtime_sec=runtime_sec,
            peak_rss_mb=peak_rss_mb,
            candidates_evaluated=candidates_evaluated,
            artifact_dir=str(Path(run_dir).resolve()),
            notes=notes,
        )
        return summary, progress_rows

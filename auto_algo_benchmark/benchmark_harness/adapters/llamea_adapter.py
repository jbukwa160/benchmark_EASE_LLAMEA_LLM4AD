from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..tasks import (
    BenchmarkTask,
    evaluate_solver_callable,
    llamea_task_prompt,
    validate_python_code,
)
from ..utils import ResourceMonitor, RunSummary, ensure_dir, import_from_repo, pushd, safe_float


class LLAMEAAdapter(FrameworkAdapter):
    def __init__(self, framework_cfg: dict[str, Any], global_cfg: dict[str, Any], output_dir: str | Path):
        super().__init__("llamea", framework_cfg, global_cfg, output_dir)

    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict[str, Any]]]:
        import os

        import_from_repo(self.framework_cfg["repo_path"], "llamea")
        from llamea import LLaMEA, Ollama_LLM  # type: ignore

        run_dir = ensure_dir(self.output_dir / "artifacts" / "llamea" / task.name / f"seed_{seed}")
        os.environ["OLLAMA_HOST"] = self.global_cfg["ollama"]["base_url"]

        budget = int(self.framework_cfg.get("search_budget", 12))
        n_parents = int(self.framework_cfg.get("n_parents", 1))
        n_offspring = int(self.framework_cfg.get("n_offspring", 1))
        max_workers = int(self.framework_cfg.get("max_workers", 1))
        eval_timeout = int(self.framework_cfg.get("eval_timeout", 1200))
        model_name = self.global_cfg["ollama"]["model"]

        def evaluate(solution, explogger=None):
            code = getattr(solution, "code", "")

            try:
                # Strip markdown code fences if the model returns them
                if "```" in code:
                    parts = code.split("```")
                    if len(parts) >= 3:
                        code = parts[1]
                        if code.startswith("python"):
                            code = code[len("python"):].lstrip()

                validate_python_code(code)

                local_ns: dict[str, Any] = {}
                exec(
                    code,
                    {
                        "np": __import__("numpy"),
                        "numpy": __import__("numpy"),
                        "__builtins__": __builtins__,
                    },
                    local_ns,
                )

                solver = local_ns.get("solve")
                if solver is None:
                    raise ValueError("Generated code does not define a top-level solve()")

                result = evaluate_solver_callable(solver, task)
                solution.add_metadata("raw_objective_mean", result["raw_objective_mean"])
                solution.add_metadata("per_problem", result["per_problem"])
                solution.set_scores(result["fitness"], f"Benchmark fitness: {result['fitness']:.6f}")

            except Exception as exc:
                print(f"[LLAMEA EVAL ERROR] task={task.name} seed={seed}: {type(exc).__name__}: {exc}")
                try:
                    print("----- GENERATED CODE START -----")
                    print(code)
                    print("----- GENERATED CODE END -----")
                except Exception:
                    pass

                solution.add_metadata("raw_objective_mean", None)
                solution.set_scores(float("-inf"), f"Execution failed: {exc}", exc)

            return solution

        llm = Ollama_LLM(model=model_name)

        progress_rows: list[dict[str, Any]] = []
        best_raw = None
        status = "success"
        notes = ""
        best_score = None
        candidates_evaluated = 0
        artifact_dir = str(run_dir)

        try:
            with pushd(run_dir), ResourceMonitor() as monitor:
                es = LLaMEA(
                    evaluate,
                    llm=llm,
                    n_parents=n_parents,
                    n_offspring=n_offspring,
                    task_prompt=llamea_task_prompt(task),
                    experiment_name=f"{task.name}_seed_{seed}",
                    budget=budget,
                    max_workers=max_workers,
                    eval_timeout=eval_timeout,
                    elitism=True,
                    log=True,
                )
                best = es.run()
                candidates_evaluated = len(es.run_history)

                running_best = float("-inf")
                for idx, sol in enumerate(es.run_history, start=1):
                    score = safe_float(getattr(sol, "fitness", None))
                    if score is None:
                        score = float("-inf")

                    running_best = max(running_best, score)
                    progress_rows.append(
                        {
                            "framework": "llamea",
                            "benchmark": task.name,
                            "seed": seed,
                            "sample_index": idx,
                            "elapsed_sec": None,
                            "candidate_score": score,
                            "best_so_far": running_best,
                        }
                    )

                best_score = safe_float(getattr(best, "fitness", None))
                best_raw = (
                    safe_float(getattr(best, "metadata", {}).get("raw_objective_mean"))
                    if hasattr(best, "metadata")
                    else None
                )
                artifact_dir = str(Path(getattr(getattr(es, "logger", None), "dirname", run_dir)).resolve())
                runtime_sec = monitor.runtime_sec
                peak_rss_mb = monitor.peak_rss_mb

        except Exception as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
            runtime_sec = 0.0
            peak_rss_mb = None
            progress_rows.append(
                {
                    "framework": "llamea",
                    "benchmark": task.name,
                    "seed": seed,
                    "sample_index": 0,
                    "elapsed_sec": None,
                    "candidate_score": None,
                    "best_so_far": None,
                }
            )
            traceback.print_exc()

        summary = RunSummary(
            framework="llamea",
            benchmark=task.name,
            seed=seed,
            status=status,
            best_search_score=best_score,
            raw_objective_mean=best_raw,
            runtime_sec=runtime_sec,
            peak_rss_mb=peak_rss_mb,
            candidates_evaluated=candidates_evaluated,
            artifact_dir=artifact_dir,
            notes=notes,
        )
        return summary, progress_rows
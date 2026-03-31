from __future__ import annotations

import json
import math
import os
import random
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..tasks import (
    BenchmarkTask,
    SkipCurrentGeneration,
    _read_skip_request_count,
    evaluate_solver_callable,
    get_safe_exec_globals,
    llamea_task_prompt,
    score_from_best_f,
    validate_python_code,
)
from ..utils import ResourceMonitor, RunSummary, import_from_repo, pushd, safe_float


PENALTY_SCORE = score_from_best_f(1e12)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _build_progress_rows(task: BenchmarkTask, seed: int, eval_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    progress_rows: list[dict[str, Any]] = []
    running_best = float("-inf")
    elapsed_so_far = 0.0
    for idx, item in enumerate(eval_records, start=1):
        score = safe_float(item.get("candidate_score"))
        if score is None:
            score = float(PENALTY_SCORE)
        running_best = max(running_best, score)
        elapsed_so_far += float(item.get("eval_time_sec") or 0.0)
        progress_rows.append(
            {
                "framework": "llamea",
                "benchmark": task.name,
                "seed": seed,
                "sample_index": idx,
                "elapsed_sec": elapsed_so_far,
                "candidate_score": score,
                "best_so_far": running_best,
                "is_valid_candidate": bool(item.get("is_valid_candidate", False)),
                "fail_reason": str(item.get("fail_reason", "") or ""),
            }
        )
    return progress_rows


def _write_live_snapshot(run_dir: Path, task: BenchmarkTask, seed: int, eval_records: list[dict[str, Any]]) -> None:
    _atomic_write_json(run_dir / "eval_records.json", eval_records)
    progress_rows = _build_progress_rows(task, seed, eval_records)
    _atomic_write_json(run_dir / "progress_rows.live.json", progress_rows)

    best_score = None
    best_raw = None
    if eval_records:
        best_item = max(
            eval_records,
            key=lambda x: safe_float(x.get("candidate_score"))
            if safe_float(x.get("candidate_score")) is not None
            else float(PENALTY_SCORE),
        )
        best_score = safe_float(best_item.get("candidate_score"))
        best_raw = safe_float(best_item.get("raw_objective_mean"))

    _atomic_write_json(
        run_dir / "summary.live.json",
        {
            "framework": "llamea",
            "benchmark": task.name,
            "seed": seed,
            "candidates_evaluated": len(eval_records),
            "best_search_score": best_score,
            "raw_objective_mean": best_raw,
            "artifact_dir": str(run_dir.resolve()),
            "updated_at": time.time(),
        },
    )


class LLAMEAAdapter(FrameworkAdapter):
    def __init__(self, framework_cfg: dict[str, Any], global_cfg: dict[str, Any], output_dir: str | Path):
        super().__init__("llamea", framework_cfg, global_cfg, output_dir)

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        text = code or ""
        if "```" not in text:
            return text
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.lstrip().startswith("python"):
                stripped = text.lstrip()
                text = stripped[len("python"):].lstrip()
        return text.strip()

    def _patch_llamea_runtime(self) -> None:
        import llamea.loggers as llamea_loggers  # type: ignore
        import llamea.llamea as llamea_core  # type: ignore
        import llamea.llm as llamea_llm  # type: ignore
        import ollama  # type: ignore

        if getattr(llamea_loggers, "_auto_algo_benchmark_patch_applied", False):
            return

        def safe_create_log_dir(logger_self, name: str = ""):
            model_name = str(name).split("/")[-1].replace(":", "_").replace("/", "_")
            today = getattr(logger_self, "working_date", "run")

            base_dir = os.path.join(str(self.output_dir), "llamea_logs")
            os.makedirs(base_dir, exist_ok=True)

            dirname = os.path.join(base_dir, f"exp-{today}-{model_name}")
            os.makedirs(dirname, exist_ok=True)
            os.makedirs(os.path.join(dirname, "configspace"), exist_ok=True)
            os.makedirs(os.path.join(dirname, "code"), exist_ok=True)
            return dirname

        def safe_pickle_archive(_llamea_self):
            return None

        def patched_ollama_init(ollama_self, model="llama3.2", base_url=None, **kwargs):
            llamea_llm.LLM.__init__(ollama_self, "", model, None, **kwargs)
            env_base = os.environ.get("OLLAMA_HOST")
            ollama_self.base_url = base_url or env_base
            ollama_self.client = ollama.Client(host=ollama_self.base_url) if ollama_self.base_url else ollama.Client()

        def patched_ollama_query(ollama_self, session_messages, max_retries: int = 5, default_delay: int = 10):
            big_message = ""
            for msg in session_messages:
                big_message += msg["content"]
            attempt = 0
            while True:
                try:
                    response = ollama_self.client.chat(
                        model=ollama_self.model,
                        messages=[{"role": "user", "content": big_message}],
                    )
                    return response["message"]["content"]
                except ollama.ResponseError as err:
                    attempt += 1
                    if attempt > max_retries or getattr(err, "status_code", None) not in (429, 500, 503):
                        raise
                    time.sleep(default_delay * attempt)
                except Exception:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    time.sleep(default_delay * attempt)

        llamea_loggers.ExperimentLogger.create_log_dir = safe_create_log_dir
        llamea_core.LLaMEA.pickle_archive = safe_pickle_archive
        llamea_llm.Ollama_LLM.__init__ = patched_ollama_init
        llamea_llm.Ollama_LLM.query = patched_ollama_query
        llamea_loggers._auto_algo_benchmark_patch_applied = True

    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict[str, Any]]]:
        run_dir = Path(self.output_dir) / "artifacts" / "llamea" / task.name / f"seed_{seed}"
        _write_live_snapshot(run_dir, task, seed, [])

        import_from_repo(self.framework_cfg["repo_path"], "llamea")
        self._patch_llamea_runtime()

        from llamea import LLaMEA, Ollama_LLM  # type: ignore

        run_dir.mkdir(parents=True, exist_ok=True)
        failure_log_path = run_dir / "failure_examples.txt"

        os.environ["OLLAMA_HOST"] = self.global_cfg["ollama"]["base_url"]

        budget = int(self.framework_cfg.get("search_budget", 12))
        n_parents = int(self.framework_cfg.get("n_parents", 1))
        n_offspring = int(self.framework_cfg.get("n_offspring", 1))
        max_workers = int(self.framework_cfg.get("max_workers", 1))
        eval_timeout = int(self.framework_cfg.get("eval_timeout", 1200))
        model_name = self.global_cfg["ollama"]["model"]
        base_url = self.global_cfg["ollama"]["base_url"]

        failure_count = 0
        eval_records: list[dict[str, Any]] = []
        skip_state = {"consumed": 0}

        def evaluate(solution, explogger=None):
            nonlocal failure_count
            start_time = time.time()
            code = self._strip_code_fences(getattr(solution, "code", ""))
            fail_reason = ""
            current_skip = _read_skip_request_count()

            if current_skip > skip_state["consumed"]:
                skip_state["consumed"] = current_skip
                fail_reason = "Skipped current generation/candidate from terminal before evaluation started"
                solution.add_metadata("raw_objective_mean", 1e12)
                solution.add_metadata("per_problem", [])
                solution.add_metadata("fail_reason", fail_reason)
                solution.set_scores(float(PENALTY_SCORE), fail_reason)
            else:
                try:
                    validate_python_code(code)
                    exec_globals = get_safe_exec_globals()
                    local_ns: dict[str, Any] = {}
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        exec(code, exec_globals, local_ns)

                    solver = local_ns.get("solve")
                    if solver is None:
                        solver = exec_globals.get("solve")
                    if solver is None or not callable(solver):
                        raise ValueError("Generated code does not define a top-level solve()")

                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        result = evaluate_solver_callable(solver, task, skip_baseline=current_skip)
                    solution.add_metadata("raw_objective_mean", result["raw_objective_mean"])
                    solution.add_metadata("per_problem", result["per_problem"])
                    solution.add_metadata("fail_reason", "")
                    solution.set_scores(result["fitness"], f"Benchmark fitness: {result['fitness']:.6f}")
                except SkipCurrentGeneration as exc:
                    skip_state["consumed"] = max(skip_state["consumed"], _read_skip_request_count())
                    fail_reason = str(exc) or "Skipped current generation/candidate from terminal"
                    solution.add_metadata("raw_objective_mean", 1e12)
                    solution.add_metadata("per_problem", [])
                    solution.add_metadata("fail_reason", fail_reason)
                    solution.set_scores(float(PENALTY_SCORE), fail_reason)
                except (RuntimeWarning, FloatingPointError, OverflowError) as exc:
                    fail_reason = f"Numerical instability: {type(exc).__name__}: {exc}"
                    solution.add_metadata("raw_objective_mean", 1e12)
                    solution.add_metadata("per_problem", [])
                    solution.add_metadata("fail_reason", fail_reason)
                    solution.set_scores(float(PENALTY_SCORE), f"Execution failed: {exc}", exc)

                    if failure_count < 50:
                        try:
                            with failure_log_path.open("a", encoding="utf-8") as f:
                                f.write(f"task={task.name} seed={seed} failure={fail_reason}\n")
                                f.write("----- GENERATED CODE START -----\n")
                                f.write(code)
                                f.write("\n----- GENERATED CODE END -----\n\n")
                            failure_count += 1
                        except Exception:
                            pass
                except Exception as exc:
                    fail_reason = f"{type(exc).__name__}: {exc}"
                    solution.add_metadata("raw_objective_mean", 1e12)
                    solution.add_metadata("per_problem", [])
                    solution.add_metadata("fail_reason", fail_reason)
                    solution.set_scores(float(PENALTY_SCORE), f"Execution failed: {exc}", exc)

                    if failure_count < 50:
                        try:
                            with failure_log_path.open("a", encoding="utf-8") as f:
                                f.write(f"task={task.name} seed={seed} failure={fail_reason}\n")
                                f.write("----- GENERATED CODE START -----\n")
                                f.write(code)
                                f.write("\n----- GENERATED CODE END -----\n\n")
                            failure_count += 1
                        except Exception:
                            pass

            record = {
                "candidate_score": safe_float(getattr(solution, "fitness", None)) or float(PENALTY_SCORE),
                "raw_objective_mean": safe_float(getattr(solution, "metadata", {}).get("raw_objective_mean")) if hasattr(solution, "metadata") else 1e12,
                "is_valid_candidate": (safe_float(getattr(solution, "fitness", None)) or float(PENALTY_SCORE)) > PENALTY_SCORE + 1e-9,
                "fail_reason": fail_reason,
                "eval_time_sec": time.time() - start_time,
            }
            eval_records.append(record)
            try:
                _write_live_snapshot(run_dir, task, seed, list(eval_records))
            except Exception:
                pass
            return solution

        llm = Ollama_LLM(model=model_name, base_url=base_url)

        progress_rows: list[dict[str, Any]] = []
        best_raw = None
        status = "success"
        notes = ""
        best_score = None
        candidates_evaluated = 0
        artifact_dir = str(run_dir)
        peak_rss_mb = None
        runtime_sec = 0.0

        try:
            effective_eval_timeout = eval_timeout if max_workers > 1 else None
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
                    eval_timeout=effective_eval_timeout,
                    elitism=True,
                    log=True,
                )

                best = es.run()
                candidates_evaluated = len(es.run_history)
                progress_rows = _build_progress_rows(task, seed, eval_records)
                best_score = safe_float(getattr(best, "fitness", None))
                best_raw = safe_float(getattr(best, "metadata", {}).get("raw_objective_mean")) if hasattr(best, "metadata") else None
                artifact_dir = str(Path(getattr(getattr(es, "logger", None), "dirname", run_dir)).resolve())
            runtime_sec = monitor.runtime_sec
            peak_rss_mb = monitor.peak_rss_mb
            _write_live_snapshot(run_dir, task, seed, eval_records)

            if best_score is None or best_score <= PENALTY_SCORE + 1e-9:
                status = "no_valid_candidate"
                notes = "All generated candidates fell back to the penalty baseline. See failure_examples.txt for examples."
            elif failure_log_path.exists():
                notes = "Some candidates failed validation or execution. See failure_examples.txt for examples."
        except Exception as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
            runtime_sec = 0.0
            peak_rss_mb = None
            progress_rows = _build_progress_rows(task, seed, eval_records)
            if not progress_rows:
                progress_rows.append(
                    {
                        "framework": "llamea",
                        "benchmark": task.name,
                        "seed": seed,
                        "sample_index": 0,
                        "elapsed_sec": None,
                        "candidate_score": None,
                        "best_so_far": None,
                        "is_valid_candidate": False,
                        "fail_reason": notes,
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
        try:
            _atomic_write_json(Path(run_dir) / "summary.live.json", dict(summary.__dict__))
            if progress_rows:
                _atomic_write_json(Path(run_dir) / "progress_rows.live.json", progress_rows)
        except Exception:
            pass
        return summary, progress_rows

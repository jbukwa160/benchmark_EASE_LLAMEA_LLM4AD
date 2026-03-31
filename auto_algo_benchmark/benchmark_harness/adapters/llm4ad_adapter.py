from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..tasks import (
    BenchmarkTask,
    SkipCurrentGeneration,
    evaluate_solver_callable,
    get_safe_exec_globals,
    llm4ad_task_description,
    llm4ad_template_program,
    score_from_best_f,
    validate_python_code,
)
from ..utils import ResourceMonitor, RunSummary, ensure_dir, import_from_repo, safe_float


PENALTY_SCORE = score_from_best_f(1e12)


def _choose_mp_context(fork_proc: Any) -> mp.context.BaseContext:
    if fork_proc in (True, "fork"):
        return mp.get_context("fork")
    if fork_proc in (False, "spawn"):
        return mp.get_context("spawn")
    if sys.platform.startswith(("linux", "darwin")):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _read_skip_request_count() -> int:
    path_str = os.environ.get("BENCHMARK_SKIP_SIGNAL_FILE", "").strip()
    if not path_str:
        return 0
    path = Path(path_str)
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    for key in ("skip_count", "generation"):
        try:
            return int(payload.get(key, 0) or 0)
        except Exception:
            continue
    return 0


def _skip_requested_since(start_count: int) -> bool:
    return _read_skip_request_count() > int(start_count)


def _terminate_process(proc: mp.Process | None, join_timeout: float = 5.0) -> None:
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=join_timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=join_timeout)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_llm4ad_live_snapshot(run_dir: Path, task: BenchmarkTask, seed: int, eval_records: list[dict[str, Any]]) -> None:
    _atomic_write_json(run_dir / "eval_records.live.json", eval_records)

    running_best = float("-inf")
    elapsed_so_far = 0.0
    best_raw = None
    progress_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(eval_records, start=1):
        score = safe_float(item.get("candidate_score"))
        if score is None:
            score = float(PENALTY_SCORE)
        elapsed_so_far += float(item.get("eval_time_sec") or 0.0)
        if score > running_best:
            running_best = score
            best_raw = safe_float(item.get("raw_objective_mean"))
        progress_rows.append(
            {
                "framework": "llm4ad",
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

    _atomic_write_json(run_dir / "progress_rows.live.json", progress_rows)
    _atomic_write_json(
        run_dir / "summary.live.json",
        {
            "framework": "llm4ad",
            "benchmark": task.name,
            "seed": seed,
            "candidates_evaluated": len(eval_records),
            "best_search_score": None if not eval_records else running_best,
            "raw_objective_mean": best_raw,
            "artifact_dir": str(run_dir.resolve()),
            "updated_at": time.time(),
        },
    )


def _evaluate_program_worker(program_str: str, task: BenchmarkTask, skip_baseline: int, result_queue) -> None:
    try:
        validate_python_code(program_str)
        namespace = get_safe_exec_globals()
        exec(program_str, namespace)
        solver = namespace.get("solve")
        if solver is None or not callable(solver):
            raise ValueError("Generated program does not define a callable solve()")
        result = evaluate_solver_callable(solver, task, skip_baseline=skip_baseline)
        result_queue.put(
            {
                "ok": True,
                "fitness": float(result["fitness"]),
                "raw_objective_mean": float(result["raw_objective_mean"]),
                "objective_calls_total": int(result.get("objective_calls_total", 0)),
                "fail_reason": "",
            }
        )
    except SkipCurrentGeneration as exc:
        result_queue.put(
            {
                "ok": False,
                "fitness": float(PENALTY_SCORE),
                "raw_objective_mean": 1e12,
                "objective_calls_total": 0,
                "fail_reason": str(exc) or "Skipped current generation from terminal",
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "fitness": float(PENALTY_SCORE),
                "raw_objective_mean": 1e12,
                "objective_calls_total": 0,
                "fail_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_llm4ad_task_worker(
    framework_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    output_dir: str,
    task: BenchmarkTask,
    seed: int,
    result_queue,
) -> None:
    try:
        import_from_repo(framework_cfg["repo_path"], "llm4ad")

        from llm4ad.base import Evaluation  # type: ignore
        from llm4ad.method.eoh import EoH, EoHProfiler  # type: ignore
        from llm4ad.tools.llm.local_ollama import LocalOllamaLLM  # type: ignore

        run_dir = ensure_dir(Path(output_dir) / "artifacts" / "llm4ad" / task.name / f"seed_{seed}")

        timeout_seconds = int(framework_cfg.get("timeout_seconds", 1200))
        max_sample_nums = int(framework_cfg.get("max_sample_nums", 12))
        max_generations = int(framework_cfg.get("max_generations", 6))
        pop_size = int(framework_cfg.get("pop_size", 4))
        selection_num = int(framework_cfg.get("selection_num", 2))
        num_samplers = int(framework_cfg.get("num_samplers", 1))
        num_evaluators = int(framework_cfg.get("num_evaluators", 1))
        model_name = global_cfg["ollama"]["model"]
        base_url = global_cfg["ollama"]["base_url"]
        os.environ["OLLAMA_HOST"] = base_url
        daemon_eval_process = bool(framework_cfg.get("daemon_eval_process", False))
        fork_proc = framework_cfg.get("fork_proc", "auto")

        lock = threading.Lock()
        eval_records: list[dict[str, Any]] = []

        class ContinuousOptimizationEvaluation(Evaluation):
            def __init__(self):
                super().__init__(
                    template_program=llm4ad_template_program(),
                    task_description=llm4ad_task_description(task),
                    timeout_seconds=None,
                    random_seed=None,
                    exec_code=False,
                    safe_evaluate=False,
                )

            def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
                skip_baseline = _read_skip_request_count()
                started = time.perf_counter()
                payload: dict[str, Any]
                ctx = _choose_mp_context(fork_proc)
                queue = ctx.Queue()
                proc = ctx.Process(
                    target=_evaluate_program_worker,
                    args=(program_str, task, skip_baseline, queue),
                    daemon=daemon_eval_process,
                )
                proc.start()
                try:
                    while True:
                        if _skip_requested_since(skip_baseline):
                            payload = {
                                "ok": False,
                                "fitness": float(PENALTY_SCORE),
                                "raw_objective_mean": 1e12,
                                "objective_calls_total": 0,
                                "fail_reason": "Skipped current generation from terminal",
                            }
                            break
                        try:
                            payload = queue.get_nowait()
                            break
                        except Exception:
                            pass
                        if not proc.is_alive():
                            try:
                                payload = queue.get_nowait()
                            except Exception:
                                payload = {
                                    "ok": False,
                                    "fitness": float(PENALTY_SCORE),
                                    "raw_objective_mean": 1e12,
                                    "objective_calls_total": 0,
                                    "fail_reason": f"RuntimeError: candidate evaluation process exited with code {proc.exitcode}",
                                }
                            break
                        if time.perf_counter() - started >= timeout_seconds:
                            payload = {
                                "ok": False,
                                "fitness": float(PENALTY_SCORE),
                                "raw_objective_mean": 1e12,
                                "objective_calls_total": 0,
                                "fail_reason": f"TimeoutError: candidate evaluation exceeded {timeout_seconds} seconds",
                            }
                            break
                        time.sleep(0.1)
                finally:
                    _terminate_process(proc)
                    queue.close()
                    queue.join_thread()

                fitness = safe_float(payload.get("fitness"))
                if fitness is None or not math.isfinite(fitness):
                    fitness = float(PENALTY_SCORE)
                raw_objective = safe_float(payload.get("raw_objective_mean"))
                if raw_objective is None or not math.isfinite(raw_objective):
                    raw_objective = 1e12
                fail_reason = str(payload.get("fail_reason", "") or "")
                eval_time = time.perf_counter() - started

                with lock:
                    eval_records.append(
                        {
                            "candidate_score": fitness,
                            "raw_objective_mean": raw_objective,
                            "objective_calls_total": int(payload.get("objective_calls_total", 0) or 0),
                            "is_valid_candidate": bool(payload.get("ok", False)) and fitness > PENALTY_SCORE + 1e-9,
                            "fail_reason": fail_reason,
                            "eval_time_sec": eval_time,
                        }
                    )
                    _write_llm4ad_live_snapshot(run_dir, task, seed, eval_records)

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

            _atomic_write_json(Path(run_dir) / "eval_records.json", eval_records)

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
                        "framework": "llm4ad",
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
            candidates_evaluated = len(eval_records)
            if eval_records:
                best_item = max(
                    eval_records,
                    key=lambda x: safe_float(x.get("candidate_score"))
                    if safe_float(x.get("candidate_score")) is not None
                    else float(PENALTY_SCORE),
                )
                best_score = safe_float(best_item.get("candidate_score"))
                best_raw = safe_float(best_item.get("raw_objective_mean"))
            if best_score is None or best_score <= PENALTY_SCORE + 1e-9:
                status = "no_valid_candidate"
                notes = "All generated candidates fell back to the penalty baseline. Check eval_records.json for failure reasons."
        except Exception as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        summary = {
            "framework": "llm4ad",
            "benchmark": task.name,
            "seed": seed,
            "status": status,
            "best_search_score": best_score,
            "raw_objective_mean": best_raw,
            "runtime_sec": runtime_sec,
            "peak_rss_mb": peak_rss_mb,
            "candidates_evaluated": candidates_evaluated,
            "artifact_dir": str(Path(run_dir).resolve()),
            "notes": notes,
        }
        result_queue.put({"summary": summary, "progress_rows": progress_rows})
    except Exception as exc:
        result_queue.put(
            {
                "summary": {
                    "framework": "llm4ad",
                    "benchmark": task.name,
                    "seed": seed,
                    "status": "failed",
                    "best_search_score": None,
                    "raw_objective_mean": None,
                    "runtime_sec": 0.0,
                    "peak_rss_mb": None,
                    "candidates_evaluated": 0,
                    "artifact_dir": str((Path(output_dir) / "artifacts" / "llm4ad" / task.name / f"seed_{seed}").resolve()),
                    "notes": f"{type(exc).__name__}: {exc}",
                },
                "progress_rows": [],
            }
        )


class LLM4ADAdapter(FrameworkAdapter):
    def __init__(self, framework_cfg: dict[str, Any], global_cfg: dict[str, Any], output_dir: str | Path):
        super().__init__("llm4ad", framework_cfg, global_cfg, output_dir)

    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict[str, Any]]]:
        run_dir = ensure_dir(self.output_dir / "artifacts" / "llm4ad" / task.name / f"seed_{seed}")
        fork_proc = self.framework_cfg.get("fork_proc", "auto")
        ctx = _choose_mp_context(fork_proc)
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_run_llm4ad_task_worker,
            args=(self.framework_cfg, self.global_cfg, str(self.output_dir), task, seed, result_queue),
        )
        proc.start()
        proc.join()

        payload = None
        try:
            payload = result_queue.get(timeout=1.0)
        except Exception:
            payload = None
        finally:
            result_queue.close()
            result_queue.join_thread()

        if payload is None:
            summary = RunSummary(
                framework="llm4ad",
                benchmark=task.name,
                seed=seed,
                status="failed",
                best_search_score=None,
                raw_objective_mean=None,
                runtime_sec=0.0,
                peak_rss_mb=None,
                candidates_evaluated=0,
                artifact_dir=str(Path(run_dir).resolve()),
                notes=f"Worker process exited without returning results (exit code: {proc.exitcode}).",
            )
            return summary, []

        summary = RunSummary(**payload["summary"])
        progress_rows = list(payload.get("progress_rows", []))
        return summary, progress_rows
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Any

from .base import FrameworkAdapter
from ..tasks import (
    BenchmarkTask,
    evaluate_solver_callable,
    get_safe_exec_globals,
    llamea_task_prompt,
    score_from_best_f,
    validate_python_code,
)
from ..utils import ResourceMonitor, RunSummary, import_from_repo, pushd, safe_float


PENALTY_SCORE = score_from_best_f(1e12)


def _choose_mp_context() -> mp.context.BaseContext:
    if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _read_skip_generation() -> int:
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
    try:
        return int(payload.get("generation", 0) or 0)
    except Exception:
        return 0


def _skip_requested_since(start_generation: int) -> bool:
    return _read_skip_generation() > int(start_generation)


def _terminate_process(proc: mp.Process | None, join_timeout: float = 5.0) -> None:
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=join_timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=join_timeout)


class SkipRequestedError(RuntimeError):
    pass


def _run_llamea_task_worker(
    framework_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    output_dir: str,
    task: BenchmarkTask,
    seed: int,
    result_queue,
) -> None:
    try:
        import_from_repo(framework_cfg["repo_path"], "llamea")
        _patch_llamea_runtime(output_dir)

        from llamea import LLaMEA, Ollama_LLM  # type: ignore

        run_dir = Path(output_dir) / "artifacts" / "llamea" / task.name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        failure_log_path = run_dir / "failure_examples.txt"

        os.environ["OLLAMA_HOST"] = global_cfg["ollama"]["base_url"]

        budget = int(framework_cfg.get("search_budget", 12))
        n_parents = int(framework_cfg.get("n_parents", 1))
        n_offspring = int(framework_cfg.get("n_offspring", 1))
        max_workers = int(framework_cfg.get("max_workers", 1))
        eval_timeout = int(framework_cfg.get("eval_timeout", 1200))
        model_name = global_cfg["ollama"]["model"]
        base_url = global_cfg["ollama"]["base_url"]
        skip_generation = _read_skip_generation()
        os.environ["BENCHMARK_TASK_SKIP_BASELINE"] = str(skip_generation)

        failure_count = 0
        failure_lock = threading.Lock()

        def evaluate(solution, explogger=None):
            nonlocal failure_count
            if _skip_requested_since(skip_generation):
                raise SkipRequestedError("Skipped by user from terminal")

            code = _strip_code_fences(getattr(solution, "code", ""))

            try:
                validate_python_code(code)
                namespace = get_safe_exec_globals()
                namespace["random"] = random
                exec(code, namespace)

                solver = namespace.get("solve")
                if solver is None or not callable(solver):
                    raise ValueError("Generated code does not define a callable solve()")

                result = evaluate_solver_callable(solver, task)
                solution.add_metadata("raw_objective_mean", result["raw_objective_mean"])
                solution.add_metadata("per_problem", result["per_problem"])
                solution.add_metadata("objective_calls_total", result.get("objective_calls_total", 0))
                solution.add_metadata("fail_reason", "")
                solution.set_scores(result["fitness"], f"Benchmark fitness: {result['fitness']:.6f}")

            except SkipRequestedError:
                raise
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                solution.add_metadata("raw_objective_mean", 1e12)
                solution.add_metadata("per_problem", [])
                solution.add_metadata("objective_calls_total", 0)
                solution.add_metadata("fail_reason", failure_reason)
                solution.set_scores(float(PENALTY_SCORE), f"Execution failed: {exc}", exc)

                with failure_lock:
                    should_log = failure_count < 50
                    if should_log:
                        failure_count += 1
                if should_log:
                    try:
                        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with failure_log_path.open("a", encoding="utf-8") as f:
                            f.write(f"task={task.name} seed={seed} failure={failure_reason}\n")
                            f.write("----- GENERATED CODE START -----\n")
                            f.write(code)
                            f.write("\n----- GENERATED CODE END -----\n\n")
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
                        score = float(PENALTY_SCORE)

                    running_best = max(running_best, score)
                    fail_reason = ""
                    metadata = getattr(sol, "metadata", {}) or {}
                    if isinstance(metadata, dict):
                        fail_reason = str(metadata.get("fail_reason", "") or "")

                    progress_rows.append(
                        {
                            "framework": "llamea",
                            "benchmark": task.name,
                            "seed": seed,
                            "sample_index": idx,
                            "elapsed_sec": None,
                            "candidate_score": score,
                            "best_so_far": running_best,
                            "is_valid_candidate": score > PENALTY_SCORE + 1e-9,
                            "fail_reason": fail_reason,
                        }
                    )

                best_score = safe_float(getattr(best, "fitness", None))
                metadata = getattr(best, "metadata", {}) if best is not None else {}
                if isinstance(metadata, dict):
                    best_raw = safe_float(metadata.get("raw_objective_mean"))
                artifact_dir = str(Path(getattr(getattr(es, "logger", None), "dirname", run_dir)).resolve())

            runtime_sec = monitor.runtime_sec
            peak_rss_mb = monitor.peak_rss_mb

            if best_score is None or best_score <= PENALTY_SCORE + 1e-9:
                status = "no_valid_candidate"
                notes = "All generated candidates fell back to the penalty baseline. See failure_examples.txt for examples."
            elif failure_log_path.exists():
                notes = "Some candidates failed validation or execution. See failure_examples.txt for examples."

        except SkipRequestedError:
            status = "skipped"
            notes = "Skipped by user from terminal with the S key."
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
                    "is_valid_candidate": False,
                    "fail_reason": notes,
                }
            )
            traceback.print_exc()

        summary = {
            "framework": "llamea",
            "benchmark": task.name,
            "seed": seed,
            "status": status,
            "best_search_score": best_score,
            "raw_objective_mean": best_raw,
            "runtime_sec": runtime_sec,
            "peak_rss_mb": peak_rss_mb,
            "candidates_evaluated": candidates_evaluated,
            "artifact_dir": artifact_dir,
            "notes": notes,
        }
        result_queue.put({"summary": summary, "progress_rows": progress_rows})
    except Exception as exc:
        result_queue.put(
            {
                "summary": {
                    "framework": "llamea",
                    "benchmark": task.name,
                    "seed": seed,
                    "status": "failed",
                    "best_search_score": None,
                    "raw_objective_mean": None,
                    "runtime_sec": 0.0,
                    "peak_rss_mb": None,
                    "candidates_evaluated": 0,
                    "artifact_dir": str((Path(output_dir) / "artifacts" / "llamea" / task.name / f"seed_{seed}").resolve()),
                    "notes": f"{type(exc).__name__}: {exc}",
                },
                "progress_rows": [],
            }
        )


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


def _patch_llamea_runtime(output_dir: str) -> None:
    import llamea.loggers as llamea_loggers  # type: ignore
    import llamea.llamea as llamea_core  # type: ignore
    import llamea.llm as llamea_llm  # type: ignore
    import ollama  # type: ignore

    if getattr(llamea_loggers, "_auto_algo_benchmark_patch_applied", False):
        return

    def safe_create_log_dir(logger_self, name: str = ""):
        model_name = str(name).split("/")[-1].replace(":", "_").replace("/", "_")
        today = getattr(logger_self, "working_date", "run")

        base_dir = os.path.join(str(output_dir), "llamea_logs")
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
        try:
            baseline = int(os.environ.get("BENCHMARK_TASK_SKIP_BASELINE", "0") or 0)
        except Exception:
            baseline = 0
        if _skip_requested_since(baseline):
            raise SkipRequestedError("Skipped by user from terminal")

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
                if _skip_requested_since(baseline):
                    raise SkipRequestedError("Skipped by user from terminal")
                if attempt > max_retries or getattr(err, "status_code", None) not in (429, 500, 503):
                    raise
                time.sleep(default_delay * attempt)
            except SkipRequestedError:
                raise
            except Exception:
                attempt += 1
                if _skip_requested_since(baseline):
                    raise SkipRequestedError("Skipped by user from terminal")
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)

    llamea_loggers.ExperimentLogger.create_log_dir = safe_create_log_dir
    llamea_core.LLaMEA.pickle_archive = safe_pickle_archive
    llamea_llm.Ollama_LLM.__init__ = patched_ollama_init
    llamea_llm.Ollama_LLM.query = patched_ollama_query
    llamea_loggers._auto_algo_benchmark_patch_applied = True


class LLAMEAAdapter(FrameworkAdapter):
    def __init__(self, framework_cfg: dict[str, Any], global_cfg: dict[str, Any], output_dir: str | Path):
        super().__init__("llamea", framework_cfg, global_cfg, output_dir)

    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict[str, Any]]]:
        run_dir = Path(self.output_dir) / "artifacts" / "llamea" / task.name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        skip_generation = _read_skip_generation()
        ctx = _choose_mp_context()
        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_run_llamea_task_worker,
            args=(self.framework_cfg, self.global_cfg, str(self.output_dir), task, seed, result_queue),
        )
        proc.start()

        skipped = False
        try:
            while proc.is_alive():
                proc.join(timeout=0.25)
                if _skip_requested_since(skip_generation):
                    skipped = True
                    _terminate_process(proc)
                    break
        finally:
            if proc.is_alive():
                _terminate_process(proc)

        if skipped:
            summary = RunSummary(
                framework="llamea",
                benchmark=task.name,
                seed=seed,
                status="skipped",
                best_search_score=None,
                raw_objective_mean=None,
                runtime_sec=0.0,
                peak_rss_mb=None,
                candidates_evaluated=0,
                artifact_dir=str(run_dir.resolve()),
                notes="Skipped by user from terminal with the S key.",
            )
            return summary, []

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
                framework="llamea",
                benchmark=task.name,
                seed=seed,
                status="failed",
                best_search_score=None,
                raw_objective_mean=None,
                runtime_sec=0.0,
                peak_rss_mb=None,
                candidates_evaluated=0,
                artifact_dir=str(run_dir.resolve()),
                notes=f"Worker process exited without returning results (exit code: {proc.exitcode}).",
            )
            return summary, []

        summary = RunSummary(**payload["summary"])
        progress_rows = list(payload.get("progress_rows", []))
        return summary, progress_rows

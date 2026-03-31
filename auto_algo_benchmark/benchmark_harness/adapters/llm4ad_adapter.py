from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import queue
import random
import threading
import traceback
import urllib.request
import warnings
from pathlib import Path
from types import MethodType
from typing import Any

from .base import FrameworkAdapter
from ..tasks import BenchmarkTask, ManualSkipRequested, evaluate_solver_callable, llamea_task_prompt, score_from_best_f, validate_python_code
from ..utils import ResourceMonitor, RunSummary, import_from_repo, pushd, safe_float


PENALTY_SCORE = score_from_best_f(1e12)
SKIP_ENV_VAR = "AUTO_BENCHMARK_SKIP_FLAG"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _skip_flag_path() -> Path | None:
    raw = os.environ.get(SKIP_ENV_VAR)
    return Path(raw) if raw else None


def _consume_skip_request() -> bool:
    path = _skip_flag_path()
    if path is None:
        return False
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        if path.exists():
            return True
    return False


def _ollama_chat_worker(result_queue: mp.Queue, base_url: str, model_name: str, big_message: str) -> None:
    try:
        url = base_url.rstrip("/") + "/api/chat"
        payload = json.dumps(
            {
                "model": model_name,
                "messages": [{"role": "user", "content": big_message}],
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result_queue.put({"ok": True, "response": str(data.get("message", {}).get("content", ""))})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _evaluate_solution_worker(result_queue: mp.Queue, code: str, task: BenchmarkTask) -> None:
    try:
        import math
        import random
        import warnings

        import numpy as np

        local_ns: dict[str, Any] = {}
        exec_globals = {
            "np": np,
            "numpy": np,
            "math": math,
            "random": random,
            "__builtins__": __builtins__,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            with np.errstate(all="raise"):
                exec(code, exec_globals, local_ns)
                solver = local_ns.get("solve") or exec_globals.get("solve")
                if solver is None:
                    raise ValueError("Generated code does not define a top-level solve()")
                result = evaluate_solver_callable(solver, task)
        result_queue.put({"ok": True, "result": result})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


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

        llamea_loggers.ExperimentLogger.create_log_dir = safe_create_log_dir
        llamea_core.LLaMEA.pickle_archive = safe_pickle_archive
        llamea_loggers._auto_algo_benchmark_patch_applied = True

    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict[str, Any]]]:
        import_from_repo(self.framework_cfg["repo_path"], "llamea")
        self._patch_llamea_runtime()

        from joblib import Parallel, delayed  # type: ignore
        from llamea import LLaMEA, Ollama_LLM  # type: ignore
        import numpy as np

        run_dir = Path(self.output_dir) / "artifacts" / "llamea" / task.name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        failure_log_path = run_dir / "failure_examples.txt"
        progress_live_path = run_dir / "progress_rows.live.json"
        summary_live_path = run_dir / "summary.live.json"
        eval_live_path = run_dir / "eval_records.json"

        os.environ["OLLAMA_HOST"] = self.global_cfg["ollama"]["base_url"]

        budget = int(self.framework_cfg.get("search_budget", 12))
        n_parents = int(self.framework_cfg.get("n_parents", 1))
        n_offspring = int(self.framework_cfg.get("n_offspring", 1))
        max_workers = int(self.framework_cfg.get("max_workers", 1))
        eval_timeout = int(self.framework_cfg.get("eval_timeout", 1200))
        model_name = self.global_cfg["ollama"]["model"]
        base_url = self.global_cfg["ollama"]["base_url"]

        failure_count = 0
        progress_rows: list[dict[str, Any]] = []
        eval_records: list[dict[str, Any]] = []
        best_raw = None
        status = "success"
        notes = ""
        best_score = None
        candidates_evaluated = 0
        artifact_dir = str(run_dir)
        peak_rss_mb = None
        runtime_sec = 0.0
        lock = threading.Lock()

        def flush_live(es: Any | None = None) -> None:
            nonlocal best_raw, best_score, candidates_evaluated
            with lock:
                _write_json(progress_live_path, progress_rows)
                _write_json(eval_live_path, eval_records)
                if progress_rows:
                    best_score = max((safe_float(x.get("best_so_far")) for x in progress_rows if safe_float(x.get("best_so_far")) is not None), default=best_score)
                best_raw = min((safe_float(x.get("raw_objective_mean")) for x in eval_records if safe_float(x.get("raw_objective_mean")) is not None), default=best_raw)
                candidates_evaluated = len(progress_rows)
                _write_json(summary_live_path, {
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
                    "generation": getattr(es, "generation", None),
                })

        def evaluate(solution, explogger=None):
            nonlocal failure_count
            code = self._strip_code_fences(getattr(solution, "code", ""))

            try:
                validate_python_code(code)
                result_queue: mp.Queue = mp.Queue()
                process = mp.Process(target=_evaluate_solution_worker, args=(result_queue, code, task))
                process.start()
                payload: dict[str, Any] | None = None
                skipped = False
                while True:
                    if _consume_skip_request():
                        skipped = True
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=5)
                            if process.is_alive():
                                process.kill()
                                process.join()
                        payload = {"ok": False, "error": "ManualSkipRequested"}
                        break
                    if not process.is_alive():
                        try:
                            payload = result_queue.get_nowait()
                        except queue.Empty:
                            payload = {"ok": False, "error": "Worker exited without result"}
                        break
                    try:
                        payload = result_queue.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue

                if payload and payload.get("ok"):
                    result = payload["result"]
                    solution.add_metadata("raw_objective_mean", result["raw_objective_mean"])
                    solution.add_metadata("per_problem", result["per_problem"])
                    solution.add_metadata("fail_reason", "")
                    solution.set_scores(result["fitness"], f"Benchmark fitness: {result['fitness']:.6f}")
                else:
                    failure_reason = str((payload or {}).get("error", "Unknown evaluation failure"))
                    solution.add_metadata("raw_objective_mean", 1e12)
                    solution.add_metadata("per_problem", [])
                    solution.add_metadata("fail_reason", failure_reason)
                    solution.set_scores(float(PENALTY_SCORE), f"Execution failed: {failure_reason}", RuntimeError(failure_reason))
                    if not skipped and failure_count < 50:
                        try:
                            with failure_log_path.open("a", encoding="utf-8") as f:
                                f.write(f"task={task.name} seed={seed} failure={failure_reason}\n")
                                f.write("----- GENERATED CODE START -----\n")
                                f.write(code)
                                f.write("\n----- GENERATED CODE END -----\n\n")
                            failure_count += 1
                        except Exception:
                            pass
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                solution.add_metadata("raw_objective_mean", 1e12)
                solution.add_metadata("per_problem", [])
                solution.add_metadata("fail_reason", failure_reason)
                solution.set_scores(float(PENALTY_SCORE), f"Execution failed: {failure_reason}", exc)

            with lock:
                eval_records.append(
                    {
                        "candidate_score": safe_float(getattr(solution, "fitness", None)) if safe_float(getattr(solution, "fitness", None)) is not None else float(PENALTY_SCORE),
                        "raw_objective_mean": safe_float(getattr(solution, "metadata", {}).get("raw_objective_mean")) if hasattr(solution, "metadata") else 1e12,
                        "is_valid_candidate": (safe_float(getattr(solution, "fitness", None)) or float(PENALTY_SCORE)) > PENALTY_SCORE + 1e-9,
                        "fail_reason": str(getattr(solution, "metadata", {}).get("fail_reason", "")) if hasattr(solution, "metadata") else "",
                    }
                )
            flush_live(None)
            return solution

        llm = Ollama_LLM(model=model_name)

        def patched_query(_llm_self, session_messages, max_retries: int = 1, default_delay: int = 0):
            del max_retries, default_delay
            big_message = ""
            for msg in session_messages:
                big_message += msg["content"] + "\n"
            result_queue: mp.Queue = mp.Queue()
            process = mp.Process(target=_ollama_chat_worker, args=(result_queue, base_url, model_name, big_message))
            process.start()
            payload: dict[str, Any] | None = None
            while True:
                if _consume_skip_request():
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join()
                    raise ManualSkipRequested("Manual skip requested")
                if not process.is_alive():
                    try:
                        payload = result_queue.get_nowait()
                    except queue.Empty:
                        payload = {"ok": False, "error": "LLM worker exited without result"}
                    break
                try:
                    payload = result_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
            if not payload or not payload.get("ok"):
                raise RuntimeError(str((payload or {}).get("error", "LLM query failed")))
            response = str(payload.get("response", "") or "")
            if not response.strip():
                raise RuntimeError("Empty LLM response")
            return response

        llm.query = MethodType(patched_query, llm)

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

                def checkpoint_new_rows():
                    with lock:
                        start = len(progress_rows)
                        new_items = es.run_history[start:]
                        running_best = progress_rows[-1]["best_so_far"] if progress_rows else float("-inf")
                        for item in new_items:
                            score = safe_float(getattr(item, "fitness", None))
                            if score is None:
                                score = float(PENALTY_SCORE)
                            running_best = max(running_best, score)
                            metadata = getattr(item, "metadata", {}) or {}
                            progress_rows.append(
                                {
                                    "framework": "llamea",
                                    "benchmark": task.name,
                                    "seed": seed,
                                    "generation": getattr(item, "generation", None),
                                    "sample_index": len(progress_rows) + 1,
                                    "elapsed_sec": None,
                                    "candidate_score": score,
                                    "best_so_far": running_best,
                                    "is_valid_candidate": score > PENALTY_SCORE + 1e-9,
                                    "fail_reason": str(metadata.get("fail_reason", "") or ""),
                                }
                            )
                    flush_live(es)

                def patched_logevent(_es_self, event):
                    print(event)
                    if str(event).startswith("Started evolutionary loop") or str(event).startswith("Generation "):
                        checkpoint_new_rows()

                es.logevent = MethodType(patched_logevent, es)

                def patched_initialize(_es_self):
                    population = _es_self.population
                    jobs = [delayed(_es_self.initialize_single)() for _ in range(_es_self.n_parents - len(population))]
                    if _es_self.max_workers <= 1:
                        population_gen = [_es_self.initialize_single() for _ in range(_es_self.n_parents - len(population))]
                    else:
                        population_gen = Parallel(
                            n_jobs=_es_self.max_workers,
                            backend=_es_self.parallel_backend,
                            timeout=_es_self.eval_timeout + 15,
                            return_as="generator_unordered",
                        )(jobs)
                    for p in population_gen:
                        population.append(p)
                    if _es_self.evaluate_population:
                        population = _es_self.evaluate_population_fitness(population)
                    population = _es_self._ensure_fitness_evaluates(population)
                    for p in population:
                        _es_self.run_history.append(p)
                    if _es_self.niching == "novelty":
                        population = _es_self.apply_niching(population)
                    _es_self.generation += 1
                    _es_self.population = population
                    _es_self.update_best()

                es.initialize = MethodType(patched_initialize, es)

                def patched_run(_es_self, archive_path=None):
                    if archive_path is not None:
                        _es_self.logevent(f"Loading population from {archive_path}/log.jsonl...")
                        _es_self.get_population_from(archive_path)
                    else:
                        _es_self.logevent("No archive path provided, standard initialisation.")
                        _es_self.logevent("Initializing first population")
                        _es_self.initialize()

                    if _es_self.log:
                        _es_self.logger.log_population(_es_self.population)

                    if hasattr(_es_self.best_so_far, "fitness"):
                        _es_self.logevent(f"Started evolutionary loop, best so far: {_es_self.best_so_far.fitness}")
                    else:
                        _es_self.logevent("Started evolutionary loop")

                    if _es_self.feature_guided_mutation:
                        _es_self._update_feature_guidance()

                    while len(_es_self.run_history) < _es_self.budget:
                        new_offspring_population = _es_self._select_parents()
                        if _es_self.max_workers <= 1:
                            new_population = [_es_self.evolve_solution(individual) for individual in new_offspring_population]
                        else:
                            new_population_gen = Parallel(
                                n_jobs=_es_self.max_workers,
                                timeout=_es_self.eval_timeout + 15,
                                backend=_es_self.parallel_backend,
                                return_as="generator_unordered",
                            )(
                                delayed(_es_self.evolve_solution)(individual)
                                for individual in new_offspring_population
                            )
                            new_population = []
                            for candidate in new_population_gen:
                                if math.isnan(candidate.fitness):
                                    candidate.fitness = _es_self.worst_value
                                new_population.append(candidate)

                        if _es_self.evaluate_population:
                            new_population = _es_self.evaluate_population_fitness(new_population)

                        new_population = _es_self._ensure_fitness_evaluates(new_population)
                        for individual in new_population:
                            _es_self.run_history.append(individual)

                        _es_self.generation += 1
                        if _es_self.log:
                            _es_self.logger.log_population(new_population)
                        _es_self.population = _es_self.selection(_es_self.population, new_population)
                        _es_self.update_best()
                        if hasattr(_es_self.best_so_far, "fitness"):
                            _es_self.logevent(f"Generation {_es_self.generation}, best so far: {_es_self.best_so_far.fitness}")
                        else:
                            _es_self.logevent(f"Generation {_es_self.generation}")
                        if _es_self.feature_guided_mutation:
                            _es_self._update_feature_guidance()

                    return _es_self.best_so_far

                es.run = MethodType(patched_run, es)
                flush_live(es)
                best = es.run()
                checkpoint_new_rows()
                best_score = safe_float(getattr(best, "fitness", None)) if not isinstance(best, list) else None
                best_raw = safe_float(getattr(best, "metadata", {}).get("raw_objective_mean")) if hasattr(best, "metadata") else None
                artifact_dir = str(Path(getattr(getattr(es, "logger", None), "dirname", run_dir)).resolve())

            runtime_sec = monitor.runtime_sec
            peak_rss_mb = monitor.peak_rss_mb

            if best_score is None or best_score <= PENALTY_SCORE + 1e-9:
                status = "no_valid_candidate"
                notes = "All generated candidates fell back to the penalty baseline. See failure_examples.txt for examples."
            elif failure_log_path.exists():
                notes = "Some candidates failed validation or execution. See failure_examples.txt for examples."

        except Exception as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            flush_live(None)

        summary = RunSummary(
            framework="llamea",
            benchmark=task.name,
            seed=seed,
            status=status,
            best_search_score=best_score,
            raw_objective_mean=best_raw,
            runtime_sec=runtime_sec,
            peak_rss_mb=peak_rss_mb,
            candidates_evaluated=len(progress_rows),
            artifact_dir=artifact_dir,
            notes=notes,
        )
        _write_json(summary_live_path, {
            "framework": summary.framework,
            "benchmark": summary.benchmark,
            "seed": summary.seed,
            "status": summary.status,
            "best_search_score": summary.best_search_score,
            "raw_objective_mean": summary.raw_objective_mean,
            "runtime_sec": summary.runtime_sec,
            "peak_rss_mb": summary.peak_rss_mb,
            "candidates_evaluated": summary.candidates_evaluated,
            "artifact_dir": summary.artifact_dir,
            "notes": summary.notes,
        })
        return summary, progress_rows

"""Main entry point for the benchmark harness with unattended-safe process isolation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_harness.adapters import EASEAdapter, LLaMEAAdapter, LLM4ADAdapter
from benchmark_harness.config import load_config, resolve_path
from benchmark_harness.tasks import TASK_REGISTRY, get_task
from benchmark_harness.utils import (
    ResultStore,
    atomic_write_json,
    child_creation_kwargs,
    compute_stats,
    get_logger,
    terminate_process_tree,
    timestamp,
)

logger = get_logger("run_benchmark", level=logging.INFO)

ADAPTER_MAP = {
    "llamea": LLaMEAAdapter,
    "llm4ad": LLM4ADAdapter,
    "ease": EASEAdapter,
}


def build_adapters(cfg: dict, enabled_filter: list[str] | None) -> list:
    adapters = []
    frameworks_cfg = cfg.get("frameworks", {})
    for fw_name, adapter_cls in ADAPTER_MAP.items():
        fw_cfg = frameworks_cfg.get(fw_name, {})
        adapter = adapter_cls(framework_cfg=fw_cfg, global_cfg=cfg)
        if enabled_filter and fw_name not in enabled_filter:
            logger.info("  ⏭  %s: excluded by --frameworks flag — skipping", fw_name)
            continue
        if not adapter.is_enabled():
            logger.info("  ⏭  %s: disabled in config — skipping", fw_name)
            continue
        adapters.append(adapter)
        logger.info("  ✅  %s: enabled", fw_name)
    return adapters


def print_summary(store: ResultStore, frameworks: list, tasks: list[str], seeds: list[int]):
    separator = "─" * 72
    print(f"\n{separator}")
    print("BENCHMARK SUMMARY")
    print(separator)
    header = f"{'Framework':<12} {'Task':<18} {'Seeds':>6} {'Mean':>12} {'Std':>10} {'Best':>12}"
    print(header)
    print(separator)

    for adapter in frameworks:
        fw = adapter.name
        for task_name in tasks:
            records = store.read_all(fw, task_name)
            vals = [
                r["best_value"]
                for r in records
                if r.get("seed") in seeds and r.get("best_value") is not None and r.get("success")
            ]
            if vals:
                stats = compute_stats(vals)
                print(
                    f"{fw:<12} {task_name:<18} {len(vals):>6} "
                    f"{stats['mean']:>12.4f} {stats['std']:>10.4f} {stats['min']:>12.4f}"
                )
            else:
                print(f"{fw:<12} {task_name:<18} {'N/A':>6}")

    print(separator)


def _tail_text(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return text[-max_chars:]


def _build_failure_result(framework: str, task: str, seed: int, error: str, elapsed: float | None = None) -> dict:
    return {
        "framework": framework,
        "task": task,
        "seed": seed,
        "best_value": None,
        "success": False,
        "error": error,
        "elapsed_sec": elapsed,
        "timestamp": timestamp(),
        "extra": {},
    }


def _write_heartbeat(output_dir: Path, payload: dict) -> None:
    atomic_write_json(output_dir / "heartbeat.json", payload)


def _worker_main(worker_json_path: str) -> int:
    worker_payload = json.loads(Path(worker_json_path).read_text(encoding="utf-8"))
    config_path = Path(worker_payload["config_path"]).resolve()
    framework = worker_payload["framework"]
    task_name = worker_payload["task"]
    seed = int(worker_payload["seed"])
    result_json = Path(worker_payload["result_json"])

    warnings.filterwarnings("ignore", message=".*SequentialBackend.*timeout.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    cfg = load_config(str(config_path))
    cfg["_config_dir"] = str(config_path.parent)
    cfg["_output_dir_resolved"] = str(resolve_path(config_path.parent, cfg.get("output_dir", "../benchmark_results")))

    fw_cfg = cfg.get("frameworks", {}).get(framework, {})
    adapter = ADAPTER_MAP[framework](framework_cfg=fw_cfg, global_cfg=cfg)
    task_defaults = cfg.get("task_defaults", {})
    task = get_task(task_name, task_defaults.get("lower_bound", -5.0), task_defaults.get("upper_bound", 5.0))

    try:
        result = adapter.run(task, seed)
    except Exception:
        result = _build_failure_result(framework, task_name, seed, traceback.format_exc())

    atomic_write_json(result_json, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("success") else 1


def _run_isolated_experiment(
    *,
    script_path: Path,
    config_path: Path,
    output_dir: Path,
    framework: str,
    task_name: str,
    seed: int,
    timeout_sec: int,
) -> dict:
    logs_dir = output_dir / "logs"
    tmp_dir = output_dir / "_tmp"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{framework}__{task_name}__seed{seed}"
    log_path = logs_dir / f"{stem}.log"
    result_path = tmp_dir / f"{stem}.json"
    payload_path = tmp_dir / f"{stem}.payload.json"

    payload = {
        "config_path": str(config_path),
        "framework": framework,
        "task": task_name,
        "seed": seed,
        "result_json": str(result_path),
    }
    atomic_write_json(payload_path, payload)

    start = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log_file:
        import subprocess

        proc = subprocess.Popen(
            [sys.executable, str(script_path), "--worker-json", str(payload_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(script_path.parent),
            text=True,
            **child_creation_kwargs(),
        )
        try:
            return_code = proc.wait(timeout=timeout_sec)
        except Exception:
            terminate_process_tree(proc)
            elapsed = round(time.perf_counter() - start, 3)
            return _build_failure_result(
                framework,
                task_name,
                seed,
                f"Run timed out after {timeout_sec}s. See log: {log_path}",
                elapsed,
            )

    elapsed = round(time.perf_counter() - start, 3)
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            result = _build_failure_result(
                framework,
                task_name,
                seed,
                f"Worker produced unreadable result file. See log: {log_path}",
                elapsed,
            )
    else:
        tail = _tail_text(log_path)
        result = _build_failure_result(
            framework,
            task_name,
            seed,
            f"Worker exited with code {return_code} and no result file. Log tail:\n{tail}",
            elapsed,
        )

    if result.get("elapsed_sec") is None:
        result["elapsed_sec"] = elapsed
    result.setdefault("extra", {})
    result["extra"]["log_path"] = str(log_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM evolutionary algorithm benchmarks.")
    parser.add_argument("--config", default="benchmark_config.json", help="Path to benchmark_config.json")
    parser.add_argument("--frameworks", nargs="+", choices=list(ADAPTER_MAP.keys()))
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_json:
        return _worker_main(args.worker_json)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(args.config).resolve()
    logger.info("Loading config: %s", config_path)
    cfg = load_config(str(config_path))
    cfg["_config_dir"] = str(config_path.parent)

    output_dir = resolve_path(config_path.parent, cfg.get("output_dir", "../benchmark_results"))
    cfg["_output_dir_resolved"] = str(output_dir)

    runner_cfg = cfg.get("runner", {})
    isolate_each_run = bool(runner_cfg.get("isolate_each_run", True))
    per_run_timeout_sec = int(runner_cfg.get("per_run_timeout_sec", 7200))
    skip_completed = bool(runner_cfg.get("skip_completed", True))

    seeds = args.seeds if args.seeds else cfg.get("seeds", [0])
    task_names = args.tasks if args.tasks else cfg.get("tasks", list(TASK_REGISTRY.keys()))
    task_defaults = cfg.get("task_defaults", {})
    lb = task_defaults.get("lower_bound", -5.0)
    ub = task_defaults.get("upper_bound", 5.0)

    logger.info("Seeds:    %s", seeds)
    logger.info("Tasks:    %s", task_names)
    logger.info("Ollama:   %s @ %s", cfg["ollama"]["model"], cfg["ollama"]["base_url"])
    logger.info("Output:   %s", output_dir)
    logger.info("Initialising framework adapters:")
    adapters = build_adapters(cfg, args.frameworks)

    if not adapters:
        logger.error("No frameworks are enabled. Check your config or --frameworks flag.")
        return 1

    if args.dry_run:
        logger.info("DRY RUN — all checks passed. Not running any experiments.")
        total = len(adapters) * len(task_names) * len(seeds)
        logger.info("Would run %s experiments.", total)
        return 0

    store = ResultStore(str(output_dir), append=cfg.get("append_results", False))

    total = len(adapters) * len(task_names) * len(seeds)
    completed = 0
    failed = 0
    skipped = 0
    run_start = time.perf_counter()

    logger.info("\n%s", "=" * 60)
    logger.info("Starting %s experiment(s)", total)
    logger.info("%s\n", "=" * 60)

    script_path = Path(__file__).resolve()

    for adapter in adapters:
        for task_name in task_names:
            task = get_task(task_name, lb, ub)
            for seed in seeds:
                completed += 1
                label = f"[{completed}/{total}] {adapter.name} | {task_name} | seed={seed}"
                logger.info("\n%s\n%s", "─" * 60, label)

                if skip_completed and store.has_completed_record(adapter.name, task_name, seed):
                    skipped += 1
                    logger.info("  ⏭  Skipping existing result")
                    continue

                _write_heartbeat(
                    output_dir,
                    {
                        "status": "running",
                        "framework": adapter.name,
                        "task": task_name,
                        "seed": seed,
                        "started_at": timestamp(),
                        "completed_counter": completed,
                        "total": total,
                    },
                )

                if isolate_each_run:
                    result = _run_isolated_experiment(
                        script_path=script_path,
                        config_path=config_path,
                        output_dir=output_dir,
                        framework=adapter.name,
                        task_name=task.name,
                        seed=seed,
                        timeout_sec=per_run_timeout_sec,
                    )
                else:
                    try:
                        result = adapter.run(task, seed)
                    except Exception:
                        result = _build_failure_result(adapter.name, task_name, seed, traceback.format_exc())

                store.write(adapter.name, task_name, result)

                if result.get("success"):
                    val = result.get("best_value")
                    elapsed = result.get("elapsed_sec", 0)
                    if val is not None:
                        logger.info("  ✅  best_value=%0.6f  elapsed=%0.1fs", val, elapsed)
                    else:
                        logger.info("  ✅  done  elapsed=%0.1fs", elapsed)
                else:
                    failed += 1
                    err = str(result.get("error", ""))[:240]
                    logger.warning("  ❌  FAILED: %s", err)

    total_elapsed = time.perf_counter() - run_start
    _write_heartbeat(
        output_dir,
        {
            "status": "finished",
            "finished_at": timestamp(),
            "failed": failed,
            "skipped": skipped,
            "total_time_sec": round(total_elapsed, 3),
        },
    )

    logger.info("\n%s", "=" * 60)
    logger.info(
        "Finished: %s runs, %s failed, %s skipped, total time %0.1fs",
        completed,
        failed,
        skipped,
        total_elapsed,
    )
    logger.info("Results written to: %s", output_dir)
    logger.info("%s", "=" * 60)

    print_summary(store, adapters, task_names, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

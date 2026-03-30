from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from .adapters import EaseAdapter, LLM4ADAdapter, LLAMEAAdapter
from .config import load_config
from .tasks import builtin_task_specs
from .utils import append_csv_row, ensure_dir


PROGRESS_FIELDS = [
    "framework",
    "benchmark",
    "seed",
    "sample_index",
    "elapsed_sec",
    "candidate_score",
    "best_so_far",
    "is_valid_candidate",
    "fail_reason",
]


def build_adapters(cfg: dict, output_dir: Path):
    adapters = []
    fw = cfg["frameworks"]
    if fw.get("llamea", {}).get("enabled", False):
        adapters.append(LLAMEAAdapter(fw["llamea"], cfg, output_dir))
    if fw.get("llm4ad", {}).get("enabled", False):
        adapters.append(LLM4ADAdapter(fw["llm4ad"], cfg, output_dir))
    if fw.get("ease", {}).get("enabled", False):
        adapters.append(EaseAdapter(fw["ease"], cfg, output_dir))
    return adapters


def run_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--append", action="store_true", help="Append to existing CSV files instead of replacing them")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg.get("output_dir", "benchmark_results"))

    summary_csv = output_dir / "summary.csv"
    progress_csv = output_dir / "progress.csv"
    append_mode = bool(args.append or cfg.get("append_results", False))
    if not append_mode:
        if summary_csv.exists():
            summary_csv.unlink()
        if progress_csv.exists():
            progress_csv.unlink()

    task_specs = builtin_task_specs(cfg.get("task_defaults", {}))
    tasks = [task_specs[name] for name in cfg.get("tasks", [])]
    seeds = [int(x) for x in cfg.get("seeds", [0])]

    adapters = build_adapters(cfg, output_dir)

    for adapter in adapters:
        for task in tasks:
            for seed in seeds:
                print(f"Running {adapter.framework_name} | task={task.name} | seed={seed}")
                summary, progress_rows = adapter.run_one(task, seed)
                append_csv_row(summary_csv, asdict(summary), fieldnames=list(asdict(summary).keys()))
                for row in progress_rows:
                    append_csv_row(progress_csv, row, fieldnames=PROGRESS_FIELDS)

    print(f"Results written to: {output_dir}")
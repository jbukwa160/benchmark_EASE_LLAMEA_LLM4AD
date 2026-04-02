"""
analyze_benchmark.py — Load, aggregate, and visualise benchmark results.

Usage:
    python analyze_benchmark.py                        # reads ../benchmark_results
    python analyze_benchmark.py --results-dir ./my_results
    python analyze_benchmark.py --plot                 # save convergence plots
    python analyze_benchmark.py --export summary.csv   # export CSV summary
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_harness.utils import get_logger, ResultStore, compute_stats

logger = get_logger("analyze_benchmark")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> list[dict]:
    """Load every .jsonl file from results_dir into a flat list of records."""
    records = []
    p = Path(results_dir)
    if not p.exists():
        logger.error(f"Results directory not found: {p}")
        return records

    jsonl_files = list(p.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {p}")
        return records

    for fpath in sorted(jsonl_files):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    logger.info(f"Loaded {len(records)} records from {len(jsonl_files)} files")
    return records


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(records: list[dict]):
    """Print a grouped summary: framework × task → stats over seeds."""
    from collections import defaultdict

    groups: dict[tuple, list[float]] = defaultdict(list)
    failed: dict[tuple, int] = defaultdict(int)

    for r in records:
        key = (r.get("framework", "?"), r.get("task", "?"))
        if r.get("success") and r.get("best_value") is not None:
            groups[key].append(float(r["best_value"]))
        else:
            failed[key] += 1

    all_keys = sorted(set(list(groups.keys()) + list(failed.keys())))

    sep = "─" * 80
    print(f"\n{sep}")
    print("BENCHMARK RESULTS — SUMMARY")
    print(sep)
    print(
        f"{'Framework':<12} {'Task':<20} {'N':>4} {'Mean':>12} "
        f"{'Std':>10} {'Best':>12} {'Fail':>6}"
    )
    print(sep)

    for key in all_keys:
        fw, task = key
        vals = groups.get(key, [])
        n_fail = failed.get(key, 0)
        if vals:
            stats = compute_stats(vals)
            print(
                f"{fw:<12} {task:<20} {len(vals):>4} "
                f"{stats['mean']:>12.6f} {stats['std']:>10.6f} "
                f"{stats['min']:>12.6f} {n_fail:>6}"
            )
        else:
            print(f"{fw:<12} {task:<20} {'0':>4} {'N/A':>12} {'':>10} {'':>12} {n_fail:>6}")

    print(sep)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(records: list[dict], path: str):
    if not HAS_PANDAS:
        # Manual CSV
        import csv
        fields = ["framework", "task", "seed", "best_value", "success",
                  "elapsed_sec", "error", "timestamp"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(records)
        logger.info(f"Exported CSV (no pandas): {path}")
        return

    df = pd.json_normalize(records)
    df.to_csv(path, index=False)
    logger.info(f"Exported CSV: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(records: list[dict], output_dir: str):
    if not HAS_MPL:
        logger.warning("matplotlib not installed — skipping plots.")
        return

    from collections import defaultdict
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group by task
    tasks = sorted(set(r.get("task", "?") for r in records))
    frameworks = sorted(set(r.get("framework", "?") for r in records))

    for task_name in tasks:
        fig, ax = plt.subplots(figsize=(8, 5))
        has_data = False

        for fw in frameworks:
            vals = [
                r["best_value"]
                for r in records
                if r.get("framework") == fw
                and r.get("task") == task_name
                and r.get("success")
                and r.get("best_value") is not None
            ]
            if not vals:
                continue
            has_data = True
            seeds = [
                r["seed"]
                for r in records
                if r.get("framework") == fw
                and r.get("task") == task_name
                and r.get("success")
                and r.get("best_value") is not None
            ]
            ax.scatter(seeds, vals, label=fw, alpha=0.7, s=60)
            mean_val = float(np.mean(vals))
            ax.axhline(mean_val, linestyle="--", alpha=0.4,
                       color=ax.get_lines()[-1].get_color() if ax.get_lines() else None)

        if has_data:
            ax.set_title(f"Best objective value — {task_name}")
            ax.set_xlabel("Seed")
            ax.set_ylabel("Best objective (lower = better)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_path = out / f"results_{task_name}.png"
            fig.tight_layout()
            fig.savefig(str(plot_path), dpi=150)
            logger.info(f"  Saved: {plot_path}")

        plt.close(fig)

    # Bar chart: mean best per (framework, task)
    from collections import defaultdict
    means: dict[str, dict[str, float]] = defaultdict(dict)
    stds: dict[str, dict[str, float]] = defaultdict(dict)

    for fw in frameworks:
        for task_name in tasks:
            vals = [
                r["best_value"]
                for r in records
                if r.get("framework") == fw
                and r.get("task") == task_name
                and r.get("success")
                and r.get("best_value") is not None
            ]
            if vals:
                means[fw][task_name] = float(np.mean(vals))
                stds[fw][task_name] = float(np.std(vals))

    if means:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(tasks))
        width = 0.8 / max(len(frameworks), 1)

        for i, fw in enumerate(frameworks):
            fw_means = [means[fw].get(t, float("nan")) for t in tasks]
            fw_stds  = [stds[fw].get(t, 0.0)          for t in tasks]
            offset = (i - len(frameworks) / 2 + 0.5) * width
            bars = ax.bar(x + offset, fw_means, width * 0.9, label=fw,
                          yerr=fw_stds, capsize=4, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=15, ha="right")
        ax.set_ylabel("Mean best objective (lower = better)")
        ax.set_title("Framework comparison — mean best objective per task")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        bar_path = out / "comparison_bar.png"
        fig.savefig(str(bar_path), dpi=150)
        logger.info(f"  Saved: {bar_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse LLM evolutionary benchmark results."
    )
    parser.add_argument(
        "--results-dir", default="../benchmark_results",
        help="Directory containing .jsonl result files."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save matplotlib plots to the results directory."
    )
    parser.add_argument(
        "--export", metavar="CSV_PATH",
        help="Export all records to a CSV file."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true"
    )
    args = parser.parse_args()

    records = load_all_results(args.results_dir)
    if not records:
        logger.warning("No records found — nothing to analyse.")
        return

    print_summary_table(records)

    if args.export:
        export_csv(records, args.export)

    if args.plot:
        logger.info("Generating plots …")
        plot_results(records, args.results_dir)


if __name__ == "__main__":
    main()

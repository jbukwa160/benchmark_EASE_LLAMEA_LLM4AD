from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import ensure_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    out_dir = ensure_dir(results_dir / "analysis")

    rows = []
    for path in results_dir.rglob("seed_*.json"):
        if path.parent.name == "analysis":
            continue
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        raise SystemExit("No result files found")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_results.csv", index=False)

    summary = (
        df.groupby(["framework", "task"], dropna=False)
        .agg(
            trials=("seed", "count"),
            ok_trials=("status", lambda s: int((s == "ok").sum())),
            mean_score=("score", "mean"),
            mean_best_f=("best_f", "mean"),
            median_best_f=("best_f", "median"),
            mean_duration_seconds=("duration_seconds", "mean"),
        )
        .reset_index()
    )
    summary["success_rate"] = summary["ok_trials"] / summary["trials"]
    summary.to_csv(out_dir / "summary_by_framework_task.csv", index=False)

    score_pivot = summary.pivot(index="task", columns="framework", values="mean_score")
    score_pivot.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Mean score")
    plt.title("Mean score by framework and task")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_score_by_framework_task.png", dpi=160)
    plt.close()

    success_pivot = summary.pivot(index="task", columns="framework", values="success_rate")
    success_pivot.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Success rate")
    plt.title("Success rate by framework and task")
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate_by_framework_task.png", dpi=160)
    plt.close()

    duration_pivot = summary.pivot(index="task", columns="framework", values="mean_duration_seconds")
    duration_pivot.plot(kind="bar", figsize=(10, 6))
    plt.ylabel("Mean duration (s)")
    plt.title("Mean duration by framework and task")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_duration_by_framework_task.png", dpi=160)
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

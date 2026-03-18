from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def analyze_results(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.csv"
    progress_path = results_dir / "progress.csv"

    summary = pd.read_csv(summary_path)
    progress = pd.read_csv(progress_path) if progress_path.exists() and progress_path.stat().st_size > 0 else pd.DataFrame()

    plots_dir = ensure_dir(results_dir / "plots")

    agg = (
        summary.groupby(["framework", "benchmark"], dropna=False)
        .agg(
            runs=("seed", "count"),
            success_rate=("status", lambda s: (s == "success").mean()),
            mean_best_search_score=("best_search_score", "mean"),
            std_best_search_score=("best_search_score", "std"),
            mean_raw_objective=("raw_objective_mean", "mean"),
            std_raw_objective=("raw_objective_mean", "std"),
            mean_runtime_sec=("runtime_sec", "mean"),
            std_runtime_sec=("runtime_sec", "std"),
            mean_peak_rss_mb=("peak_rss_mb", "mean"),
            mean_candidates=("candidates_evaluated", "mean"),
        )
        .reset_index()
        .sort_values(["benchmark", "mean_best_search_score"], ascending=[True, False])
    )
    agg.to_csv(results_dir / "aggregate_summary.csv", index=False)

    if not progress.empty:
        for benchmark, group in progress.groupby("benchmark"):
            plt.figure()
            for framework, fg in group.groupby("framework"):
                line = (
                    fg.groupby("sample_index", dropna=False)["best_so_far"]
                    .mean()
                    .reset_index()
                    .sort_values("sample_index")
                )
                plt.plot(line["sample_index"], line["best_so_far"], label=framework)
            plt.xlabel("Candidate index")
            plt.ylabel("Best-so-far search score")
            plt.title(f"Convergence: {benchmark}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"convergence_{benchmark}.png")
            plt.close()

    for benchmark, group in summary.groupby("benchmark"):
        plt.figure()
        ordered_frameworks = list(group.sort_values("framework")["framework"].unique())
        values = [group[group["framework"] == fw]["best_search_score"].dropna().tolist() for fw in ordered_frameworks]
        if any(len(v) > 0 for v in values):
            plt.boxplot(values, tick_labels=ordered_frameworks)
            plt.ylabel("Best search score")
            plt.title(f"Distribution of final search score: {benchmark}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"boxplot_{benchmark}.png")
        plt.close()

    print(f"Wrote aggregate summary to: {results_dir / 'aggregate_summary.csv'}")
    print(f"Wrote plots to: {plots_dir}")


def analyze_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    analyze_results(args.results_dir)

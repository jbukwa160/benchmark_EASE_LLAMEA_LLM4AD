from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .tasks import score_from_best_f
from .utils import ensure_dir


PENALTY_SCORE = score_from_best_f(1e12)


def _safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _plot_grouped_bars(pivot: pd.DataFrame, ylabel: str, title: str, output_path: Path) -> None:
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Framework")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_results(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.csv"
    progress_path = results_dir / "progress.csv"

    summary = pd.read_csv(summary_path)
    progress = pd.read_csv(progress_path) if progress_path.exists() and progress_path.stat().st_size > 0 else pd.DataFrame()
    summary = _safe_numeric(summary, ["seed", "best_search_score", "raw_objective_mean", "runtime_sec", "peak_rss_mb", "candidates_evaluated"])
    progress = _safe_numeric(progress, ["seed", "sample_index", "elapsed_sec", "candidate_score", "best_so_far"]) if not progress.empty else progress

    summary["completed_ok"] = summary["status"].isin(["success", "no_valid_candidate"])
    summary["valid_run"] = summary["completed_ok"] & (summary["best_search_score"] > PENALTY_SCORE + 1e-9)
    summary["penalty_run"] = summary["completed_ok"] & ~summary["valid_run"]

    plots_dir = ensure_dir(results_dir / "plots")

    agg = (
        summary.groupby(["framework", "benchmark"], dropna=False)
        .agg(
            runs=("seed", "count"),
            completed_runs=("completed_ok", "sum"),
            valid_runs=("valid_run", "sum"),
            completion_rate=("completed_ok", "mean"),
            valid_rate=("valid_run", "mean"),
            penalty_rate=("penalty_run", "mean"),
            mean_best_search_score=("best_search_score", "mean"),
            median_best_search_score=("best_search_score", "median"),
            std_best_search_score=("best_search_score", "std"),
            mean_raw_objective=("raw_objective_mean", "mean"),
            std_raw_objective=("raw_objective_mean", "std"),
            mean_runtime_sec=("runtime_sec", "mean"),
            std_runtime_sec=("runtime_sec", "std"),
            mean_peak_rss_mb=("peak_rss_mb", "mean"),
            mean_candidates=("candidates_evaluated", "mean"),
        )
        .reset_index()
        .sort_values(["benchmark", "valid_rate", "mean_best_search_score"], ascending=[True, False, False])
    )
    agg.to_csv(results_dir / "aggregate_summary.csv", index=False)

    # Human-readable report table for presentations.
    presentation = agg[[
        "framework",
        "benchmark",
        "runs",
        "valid_runs",
        "valid_rate",
        "mean_best_search_score",
        "mean_raw_objective",
        "mean_runtime_sec",
        "mean_candidates",
    ]].copy()
    presentation["valid_rate_pct"] = (presentation["valid_rate"] * 100.0).round(1)
    presentation.to_csv(results_dir / "presentation_summary.csv", index=False)

    if not progress.empty:
        progress["is_valid_candidate"] = progress.get("is_valid_candidate", False)
        progress["is_valid_candidate"] = progress["is_valid_candidate"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
        for benchmark, group in progress.groupby("benchmark"):
            plt.figure(figsize=(8, 5))
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
            plt.title(f"Convergence on {benchmark}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"convergence_{benchmark}.png")
            plt.close()

            valid_counts = (
                group.groupby("framework")["is_valid_candidate"].mean().sort_index() * 100.0
            )
            plt.figure(figsize=(7, 4))
            plt.bar(valid_counts.index.astype(str), valid_counts.values)
            plt.ylabel("Valid candidate rate (%)")
            plt.title(f"Candidate validity on {benchmark}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"candidate_validity_{benchmark}.png")
            plt.close()

    for benchmark, group in summary.groupby("benchmark"):
        plt.figure(figsize=(7, 4))
        ordered_frameworks = list(group.sort_values("framework")["framework"].unique())
        values = [group[group["framework"] == fw]["best_search_score"].dropna().tolist() for fw in ordered_frameworks]
        if any(len(v) > 0 for v in values):
            plt.boxplot(values, tick_labels=ordered_frameworks)
            plt.ylabel("Best search score")
            plt.title(f"Final score distribution on {benchmark}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"boxplot_{benchmark}.png")
        plt.close()

    success_pivot = agg.pivot(index="benchmark", columns="framework", values="valid_rate").fillna(0.0) * 100.0
    _plot_grouped_bars(success_pivot, "Valid run rate (%)", "Valid run rate by benchmark", plots_dir / "valid_run_rate_by_benchmark.png")

    score_pivot = agg.pivot(index="benchmark", columns="framework", values="mean_best_search_score")
    _plot_grouped_bars(score_pivot, "Mean best search score", "Mean final score by benchmark", plots_dir / "mean_score_by_benchmark.png")

    runtime_pivot = agg.pivot(index="benchmark", columns="framework", values="mean_runtime_sec")
    _plot_grouped_bars(runtime_pivot, "Mean runtime (sec)", "Mean runtime by benchmark", plots_dir / "mean_runtime_by_benchmark.png")

    candidates_pivot = agg.pivot(index="benchmark", columns="framework", values="mean_candidates")
    _plot_grouped_bars(candidates_pivot, "Mean candidates evaluated", "Search effort by benchmark", plots_dir / "mean_candidates_by_benchmark.png")

    overall = (
        summary.groupby("framework")
        .agg(
            valid_rate=("valid_run", "mean"),
            mean_best_search_score=("best_search_score", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_candidates=("candidates_evaluated", "mean"),
        )
        .reset_index()
        .sort_values("mean_best_search_score", ascending=False)
    )
    overall.to_csv(results_dir / "overall_framework_summary.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(overall["framework"], overall["valid_rate"] * 100.0)
    plt.ylabel("Valid run rate (%)")
    plt.title("Overall valid run rate")
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_valid_run_rate.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(overall["framework"], overall["mean_best_search_score"])
    plt.ylabel("Mean best search score")
    plt.title("Overall mean final score")
    plt.tight_layout()
    plt.savefig(plots_dir / "overall_mean_final_score.png")
    plt.close()

    report_lines = [
        "# Benchmark Results Summary",
        "",
        f"Penalty baseline score: {PENALTY_SCORE:.1f}",
        "",
        "## Per framework and benchmark",
        presentation.to_markdown(index=False),
        "",
        "## Overall framework summary",
        overall.to_markdown(index=False),
        "",
        "A run is counted as valid when it completed and its best score beat the penalty baseline.",
    ]
    (results_dir / "presentation_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote aggregate summary to: {results_dir / 'aggregate_summary.csv'}")
    print(f"Wrote presentation summary to: {results_dir / 'presentation_summary.csv'}")
    print(f"Wrote plots to: {plots_dir}")


def analyze_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    analyze_results(args.results_dir)
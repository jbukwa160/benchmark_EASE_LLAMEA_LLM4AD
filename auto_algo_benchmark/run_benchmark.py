import argparse
import subprocess
import sys
from pathlib import Path

from benchmark_harness.cli import run_cli
from benchmark_harness.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    sys.argv = ["run_benchmark.py", "--config", args.config]
    if args.append:
        sys.argv.append("--append")
    run_cli()

    if not args.skip_analysis:
        config_path = Path(args.config).resolve()
        output_dir_cfg = cfg.get("output_dir", "benchmark_results")
        results_dir = (config_path.parent / output_dir_cfg).resolve()

        print(f"Benchmark finished. Running analysis on: {results_dir}")

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("analyze_benchmark.py")),
                "--results-dir",
                str(results_dir),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
import argparse
import subprocess
import sys
from pathlib import Path

from benchmark_harness.cli import run_cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    # Run benchmark first
    sys.argv = ["run_benchmark.py", "--config", args.config]
    if args.append:
        sys.argv.append("--append")

    run_cli()

    # Then run analysis automatically
    if not args.skip_analysis:
        config_path = Path(args.config).resolve()

        # results dir matches your config style: "../benchmark_results"
        results_dir = (config_path.parent / "../benchmark_results").resolve()

        print(f"\nBenchmark finished. Running analysis on: {results_dir}\n")

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
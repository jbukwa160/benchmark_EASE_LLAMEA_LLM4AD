#!/usr/bin/env bash
set -euo pipefail
python -m benchmark_redesign.run_benchmark configs/default_benchmark.json
python -m benchmark_redesign.analyze_benchmark ./benchmark_results

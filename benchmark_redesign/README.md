# Benchmark redesign

This is a clean benchmark runner built to stop the exact failure mode you were hitting:

- a blocked Ollama call can no longer freeze the whole experiment
- one bad sample cannot corrupt the whole run
- one timed-out trial becomes a failed JSON result, not a stuck benchmark
- malformed model output no longer crashes on `index out of range`
- result files are written atomically, so you do not end up with half-written junk

## What changed

Instead of relying on the framework-local Ollama wrappers, this redesign uses a single hard-timeout Ollama client and runs every trial in a separate worker process.

Architecture:

1. `benchmark_redesign.run_benchmark` is the parent orchestrator.
2. Every `(framework, task, seed)` trial runs in a separate child process.
3. Each child uses a hard-timeout Ollama HTTP client.
4. If a child hangs, the parent kills it and writes a failed result JSON.
5. `analyze_benchmark.py` turns the JSON results into CSV files and PNG plots.

## Supported frameworks

- `llamea`
- `llm4ad_eoh`

The design is extensible, but I intentionally kept the first version smaller and harder to break.

## Folder layout expected

Keep the repos next to this benchmark folder, for example:

```text
workspace/
  benchmark_redesign/
  LLaMEA-main/
  LLM4AD-main/
```

Then the default config paths work as-is.

## Run

From inside `benchmark_redesign/`:

```bash
python -m benchmark_redesign.run_benchmark configs/default_benchmark.json
```

Then analyze results:

```bash
python -m benchmark_redesign.analyze_benchmark ./benchmark_results
```

## Output

Per-trial result files:

```text
benchmark_results/<framework>/<task>/seed_<seed>.json
```

Analysis files:

```text
benchmark_results/analysis/raw_results.csv
benchmark_results/analysis/summary_by_framework_task.csv
benchmark_results/analysis/mean_score_by_framework_task.png
benchmark_results/analysis/success_rate_by_framework_task.png
benchmark_results/analysis/mean_duration_by_framework_task.png
```

## Notes

- This redesign bypasses the fragile local Ollama wrappers inside the frameworks.
- It does **not** require editing the upstream repos to get the timeout protection.
- `frontEASE-main` was not used because it is unrelated to the benchmark frameworks.

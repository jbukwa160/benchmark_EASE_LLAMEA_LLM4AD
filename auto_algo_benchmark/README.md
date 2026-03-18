# Auto Algorithm Benchmark Harness

This harness benchmarks three automatic algorithm-design systems on a shared result format:

- **LLaMEA**
- **LLM4AD** (default method: **EoH**)
- **EASE / frontEASE** through the running HTTP API

It is built to work with a local **Ollama** server for LLaMEA and LLM4AD, and with an already-running **frontEASE/EASE** stack for the EASE adapter.

## What it does

For every `(framework, benchmark task, seed)` combination it:

1. runs one framework search
2. records runtime and peak RSS memory
3. saves a standardized `summary.csv`
4. saves a standardized `progress.csv`
5. generates aggregate tables and plots with `analyze_benchmark.py`

## Shared output schema

### `summary.csv`

- `framework`
- `benchmark`
- `seed`
- `status`
- `best_search_score`
- `raw_objective_mean`
- `runtime_sec`
- `peak_rss_mb`
- `candidates_evaluated`
- `artifact_dir`
- `notes`

### `progress.csv`

- `framework`
- `benchmark`
- `seed`
- `sample_index`
- `elapsed_sec`
- `candidate_score`
- `best_so_far`

## Benchmarks included

The harness ships with a lightweight continuous black-box optimization benchmark where the searched program must implement:

```python
def solve(objective, budget, dim, lower_bound, upper_bound, seed):
    ...
    return {
        "best_x": ...,
        "best_f": ...,
        "history": ...
    }
```

Included objective presets:

- `sphere_5d`
- `rastrigin_5d`
- `rosenbrock_5d`
- `mixed_5d`

## Configuration

Copy `benchmark_config.example.json` and edit it.

Important settings:

- `frameworks.llamea.repo_path`
- `frameworks.llm4ad.repo_path`
- `frameworks.ease.api_base_url`
- `frameworks.ease.template_task_id`
- `ollama.model`

## Running

```bash
python run_benchmark.py --config benchmark_config.json
python analyze_benchmark.py --results-dir benchmark_results
```

## EASE / frontEASE notes

The uploaded `frontEASE-main` repo is only the UI/server shell. The optimization core lives in the missing `src/FoP_IMT.Core` submodule, so this harness talks to your **running** server over HTTP instead of importing backend Python code.

The EASE adapter therefore expects:

- the API to be reachable
- valid login credentials
- an existing **template task** that you can clone for each benchmark run

That avoids guessing the full task-config JSON schema from the frontend alone.

## Assumptions

- LLaMEA and LLM4AD are already installable in your environment or accessible through the provided repo paths.
- Ollama is running locally or reachable at the configured URL.
- EASE/frontEASE is already running and can execute cloned tasks.

## Recommended first run

Start with:

- one small model
- one benchmark task
- `seeds = [0]`
- low search budgets

Then scale up after you confirm all three adapters work end-to-end.

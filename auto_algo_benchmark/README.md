# LLM Evolutionary Benchmark Harness

A clean, config-driven harness for benchmarking **LLaMEA**, **LLM4AD**, and **frontEASE**
against continuous black-box optimisation tasks (Sphere, Rastrigin, Rosenbrock in 5D).
All frameworks are driven by the same Ollama LLM (configured in `benchmark_config.json`).

---

## Project layout

```
auto_algo_benchmark/
├── benchmark_config.json          ← your config file
├── run_benchmark.py               ← main runner
├── analyze_benchmark.py           ← result analysis & plots
├── smoke_test.py                  ← pre-flight checks
├── requirements.txt
└── benchmark_harness/
    ├── __init__.py
    ├── config.py                  ← config loader
    ├── tasks.py                   ← task definitions
    ├── utils.py                   ← logging, result store, stats
    └── adapters/
        ├── __init__.py
        ├── base.py                ← abstract adapter
        ├── llamea_adapter.py      ← LLaMEA adapter
        ├── llm4ad_adapter.py      ← LLM4AD adapter
        └── ease_adapter.py        ← frontEASE REST adapter
```

---

## Quick start

### 1. Install harness dependencies

```bash
pip install -r requirements.txt
```

### 2. Install framework repos

```bash
# LLaMEA
cd ../LLaMEA && pip install -e .   # or: pip install llamea

# LLM4AD
cd ../LLM4AD && pip install -e .
```

### 3. Run smoke test (verify everything is wired up)

```bash
python smoke_test.py
```

Fix any ❌ errors before proceeding.

### 4. Run the benchmark

```bash
python run_benchmark.py
```

Or with overrides:

```bash
# Only LLaMEA, only sphere, only 2 seeds
python run_benchmark.py --frameworks llamea --tasks sphere_5d --seeds 0 1

# Dry run (validate only, no experiments)
python run_benchmark.py --dry-run

# Verbose output
python run_benchmark.py --verbose
```

### 5. Analyse results

```bash
python analyze_benchmark.py
python analyze_benchmark.py --plot              # saves PNG charts
python analyze_benchmark.py --export summary.csv
```

---

## benchmark_config.json reference

| Key | Description |
|---|---|
| `output_dir` | Where to write `.jsonl` result files |
| `append_results` | If `false`, each run overwrites previous results for the same (framework, task) |
| `seeds` | List of random seeds |
| `tasks` | Task names: `sphere_5d`, `rastrigin_5d`, `rosenbrock_5d` |
| `ollama.model` | Ollama model name (e.g. `llama3.1:latest`) |
| `ollama.base_url` | Ollama server URL (e.g. `http://10.5.32.17:11434`) |
| `task_defaults.budget` | Max function evaluations per algorithm evaluation |
| `task_defaults.lower_bound` / `upper_bound` | Search space bounds |
| `task_defaults.eval_seeds` | Seeds used when evaluating a generated algorithm |
| `frameworks.llamea.enabled` | Toggle LLaMEA on/off |
| `frameworks.llamea.repo_path` | Path to cloned LLaMEA repo |
| `frameworks.llamea.search_budget` | Number of LLM queries LLaMEA is allowed |
| `frameworks.llm4ad.enabled` | Toggle LLM4AD on/off |
| `frameworks.llm4ad.repo_path` | Path to cloned LLM4AD repo |
| `frameworks.ease.enabled` | Toggle frontEASE on/off |
| `frameworks.ease.api_base_url` | frontEASE server URL |
| `frameworks.ease.template_task_id` | ID of EASE task to clone per run |

---

## Results format

Each `.jsonl` file in `benchmark_results/` contains one JSON record per run:

```json
{
  "framework": "llamea",
  "task": "sphere_5d",
  "seed": 0,
  "best_value": 0.00123,
  "success": true,
  "error": null,
  "elapsed_sec": 142.5,
  "timestamp": "2025-04-01T10:00:00Z",
  "extra": { ... }
}
```

`best_value` is the **raw objective value** (lower = better for all tasks).

---

## How each adapter works

### LLaMEA (`llamea_adapter.py`)
Calls `LLaMEA(...).run(evaluate_fn)` with the Ollama server exposed as an
OpenAI-compatible endpoint (`<base_url>/v1`).  The evaluation function execs
generated algorithm code and runs it on the task over `eval_seeds`, returning
the mean objective value.

### LLM4AD (`llm4ad_adapter.py`)
Wraps the task as a `_ContinuousEvaluation` object (compatible with LLM4AD's
`Evaluation` interface) and calls `EoH(...).run()`.  Connects to Ollama via
`HttpsApi` pointing at `<base_url>`.

### frontEASE (`ease_adapter.py`)
Communicates via REST API: authenticates, clones a template task, uploads the
problem definition, starts the run, polls until completion, and retrieves the
best result.  Set `ease.template_task_id` in config before using.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `llamea` import error | `pip install llamea` or check `llamea.repo_path` |
| `llm4ad` import error | `cd ../LLM4AD && pip install -e .` |
| Ollama connection refused | Check `ollama.base_url`; ensure `ollama serve` is running |
| Model not found | `ollama pull llama3.1:latest` |
| EASE login fails | Check `username`, `password`, and `api_base_url` in config |
| All runs fail with timeout | Reduce `search_budget` / `max_sample_nums` |

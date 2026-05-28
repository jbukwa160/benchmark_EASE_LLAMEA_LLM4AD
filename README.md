# LLM Evolutionary Benchmarks

A reproducible benchmark harness for comparing **LLM-driven evolutionary optimizers** on continuous black-box tasks and optional self-evolving task suites.

| Framework | Role |
|-----------|------|
| [LLaMEA](https://github.com/lioncoder3010/LLaMEA) | LLM evolves optimization algorithms; evaluated on benchmark tasks |
| [LLM4AD](https://github.com/Optima-CityU/llm4ad) | Evolution of Heuristics (EoH) and related methods via Ollama |
| [frontEASE](https://github.com/) | Optional REST-backed optimizer (Docker stack on ports 4000 / 8086) |

**Benchmark tasks (default paper suite):** Sphere, Rastrigin, and Rosenbrock in 5D on \([-5, 5]^5\), with a fixed evaluation budget per trial.

---

## Repository layout

```
LLM_Evolutionary_benchmarks/
├── auto_algo_benchmark/     # Main harness (run here)
├── benchmark_redesign/      # Alternate runner with hard timeouts (experimental)
├── benchmark_results/       # Local outputs only (gitignored)
├── docs/                    # Setup, architecture, frameworks
├── LLaMEA/                  # Clone separately (gitignored)
├── LLM4AD/                  # Clone separately (gitignored)
└── frontEASE/               # Clone separately (gitignored)
```

---

## Quick start

### 1. Prerequisites

- **Python 3.10+**
- **Git**
- **Ollama** ([install](https://ollama.com)) with a model pulled, e.g. `ollama pull llama3.1:latest`
- Optional: **frontEASE** Docker stack if you enable `ease` in config

### 2. Clone this repo and framework dependencies

```bash
git clone <your-repo-url> LLM_Evolutionary_benchmarks
cd LLM_Evolutionary_benchmarks

# See docs/FRAMEWORKS.md for URLs and layout
git clone <LLaMEA-repo-url> LLaMEA
git clone <LLM4AD-repo-url> LLM4AD
# optional:
# git clone <frontEASE-repo-url> frontEASE
```

### 3. Python environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
pip install -r auto_algo_benchmark/requirements.txt
cd LLaMEA && pip install -e . && cd ..
cd LLM4AD && pip install -e . && cd ..
```

### 4. Configure (no secrets in git)

```bash
cd auto_algo_benchmark
copy ..\.env.example ..\.env          # Windows; or cp on Unix
# Edit benchmark_config.local.json or .env for your Ollama URL and EASE credentials
```

Set `BENCHMARK_OLLAMA_BASE_URL` if Ollama is not on `http://127.0.0.1:11434`. For EASE, set `EASE_PASSWORD` and `EASE_REST_ACCESS_TOKEN` (see `.env.example`).

### 5. Smoke test and run

```bash
cd auto_algo_benchmark
python smoke_test.py
python smoke_test.py --config configs/benchmark_paper.json

python run_benchmark.py --dry-run -v
python run_benchmark.py --config configs/benchmark_paper.json -v
```

Results are written under `benchmark_results/` (not committed). Analyze with:

```bash
python analyze_benchmark.py --results-dir ../benchmark_results --dedupe-last --valid-only --export-paper paper_summary.csv
```

**Windows (paper suite + status page):** from `auto_algo_benchmark`, run `.\run_paper_benchmark.ps1`.

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Full install, configs, running, analyzing |
| [docs/FRAMEWORKS.md](docs/FRAMEWORKS.md) | Cloning LLaMEA, LLM4AD, frontEASE |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Harness design and adapters |
| [auto_algo_benchmark/README.md](auto_algo_benchmark/README.md) | Config reference, troubleshooting |
| [benchmark_redesign/README.md](benchmark_redesign/README.md) | Alternate timeout-safe runner |

---

## What is not in this repository

The following are intentionally **excluded** via `.gitignore`:

- `benchmark_results/` — run outputs, JSONL, `per_run/*.json`
- `auto_algo_benchmark/exp-*/` — LLaMEA experiment folders
- `LLaMEA/`, `LLM4AD/`, `frontEASE/` — clone as siblings
- Local secrets (`benchmark_config.local.json`, `.env`)
- Generated figures and CSV exports from analysis

---

## License

See [LICENSE](LICENSE). Third-party frameworks (LLaMEA, LLM4AD, frontEASE) have their own licenses when you clone them.

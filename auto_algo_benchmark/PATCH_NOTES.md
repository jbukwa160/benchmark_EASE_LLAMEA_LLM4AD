# Fixed bundle notes

This bundle is meant to be extracted as-is so the folders stay side by side:
- auto_algo_benchmark/
- LLaMEA-main/
- LLM4AD-main/
- frontEASE-main/

Main fixes:
- `append_results: false` now cleans old JSONL results, heartbeat, and per-run logs.
- Worker subprocesses run unbuffered so logs update during long runs.
- LLM4AD retry attempts keep separate profiler folders instead of deleting prior evidence.
- LLM4AD best-score extraction now reads all `samples*.json` profiler files and `log.txt`.
- `smoke_test.py` also checks whether the Ollama endpoint is reachable.

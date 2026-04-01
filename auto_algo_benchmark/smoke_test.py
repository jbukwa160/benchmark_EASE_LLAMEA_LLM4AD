"""
smoke_test.py — Quick sanity checks before running a full benchmark.

Checks:
  1. Config file is valid JSON and has all required keys.
  2. Task definitions evaluate correctly.
  3. Ollama endpoint is reachable.
  4. Framework repos are importable (if enabled).
  5. (For EASE) API endpoint is reachable and login works.

Usage:
    python smoke_test.py
    python smoke_test.py --config my_config.json
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def check(label: str, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
        print(f"  {PASS}  {label}")
        return True
    except Exception as exc:
        print(f"  {FAIL}  {label}")
        print(f"       {exc}")
        return False


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print("─" * 60)


# ---------------------------------------------------------------------------

def test_config(config_path: str) -> dict | None:
    section("1. Config validation")
    p = Path(config_path)

    def _load():
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        with open(p) as f:
            return json.load(f)

    cfg = None
    try:
        cfg = _load()
        print(f"  {PASS}  Config loaded: {p}")
    except Exception as exc:
        print(f"  {FAIL}  Config load: {exc}")
        return None

    required_keys = ["ollama", "tasks", "seeds", "frameworks"]
    for key in required_keys:
        if key in cfg:
            print(f"  {PASS}  Key '{key}' present")
        else:
            print(f"  {FAIL}  Key '{key}' MISSING from config")

    ollama = cfg.get("ollama", {})
    for sub in ["model", "base_url"]:
        if ollama.get(sub):
            print(f"  {PASS}  ollama.{sub} = {ollama[sub]!r}")
        else:
            print(f"  {FAIL}  ollama.{sub} not set")

    return cfg


def test_tasks(cfg: dict):
    section("2. Task definitions")
    from benchmark_harness.tasks import get_task, TASK_REGISTRY
    import numpy as np

    task_defaults = cfg.get("task_defaults", {})
    lb = task_defaults.get("lower_bound", -5.0)
    ub = task_defaults.get("upper_bound", 5.0)

    for task_name in cfg.get("tasks", []):
        try:
            task = get_task(task_name, lb, ub)
            x = np.zeros(task.dimension)
            val = task.evaluate(x)
            print(f"  {PASS}  {task_name}: evaluate(zeros) = {val:.4f}")
        except Exception as exc:
            print(f"  {FAIL}  {task_name}: {exc}")


def test_ollama(cfg: dict):
    section("3. Ollama connectivity")
    import requests

    base_url = cfg.get("ollama", {}).get("base_url", "")
    model = cfg.get("ollama", {}).get("model", "")

    # Check base endpoint
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=10)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        print(f"  {PASS}  Ollama reachable at {base_url}")
        if model in models:
            print(f"  {PASS}  Model '{model}' is available")
        else:
            avail = ", ".join(models[:5]) + ("…" if len(models) > 5 else "")
            print(f"  {WARN}  Model '{model}' not in available list: [{avail}]")
            print(f"       Run: ollama pull {model}")
    except Exception as exc:
        print(f"  {FAIL}  Ollama not reachable: {exc}")
        return

    # Quick generation test via OpenAI-compatible endpoint
    try:
        import requests as _req
        v1_url = base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
        }
        r = _req.post(v1_url, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        print(f"  {PASS}  Chat completion OK (response: {content!r})")
    except Exception as exc:
        print(f"  {WARN}  Chat completion test failed: {exc}")


def test_llamea_import(cfg: dict):
    section("4a. LLaMEA import")
    fw_cfg = cfg.get("frameworks", {}).get("llamea", {})
    if not fw_cfg.get("enabled", False):
        print(f"  {WARN}  LLaMEA is disabled in config — skipping")
        return

    repo = fw_cfg.get("repo_path", "../LLaMEA")
    p = Path(repo).resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

    try:
        import llamea  # noqa: F401
        print(f"  {PASS}  'llamea' package importable (repo: {p})")
    except ImportError as exc:
        print(f"  {FAIL}  Cannot import 'llamea': {exc}")
        print(f"       Make sure repo_path ({repo}) is correct and the package is installed.")

    try:
        from openai import OpenAI  # noqa: F401
        print(f"  {PASS}  'openai' package importable")
    except ImportError:
        print(f"  {FAIL}  'openai' package not found — run: pip install openai")


def test_llm4ad_import(cfg: dict):
    section("4b. LLM4AD import")
    fw_cfg = cfg.get("frameworks", {}).get("llm4ad", {})
    if not fw_cfg.get("enabled", False):
        print(f"  {WARN}  LLM4AD is disabled in config — skipping")
        return

    repo = fw_cfg.get("repo_path", "../LLM4AD")
    p = Path(repo).resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

    try:
        from llm4ad.tools.llm.llm_api_https import HttpsApi  # noqa: F401
        print(f"  {PASS}  'llm4ad.tools.llm.llm_api_https' importable (repo: {p})")
    except ImportError as exc:
        print(f"  {FAIL}  Cannot import LLM4AD: {exc}")
        print(f"       Make sure repo_path ({repo}) is correct and dependencies installed.")

    try:
        from llm4ad.method.eoh import EoH, EoHProfiler  # noqa: F401
        print(f"  {PASS}  'llm4ad.method.eoh.EoH' importable")
    except ImportError as exc:
        print(f"  {FAIL}  Cannot import EoH: {exc}")


def test_ease_api(cfg: dict):
    section("4c. frontEASE API")
    fw_cfg = cfg.get("frameworks", {}).get("ease", {})
    if not fw_cfg.get("enabled", False):
        print(f"  {WARN}  EASE is disabled in config — skipping")
        return

    import requests

    base_url = fw_cfg.get("api_base_url", "")
    try:
        r = requests.get(base_url.rstrip("/") + "/api/health", timeout=10)
        r.raise_for_status()
        print(f"  {PASS}  EASE health endpoint OK at {base_url}")
    except Exception as exc:
        print(f"  {WARN}  EASE health check failed: {exc}")

    try:
        r = requests.post(
            base_url.rstrip("/") + "/api/auth/login",
            json={"username": fw_cfg.get("username", ""),
                  "password": fw_cfg.get("password", "")},
            timeout=15,
        )
        r.raise_for_status()
        print(f"  {PASS}  EASE login OK")
    except Exception as exc:
        print(f"  {FAIL}  EASE login failed: {exc}")

    template_id = fw_cfg.get("template_task_id", "")
    if "PUT-YOUR" in template_id or not template_id:
        print(f"  {WARN}  ease.template_task_id is not configured")
    else:
        print(f"  {PASS}  ease.template_task_id = {template_id!r}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smoke-test the benchmark setup.")
    parser.add_argument("--config", default="benchmark_config.json")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LLM EVOLUTIONARY BENCHMARK — SMOKE TEST")
    print("=" * 60)

    cfg = test_config(args.config)
    if cfg is None:
        print("\nAborted: config is invalid.\n")
        sys.exit(1)

    test_tasks(cfg)
    test_ollama(cfg)
    test_llamea_import(cfg)
    test_llm4ad_import(cfg)
    test_ease_api(cfg)

    print("\n" + "=" * 60)
    print("  Smoke test complete. Fix any ❌ issues above before running.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

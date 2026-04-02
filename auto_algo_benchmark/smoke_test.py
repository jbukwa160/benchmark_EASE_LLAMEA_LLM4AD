"""Quick sanity checks before running a full benchmark."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_harness.adapters.llamea_adapter import _candidate_repo_dirs as llamea_candidates
from benchmark_harness.adapters.llm4ad_adapter import _candidate_repo_dirs as llm4ad_candidates

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print("─" * 60)


def test_config(config_path: str) -> tuple[dict | None, Path]:
    section("1. Config validation")
    p = Path(config_path).resolve()
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        print(f"  {PASS}  Config loaded: {p}")
        return cfg, p
    except Exception as exc:
        print(f"  {FAIL}  Config load: {exc}")
        return None, p


def test_llamea_import(cfg: dict, config_path: Path):
    section("2. LLaMEA import")
    fw_cfg = cfg.get("frameworks", {}).get("llamea", {})
    if not fw_cfg.get("enabled", False):
        print(f"  {WARN}  LLaMEA is disabled in config — skipping")
        return
    candidates = llamea_candidates(config_path.parent, fw_cfg.get("repo_path"), "llamea", ["LLaMEA", "llamea"])
    if not candidates:
        print(f"  {FAIL}  Could not resolve repo path for LLaMEA from {fw_cfg.get('repo_path')!r}")
        return
    repo = candidates[0]
    sys.path.insert(0, str(repo))
    try:
        import llamea  # noqa: F401
        print(f"  {PASS}  LLaMEA importable from {repo}")
    except Exception as exc:
        print(f"  {FAIL}  Cannot import llamea from {repo}: {exc}")


def test_llm4ad_import(cfg: dict, config_path: Path):
    section("3. LLM4AD import")
    fw_cfg = cfg.get("frameworks", {}).get("llm4ad", {})
    if not fw_cfg.get("enabled", False):
        print(f"  {WARN}  LLM4AD is disabled in config — skipping")
        return
    candidates = llm4ad_candidates(config_path.parent, fw_cfg.get("repo_path"), "llm4ad", ["LLM4AD", "llm4ad"])
    if not candidates:
        print(f"  {FAIL}  Could not resolve repo path for LLM4AD from {fw_cfg.get('repo_path')!r}")
        return
    repo = candidates[0]
    sys.path.insert(0, str(repo))
    try:
        from benchmark_harness.adapters.llm4ad_adapter import LLM4ADAdapter
        adapter = LLM4ADAdapter(fw_cfg, {"ollama": {"model": "test", "base_url": "http://127.0.0.1:11434"}, "_config_dir": str(config_path.parent), "task_defaults": {}})
        adapter._load_llm4ad_symbols()
        print(f"  {PASS}  LLM4AD EoH symbols importable from {repo}")
    except Exception as exc:
        print(f"  {FAIL}  Cannot load LLM4AD symbols from {repo}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the benchmark setup.")
    parser.add_argument("--config", default="benchmark_config.json")
    args = parser.parse_args()

    cfg, config_path = test_config(args.config)
    if cfg is None:
        sys.exit(1)
    test_llamea_import(cfg, config_path)
    test_llm4ad_import(cfg, config_path)


if __name__ == "__main__":
    main()

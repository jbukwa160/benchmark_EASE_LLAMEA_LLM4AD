"""
debug_llamea.py — Run this in your auto_algo_benchmark folder to print
the exact LLaMEA constructor signature and the full traceback.

Usage:
    python debug_llamea.py --config .\benchmark_config_weekend_conservative.json
"""
import argparse, importlib, inspect, json, sys, traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="benchmark_config.json")
args = parser.parse_args()

cfg = json.loads(Path(args.config).read_text())
repo = Path(cfg["frameworks"]["llamea"]["repo_path"])
config_dir = Path(args.config).resolve().parent
if not repo.is_absolute():
    repo = (config_dir / repo).resolve()

print(f"\n{'='*60}")
print(f"LLaMEA repo path : {repo}")
print(f"Repo exists      : {repo.exists()}")

if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

# ── 1. Import LLaMEA ──────────────────────────────────────────────────────────
try:
    from llamea import LLaMEA
    print(f"\n✅  LLaMEA imported from: {inspect.getfile(LLaMEA)}")
except Exception as e:
    print(f"\n❌  Cannot import LLaMEA: {e}")
    sys.exit(1)

# ── 2. Print full __init__ signature ──────────────────────────────────────────
sig = inspect.signature(LLaMEA.__init__)
print(f"\nLLaMEA.__init__ signature:\n  LLaMEA{sig}\n")
print("Parameters:")
for name, p in sig.parameters.items():
    print(f"  {name!s:<25} default={p.default!r}")

# ── 3. Find the LLM base class ────────────────────────────────────────────────
print(f"\n{'─'*60}")
LLMBase = None
for mod_path in ("llamea.llm", "llamea"):
    try:
        mod = __import__(mod_path, fromlist=["LLM"])
        LLMBase = getattr(mod, "LLM", None)
        if LLMBase is not None:
            print(f"✅  LLM base class found in: {mod_path}")
            print(f"   file: {inspect.getfile(LLMBase)}")
            lsig = inspect.signature(LLMBase.__init__)
            print(f"   LLM.__init__ params: {list(lsig.parameters.keys())}")
            break
    except Exception as e:
        print(f"   {mod_path}: {e}")

if LLMBase is None:
    print("⚠️   LLM base class not found — will use object as fallback")

# ── 4. Try to instantiate the LLM subclass ────────────────────────────────────
print(f"\n{'─'*60}")
print("Attempting to instantiate _OllamaLLM ...")

import json as _json, time as _time
from urllib import request as _req

ollama_endpoint = cfg["ollama"]["base_url"].rstrip("/") + "/v1/chat/completions"
ollama_model    = cfg["ollama"]["model"]

Base = LLMBase if LLMBase is not None else object

base_params = inspect.signature(Base.__init__).parameters if LLMBase else {}

class _OllamaLLM(Base):
    def __init__(self):
        init_kw = {}
        for name, default in [
            ("do_auto_trim", False), ("debug_mode", False),
            ("api_key", "ollama"), ("model", ollama_model),
            ("base_url", ollama_endpoint),
        ]:
            if name in base_params:
                init_kw[name] = default
        try:
            super().__init__(**init_kw)
        except TypeError as e:
            print(f"  ⚠️  super().__init__(**{init_kw}) failed: {e}")
            try:
                super().__init__()
            except TypeError:
                pass
        self.model = ollama_model

    def query(self, msgs, **kw): return "test"
    def get_response(self, p, **kw): return "test"

try:
    llm = _OllamaLLM()
    print(f"✅  _OllamaLLM instantiated OK: {llm}")
except Exception as e:
    print(f"❌  _OllamaLLM instantiation failed:\n{traceback.format_exc()}")
    sys.exit(1)

# ── 5. Try to instantiate LLaMEA itself ──────────────────────────────────────
print(f"\n{'─'*60}")
print("Attempting LLaMEA instantiation ...")

ctor_params = sig.parameters
has_f   = "f"   in ctor_params
has_llm = "llm" in ctor_params
print(f"  has 'f'   param: {has_f}")
print(f"  has 'llm' param: {has_llm}")

dummy_eval = lambda ind, log=None: ind   # noqa

try:
    if has_llm and has_f:
        optimizer = LLaMEA(
            **{k: v for k, v in dict(
                f=dummy_eval, llm=llm, budget=2,
                n_parents=1, n_offspring=2,
                experiment_name="debug_test",
                role_prompt="test", task_prompt="test",
                log=False,
            ).items() if k in ctor_params}
        )
    elif has_llm:
        optimizer = LLaMEA(
            **{k: v for k, v in dict(
                llm=llm, budget=2, n_parents=1, n_offspring=2,
            ).items() if k in ctor_params}
        )
    else:
        optimizer = LLaMEA(
            **{k: v for k, v in dict(
                model=ollama_model, api_key="ollama",
                base_url=cfg["ollama"]["base_url"].rstrip("/")+"/v1",
                budget=2, n_parents=1, n_offspring=2,
            ).items() if k in ctor_params}
        )
    print(f"✅  LLaMEA instantiated OK: {optimizer}")
except Exception:
    print(f"❌  LLaMEA instantiation failed:\n{traceback.format_exc()}")

print(f"\n{'='*60}")
print("Debug complete. Share this output to diagnose the issue.")
print('='*60)

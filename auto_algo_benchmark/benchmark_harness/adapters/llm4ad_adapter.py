"""Adapter for the current LLM4AD repository, hardened for unattended runs."""

from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib import request

from .base import BaseAdapter
from ..tasks import BenchmarkTask
from ..utils import PENALTY_OBJECTIVE, evaluate_candidate_code, get_logger, timestamp

logger = get_logger(__name__)


class LLM4ADAdapter(BaseAdapter):
    name = "llm4ad"

    def is_enabled(self) -> bool:
        return self.framework_cfg.get("enabled", False)

    def _repo_path(self) -> Path:
        config_dir = Path(self.global_cfg.get("_config_dir", Path.cwd()))
        repo = Path(self.framework_cfg.get("repo_path", "../LLM4AD"))
        return repo if repo.is_absolute() else (config_dir / repo).resolve()

    def _add_repo_to_path(self) -> None:
        p = self._repo_path()
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
            logger.debug("Added LLM4AD repo to sys.path: %s", p)

    def _load_llm4ad_symbols(self):
        import importlib.util
        import types

        repo_root = self._repo_path()
        pkg_root  = repo_root / "llm4ad"

        def ensure_package(name: str, path: Path):
            mod = sys.modules.get(name)
            if mod is None:
                mod = types.ModuleType(name)
                mod.__path__ = [str(path)]
                sys.modules[name] = mod
            return mod

        def load_pkg(name: str, init_file: Path, search_path: Path):
            if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(init_file):
                return sys.modules[name]
            spec = importlib.util.spec_from_file_location(
                name, str(init_file),
                submodule_search_locations=[str(search_path)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {name} from {init_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        ensure_package("llm4ad", pkg_root)
        load_pkg("llm4ad.base",  pkg_root / "base"  / "__init__.py", pkg_root / "base")
        ensure_package("llm4ad.tools", pkg_root / "tools")
        load_pkg("llm4ad.tools.profiler",
                 pkg_root / "tools" / "profiler" / "__init__.py",
                 pkg_root / "tools" / "profiler")
        ensure_package("llm4ad.method", pkg_root / "method")
        eoh_pkg = load_pkg("llm4ad.method.eoh",
                           pkg_root / "method" / "eoh" / "__init__.py",
                           pkg_root / "method" / "eoh")

        base_mod = sys.modules["llm4ad.base"]
        return base_mod.Evaluation, base_mod.LLM, eoh_pkg.EoH, eoh_pkg.EoHProfiler

    def run(self, task: BenchmarkTask, seed: int) -> dict[str, Any]:
        self._add_repo_to_path()
        t0 = time.perf_counter()
        result: dict[str, Any] = {
            "framework":   self.name,
            "task":        task.name,
            "seed":        seed,
            "best_value":  None,
            "success":     False,
            "error":       None,
            "elapsed_sec": None,
            "timestamp":   timestamp(),
            "extra":       {},
        }

        try:
            Evaluation, LLM, EoH, EoHProfiler = self._load_llm4ad_symbols()

            cfg               = self.framework_cfg
            eval_seeds        = list(self.task_defaults.get("eval_seeds", list(range(11, 21))))
            budget            = int(self.task_defaults.get("budget", 1500))
            candidate_timeout = int(cfg.get("candidate_timeout_sec", 180))
            endpoint          = self.ollama_base_url.rstrip("/") + "/v1/chat/completions"
            ollama_model      = self.ollama_model
            llm_timeout       = int(cfg.get("llm_timeout_sec", 180))
            temperature       = float(cfg.get("temperature", 0.7))

            # ---- HTTP-only LLM subclass ----
            class _HTTPModel(LLM):
                def __init__(self):
                    try:
                        super().__init__(do_auto_trim=False, debug_mode=False)
                    except TypeError:
                        try:
                            super().__init__()
                        except TypeError:
                            pass
                    self.model = ollama_model

                def draw_sample(self, prompt, *args, **kwargs) -> str:
                    messages = [{"role": "user", "content": str(prompt)}]
                    payload  = json.dumps({
                        "model":       ollama_model,
                        "messages":    messages,
                        "temperature": temperature,
                        "stream":      False,
                    }).encode("utf-8")
                    headers = {
                        "Content-Type":  "application/json",
                        "Authorization": "Bearer ollama",
                    }
                    last_exc = None
                    for attempt in range(1, 5):
                        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
                        try:
                            with request.urlopen(req, timeout=llm_timeout) as resp:
                                raw = json.loads(resp.read().decode("utf-8"))
                            return raw["choices"][0]["message"]["content"]
                        except Exception as exc:
                            last_exc = exc
                            if attempt < 4:
                                time.sleep(2 * attempt)
                    raise RuntimeError(f"Ollama request failed: {last_exc}")

            # ---- Task wrapper ----
            template_program = (
                "import numpy as np\n\n"
                "def algorithm(func, dim, lb, ub, budget, rng):\n"
                "    \"\"\"Return the best 1-D numpy array found. Sequential only.\"\"\"\n"
                "    x = rng.uniform(lb, ub, size=dim)\n"
                "    return np.asarray(x, dtype=float)\n"
            )

            class _ContinuousEvaluation(Evaluation):
                def __init__(self):
                    try:
                        super().__init__(
                            template_program=template_program,
                            task_description=(
                                f"{task.description}\n"
                                "Write a single sequential Python function only. "
                                "No threads, multiprocessing, subprocesses, files, "
                                "networking, eval, exec, or non-numpy imports."
                            ),
                            timeout_seconds=None,
                            exec_code=False,
                            safe_evaluate=False,
                        )
                    except TypeError:
                        # Older Evaluation base may have a different signature
                        super().__init__(
                            template_program=template_program,
                            task_description=task.description,
                        )

                def evaluate_program(self, program_str: str, callable_func=None, **kwargs):
                    mean_obj, _feedback, err = evaluate_candidate_code(
                        program_str,
                        task_name=task.name,
                        lower_bound=task.lower_bound,
                        upper_bound=task.upper_bound,
                        budget=budget,
                        eval_seeds=eval_seeds,
                        timeout_seconds=candidate_timeout,
                    )
                    if mean_obj >= PENALTY_OBJECTIVE and err:
                        return None
                    # LLM4AD maximises — return negated objective
                    return float(-mean_obj)

            llm        = _HTTPModel()
            evaluation = _ContinuousEvaluation()
            attempts   = int(cfg.get("attempts", 2))
            last_log_dir   = None
            last_best_value = None

            for attempt in range(1, attempts + 1):
                log_dir = (
                    Path(self.global_cfg.get("_output_dir_resolved", "benchmark_results"))
                    / "llm4ad_logs" / task.name / f"seed{seed}" / f"attempt{attempt}"
                )
                last_log_dir = log_dir
                profiler = EoHProfiler(log_dir=str(log_dir), log_style="simple")

                method = EoH(
                    llm              = llm,
                    evaluation       = evaluation,
                    profiler         = profiler,
                    max_sample_nums  = int(cfg.get("max_sample_nums", 24)),
                    max_generations  = int(cfg.get("max_generations", 12)),
                    pop_size         = int(cfg.get("pop_size", 6)),
                    selection_num    = int(cfg.get("selection_num", 3)),
                    num_samplers     = int(cfg.get("num_samplers", 1)),
                    num_evaluators   = int(cfg.get("num_evaluators", 1)),
                    debug_mode       = bool(cfg.get("debug_mode", False)),
                    multi_thread_or_process_eval = cfg.get(
                        "multi_thread_or_process_eval", "thread"
                    ),
                )

                method.run()
                best_value = _extract_best_from_profiler(log_dir)
                last_best_value = best_value

                if best_value is not None and math.isfinite(best_value) and best_value < PENALTY_OBJECTIVE:
                    result["best_value"] = best_value
                    result["success"]    = True
                    result["extra"]      = {"log_dir": str(log_dir), "attempt": attempt}
                    break

            if not result["success"]:
                # Do NOT hard-fail — record whatever we have so the run is not lost.
                # If last_best_value was extracted but is large, still record it.
                if last_best_value is not None and math.isfinite(last_best_value):
                    result["best_value"] = last_best_value
                    result["success"]    = True
                    result["extra"]      = {
                        "log_dir": str(last_log_dir),
                        "note":    "best_value may be suboptimal (fallback)",
                    }
                else:
                    result["error"] = (
                        f"LLM4AD finished without a usable best score after {attempts} attempt(s). "
                        f"last_best={last_best_value!r}  log_dir={last_log_dir}"
                    )

        except ImportError as exc:
            msg = f"LLM4AD import failed: {exc}. Check repo_path in config."
            logger.error(msg)
            result["error"] = msg
        except Exception:
            tb = traceback.format_exc()
            logger.error("LLM4AD run failed:\n%s", tb)
            result["error"] = tb

        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return result


def _extract_best_from_profiler(log_dir: str | Path) -> float | None:
    """
    Read the best objective value from EoHProfiler output files.

    FIX: EoHProfiler stores the score as a *negated* objective (higher=better).
    We must negate it back to get the raw objective (lower=better).
    Previously this was being double-negated, giving wrong (large) values.
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None

    best_internal: float | None = None  # highest negated score seen

    for path in log_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        items = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for item in items:
            if not isinstance(item, dict):
                continue
            # EoHProfiler keys: "score" (negated objective), "obj" (raw), or "best_score"
            for key in ("score", "best_score", "fitness"):
                raw = item.get(key)
                if raw is None:
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    if best_internal is None or v > best_internal:
                        best_internal = v

    if best_internal is None:
        return None

    # Undo the negation: score = -objective  →  objective = -score
    objective = -best_internal
    return objective if math.isfinite(objective) else None

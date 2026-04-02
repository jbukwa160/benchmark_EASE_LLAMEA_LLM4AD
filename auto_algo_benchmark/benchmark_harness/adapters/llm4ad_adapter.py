"""Adapter for the current LLM4AD repository, hardened for unattended runs."""

from __future__ import annotations

import importlib.util
import json
import math
import random
import re
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any
from urllib import request

from .base import BaseAdapter
from ..tasks import BenchmarkTask
from ..utils import PENALTY_OBJECTIVE, evaluate_candidate_code, get_logger, timestamp

logger = get_logger(__name__)


def _candidate_repo_dirs(config_dir: Path, configured: str | None, package_dir: str, framework_names: list[str]) -> list[Path]:
    raw_candidates: list[Path] = []
    if configured:
        p = Path(configured)
        raw_candidates.append(p if p.is_absolute() else (config_dir / p))

    parent = config_dir.parent
    for name in framework_names:
        raw_candidates.extend([
            parent / name,
            parent / f"{name}-main",
            config_dir / name,
            config_dir / f"{name}-main",
        ])

    try:
        for p in parent.iterdir():
            if p.is_dir() and p.name.lower().startswith(tuple(n.lower() for n in framework_names)):
                raw_candidates.append(p)
    except Exception:
        pass

    seen: set[str] = set()
    resolved: list[Path] = []
    for p in raw_candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(rp)

    valid: list[Path] = []
    for p in resolved:
        if (p / package_dir / "__init__.py").exists():
            valid.append(p)
        elif p.name == package_dir and (p / "__init__.py").exists():
            valid.append(p.parent)
    return valid


def _normalize_llm4ad_response(text: str) -> str:
    content = (text or "").strip()
    thought_match = re.search(r"\{.*?\}", content, re.DOTALL)
    thought = thought_match.group(0) if thought_match else "{random-search}"

    code_match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    code = code_match.group(1).strip() if code_match else content
    code = code.strip()

    if "def " not in code and "algorithm(" not in code:
        body = code
    else:
        body = code

    return f"{thought}\n{body}"


class LLM4ADAdapter(BaseAdapter):
    name = "llm4ad"

    def is_enabled(self) -> bool:
        return self.framework_cfg.get("enabled", False)

    def _repo_path(self) -> Path:
        config_dir = Path(self.global_cfg.get("_config_dir", Path.cwd()))
        configured = self.framework_cfg.get("repo_path", "../LLM4AD")
        candidates = _candidate_repo_dirs(config_dir, configured, "llm4ad", ["LLM4AD", "llm4ad"])
        if candidates:
            return candidates[0]
        p = Path(configured)
        return p if p.is_absolute() else (config_dir / p).resolve()

    def _add_repo_to_path(self) -> None:
        p = self._repo_path()
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
            logger.debug("Added LLM4AD repo to sys.path: %s", p)

    def _load_llm4ad_symbols(self):
        repo_root = self._repo_path()
        pkg_root = repo_root / "llm4ad"
        if not (pkg_root / "__init__.py").exists():
            raise ImportError(f"Could not find llm4ad package under: {repo_root}")

        def ensure_package(name: str, path: Path):
            mod = sys.modules.get(name)
            if mod is None:
                mod = types.ModuleType(name)
                mod.__path__ = [str(path)]
                sys.modules[name] = mod
            return mod

        def load_pkg(name: str, init_file: Path, search_path: Path):
            if not init_file.exists():
                raise ImportError(f"Missing module file: {init_file}")
            if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(init_file):
                return sys.modules[name]
            spec = importlib.util.spec_from_file_location(name, str(init_file), submodule_search_locations=[str(search_path)])
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {name} from {init_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        ensure_package("llm4ad", pkg_root)
        base_mod = load_pkg("llm4ad.base", pkg_root / "base" / "__init__.py", pkg_root / "base")
        ensure_package("llm4ad.tools", pkg_root / "tools")
        load_pkg("llm4ad.tools.profiler", pkg_root / "tools" / "profiler" / "__init__.py", pkg_root / "tools" / "profiler")
        ensure_package("llm4ad.method", pkg_root / "method")
        eoh_pkg = load_pkg("llm4ad.method.eoh", pkg_root / "method" / "eoh" / "__init__.py", pkg_root / "method" / "eoh")
        return base_mod.Evaluation, base_mod.LLM, eoh_pkg.EoH, eoh_pkg.EoHProfiler

    def run(self, task: BenchmarkTask, seed: int) -> dict[str, Any]:
        self._add_repo_to_path()
        t0 = time.perf_counter()
        result: dict[str, Any] = {
            "framework": self.name,
            "task": task.name,
            "seed": seed,
            "best_value": None,
            "success": False,
            "error": None,
            "elapsed_sec": None,
            "timestamp": timestamp(),
            "extra": {},
        }

        try:
            Evaluation, LLM, EoH, EoHProfiler = self._load_llm4ad_symbols()

            cfg = self.framework_cfg
            eval_seeds = list(self.task_defaults.get("eval_seeds", list(range(11, 21))))
            budget = int(self.task_defaults.get("budget", 1500))
            candidate_timeout = int(cfg.get("candidate_timeout_sec", 180))
            endpoint = self.ollama_base_url.rstrip("/") + "/v1/chat/completions"
            ollama_model = self.ollama_model
            llm_timeout = int(cfg.get("llm_timeout_sec", 180))
            temperature = float(cfg.get("temperature", 0.7))

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
                    payload = json.dumps({
                        "model": ollama_model,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False,
                    }).encode("utf-8")
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer ollama",
                    }
                    last_exc = None
                    for attempt in range(1, 5):
                        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
                        try:
                            with request.urlopen(req, timeout=llm_timeout) as resp:
                                raw = json.loads(resp.read().decode("utf-8"))
                            return _normalize_llm4ad_response(raw["choices"][0]["message"]["content"])
                        except Exception as exc:
                            last_exc = exc
                            if attempt < 4:
                                time.sleep(2 * attempt)
                    raise RuntimeError(f"Ollama request failed: {last_exc}")

            template_program = (
                "import numpy as np\n\n"
                "def algorithm(func, dim, lb, ub, budget, rng):\n"
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
                                "No threads, multiprocessing, subprocesses, files, networking, eval, exec, or non-numpy imports."
                            ),
                            timeout_seconds=None,
                            exec_code=False,
                            safe_evaluate=False,
                        )
                    except TypeError:
                        super().__init__(template_program=template_program, task_description=task.description)

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
                    return float(-mean_obj)

            llm = _HTTPModel()
            evaluation = _ContinuousEvaluation()
            attempts = int(cfg.get("attempts", 2))
            last_log_dir = None
            last_best_value = None

            for attempt in range(1, attempts + 1):
                log_dir = Path(self.global_cfg.get("_output_dir_resolved", "benchmark_results")) / "llm4ad_logs" / task.name / f"seed{seed}"
                if log_dir.exists():
                    import shutil
                    shutil.rmtree(log_dir, ignore_errors=True)
                log_dir.mkdir(parents=True, exist_ok=True)
                last_log_dir = log_dir

                profiler = EoHProfiler(log_dir=str(log_dir), create_random_path=False)
                random.seed(seed + attempt - 1)
                method = EoH(
                    llm=llm,
                    evaluation=evaluation,
                    profiler=profiler,
                    max_sample_nums=int(cfg.get("max_sample_nums", 24)),
                    max_generations=int(cfg.get("max_generations", 12)),
                    pop_size=int(cfg.get("pop_size", 6)),
                    selection_num=int(cfg.get("selection_num", 3)),
                    num_samplers=int(cfg.get("num_samplers", 1)),
                    num_evaluators=int(cfg.get("num_evaluators", 1)),
                    debug_mode=bool(cfg.get("debug_mode", False)),
                    multi_thread_or_process_eval=cfg.get("multi_thread_or_process_eval", "thread"),
                )

                method.run()
                best_value = self._extract_best_from_profiler(log_dir)
                last_best_value = best_value

                if best_value is not None and math.isfinite(best_value) and best_value < PENALTY_OBJECTIVE:
                    result["best_value"] = best_value
                    result["success"] = True
                    result["extra"] = {"log_dir": str(log_dir), "attempt": attempt, "repo_path": str(self._repo_path())}
                    break

            if not result["success"]:
                if last_best_value is not None and math.isfinite(last_best_value):
                    result["best_value"] = last_best_value
                    result["success"] = True
                    result["extra"] = {"log_dir": str(last_log_dir), "note": "best_value may be suboptimal (fallback)", "repo_path": str(self._repo_path())}
                else:
                    result["error"] = (
                        f"LLM4AD finished without a usable best score after {attempts} attempt(s). "
                        f"last_best={last_best_value!r} log_dir={last_log_dir}"
                    )

        except Exception:
            tb = traceback.format_exc()
            logger.error("LLM4AD run failed:\n%s", tb)
            result["error"] = tb

        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return result

    @staticmethod
    def _extract_best_from_profiler(log_dir: Path) -> float | None:
        candidates: list[float] = []
        best_json = log_dir / "samples" / "samples_best.json"
        if best_json.exists():
            try:
                data = json.loads(best_json.read_text(encoding="utf-8"))
                for item in data if isinstance(data, list) else [data]:
                    score = item.get("score")
                    if score is not None and math.isfinite(float(score)):
                        candidates.append(float(-float(score)))
            except Exception:
                pass
        if candidates:
            return min(candidates)
        return None

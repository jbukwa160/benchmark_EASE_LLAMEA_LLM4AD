"""Adapter for the current LLM4AD repository, hardened for unattended runs."""

from __future__ import annotations

import json
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
        pkg_root = repo_root / "llm4ad"

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
                name,
                str(init_file),
                submodule_search_locations=[str(search_path)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {name} from {init_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        ensure_package("llm4ad", pkg_root)
        load_pkg("llm4ad.base", pkg_root / "base" / "__init__.py", pkg_root / "base")
        ensure_package("llm4ad.tools", pkg_root / "tools")
        load_pkg("llm4ad.tools.profiler", pkg_root / "tools" / "profiler" / "__init__.py", pkg_root / "tools" / "profiler")
        ensure_package("llm4ad.method", pkg_root / "method")
        eoh_pkg = load_pkg("llm4ad.method.eoh", pkg_root / "method" / "eoh" / "__init__.py", pkg_root / "method" / "eoh")

        base_mod = sys.modules["llm4ad.base"]
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

            class _HTTPModel(LLM):
                def __init__(self, *, model: str, endpoint: str, timeout: int, temperature: float = 0.7):
                    super().__init__(do_auto_trim=False, debug_mode=False)
                    self.model = model
                    self.endpoint = endpoint
                    self.timeout = timeout
                    self.temperature = temperature

                def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
                    messages = [{"role": "user", "content": str(prompt)}]
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "stream": False,
                    }
                    data = json.dumps(payload).encode("utf-8")
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer ollama",
                    }
                    last_error = None
                    for attempt in range(1, 5):
                        req = request.Request(self.endpoint, data=data, headers=headers, method="POST")
                        try:
                            with request.urlopen(req, timeout=self.timeout) as resp:
                                raw = json.loads(resp.read().decode("utf-8"))
                            return raw["choices"][0]["message"]["content"]
                        except Exception as exc:  # pragma: no cover - network dependent
                            last_error = exc
                            if attempt == 4:
                                break
                            time.sleep(2 * attempt)
                    raise RuntimeError(f"Ollama request failed: {last_error}")

            template_program = (
                "import numpy as np\n\n"
                "def algorithm(func, dim, lb, ub, budget, rng):\n"
                "    \"\"\"Return the best 1-D numpy array found. Sequential only.\"\"\"\n"
                "    x = rng.uniform(lb, ub, size=dim)\n"
                "    return np.asarray(x, dtype=float)\n"
            )

            class _ContinuousEvaluation(Evaluation):
                def __init__(self):
                    super().__init__(
                        template_program=template_program,
                        task_description=(
                            f"{task.description}\n"
                            "Write a single sequential Python function only. "
                            "Do not use threads, multiprocessing, subprocesses, files, networking, eval, exec, or non-numpy imports."
                        ),
                        timeout_seconds=None,
                        exec_code=False,
                        safe_evaluate=False,
                    )

                def evaluate_program(self, program_str: str, callable_func=None, **kwargs):
                    mean_objective, _feedback, error = evaluate_candidate_code(
                        program_str,
                        task_name=task.name,
                        lower_bound=task.lower_bound,
                        upper_bound=task.upper_bound,
                        budget=budget,
                        eval_seeds=eval_seeds,
                        timeout_seconds=candidate_timeout,
                    )
                    if mean_objective >= PENALTY_OBJECTIVE and error:
                        return None
                    return float(-mean_objective)

            llm = _HTTPModel(
                model=self.ollama_model,
                endpoint=endpoint,
                timeout=int(cfg.get("llm_timeout_sec", 180)),
                temperature=float(cfg.get("temperature", 0.7)),
            )
            evaluation = _ContinuousEvaluation()
            attempts = int(cfg.get("attempts", 2))
            last_log_dir = None
            last_best_value = None
            for attempt in range(1, attempts + 1):
                log_dir = Path(self.global_cfg.get("_output_dir_resolved", "benchmark_results")) / "llm4ad_logs" / task.name / f"seed{seed}" / f"attempt{attempt}"
                last_log_dir = log_dir
                profiler = EoHProfiler(log_dir=str(log_dir), log_style="simple")

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
                best_value = _extract_best_from_profiler(log_dir)
                last_best_value = best_value
                if best_value is not None and best_value < PENALTY_OBJECTIVE:
                    result["best_value"] = best_value
                    result["success"] = True
                    result["extra"] = {"log_dir": str(log_dir), "attempt": attempt}
                    break

            if not result["success"]:
                raise RuntimeError(
                    f"LLM4AD finished without a valid best score after {attempts} attempt(s). "
                    f"Last best={last_best_value!r}. Last log_dir={last_log_dir}"
                )

        except ImportError as exc:
            msg = (
                f"LLM4AD import failed: {exc}. Make sure repo_path points to the current LLM4AD repo."
            )
            logger.error(msg)
            result["error"] = msg
        except Exception:
            tb = traceback.format_exc()
            logger.error("LLM4AD run failed:\n%s", tb)
            result["error"] = tb

        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return result


def _extract_best_from_profiler(log_dir: str | Path) -> float | None:
    import json

    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None

    best: float | None = None
    for path in log_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            candidates = [data]
        elif isinstance(data, list):
            candidates = [x for x in data if isinstance(x, dict)]
        else:
            candidates = []
        for item in candidates:
            score = item.get("score") or item.get("best_score") or item.get("best_value")
            if score is None:
                continue
            try:
                value = float(-float(score))
            except Exception:
                continue
            if best is None or value < best:
                best = value
    return best

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config
from .io_utils import atomic_write_json, ensure_dir
from .tasks import score_from_best_f, PENALTY_BEST_F
from .worker import result_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(Path(cfg["output_dir"]).expanduser().resolve())
    append_results = bool(cfg.get("append_results", False))

    frameworks = [name for name, fcfg in cfg["frameworks"].items() if fcfg.get("enabled", True)]
    tasks = list(cfg["tasks"])
    seeds = [int(x) for x in cfg["seeds"]]

    for framework in frameworks:
        framework_cfg = cfg["frameworks"][framework]
        run_timeout_seconds = int(framework_cfg.get("run_timeout_seconds", 1800))
        for task in tasks:
            for seed in seeds:
                out_path = result_path(output_dir, framework, task, seed)
                if append_results and out_path.exists():
                    continue
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmark_redesign.worker",
                    args.config,
                    framework,
                    task,
                    str(seed),
                ]
                started = time.time()
                try:
                    subprocess.run(cmd, check=True, timeout=run_timeout_seconds)
                except subprocess.TimeoutExpired:
                    payload = {
                        "framework": framework,
                        "task": task,
                        "seed": seed,
                        "status": "failed",
                        "score": score_from_best_f(PENALTY_BEST_F),
                        "best_f": PENALTY_BEST_F,
                        "duration_seconds": time.time() - started,
                        "error": f"Trial exceeded hard run timeout of {run_timeout_seconds} seconds",
                    }
                    atomic_write_json(out_path, payload)
                except subprocess.CalledProcessError as exc:
                    payload = {
                        "framework": framework,
                        "task": task,
                        "seed": seed,
                        "status": "failed",
                        "score": score_from_best_f(PENALTY_BEST_F),
                        "best_f": PENALTY_BEST_F,
                        "duration_seconds": time.time() - started,
                        "error": f"Worker exited with return code {exc.returncode}",
                    }
                    atomic_write_json(out_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

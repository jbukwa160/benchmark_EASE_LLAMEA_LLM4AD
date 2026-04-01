from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .config import load_config
from .io_utils import atomic_write_json, ensure_dir
from .tasks import score_from_best_f, PENALTY_BEST_F


def _make_adapter(name: str, framework_cfg: dict, global_cfg: dict):
    if name == "llamea":
        from .adapters.llamea_adapter import LLAMEAAdapter

        return LLAMEAAdapter(framework_cfg, global_cfg)
    if name == "llm4ad_eoh":
        from .adapters.llm4ad_eoh_adapter import LLM4ADEoHAdapter

        return LLM4ADEoHAdapter(framework_cfg, global_cfg)
    raise ValueError(f"Unsupported framework: {name}")


def result_path(output_dir: Path, framework: str, task: str, seed: int) -> Path:
    return output_dir / framework / task / f"seed_{seed}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("framework")
    parser.add_argument("task")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(Path(cfg["output_dir"]).expanduser().resolve())
    out_path = result_path(output_dir, args.framework, args.task, args.seed)

    framework_cfg = cfg["frameworks"][args.framework]
    adapter = _make_adapter(args.framework, framework_cfg, cfg)

    try:
        payload = adapter.run(args.task, args.seed)
    except Exception as exc:
        payload = {
            "framework": args.framework,
            "task": args.task,
            "seed": args.seed,
            "status": "failed",
            "score": score_from_best_f(PENALTY_BEST_F),
            "best_f": PENALTY_BEST_F,
            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }

    atomic_write_json(out_path, payload)
    print(json.dumps({"result_path": str(out_path), "status": payload.get("status", "unknown")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

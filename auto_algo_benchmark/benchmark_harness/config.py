"""
config.py — Load and validate benchmark_config.json
"""

import json
import os
from pathlib import Path


def load_config(path: str = "benchmark_config.json") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def resolve_path(base_dir: Path, relative: str) -> Path:
    """Resolve a potentially relative path from config relative to config file dir."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()

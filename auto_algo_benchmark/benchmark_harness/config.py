"""
config.py — Load and validate benchmark_config.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_config(path: str = "benchmark_config.json") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def resolve_path(base_dir: Path, relative: str) -> Path:
    """Resolve a potentially relative path from config relative to config file dir."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _package_present(path: Path, package_name: str) -> bool:
    return (path / package_name).exists() or (path / package_name.replace("-", "_")).exists()


def resolve_repo_path(base_dir: Path, configured_path: str, package_name: str) -> Path:
    """
    Resolve a framework repository path robustly.

    This accepts the configured path, but also auto-detects common extracted-zip
    directory names such as ``LLaMEA-main`` / ``LLM4AD-main`` and handles the case
    where the configured directory contains a single nested child that is the true
    repo root.
    """
    configured = Path(configured_path)
    direct = configured if configured.is_absolute() else (base_dir / configured).resolve()

    candidates: list[Path] = [direct]

    name = configured.name or package_name
    parent = direct.parent if str(direct.parent) else base_dir
    stem_candidates = {name, package_name, package_name.upper(), package_name.lower()}
    for stem in list(stem_candidates):
        stem_candidates.add(f"{stem}-main")
        stem_candidates.add(f"{stem}-master")
        stem_candidates.add(f"{stem}_main")
        stem_candidates.add(f"{stem}_master")
    for stem in stem_candidates:
        candidates.append(parent / stem)
        candidates.append(base_dir / stem)
        candidates.append(base_dir.parent / stem)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    for candidate in ordered:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if _package_present(candidate, package_name):
            return candidate
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        package_children = [d for d in subdirs if _package_present(d, package_name)]
        if len(package_children) == 1:
            return package_children[0]

    return direct

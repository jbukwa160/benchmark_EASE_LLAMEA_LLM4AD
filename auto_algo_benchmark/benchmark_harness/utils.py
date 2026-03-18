from __future__ import annotations

import contextlib
import csv
import importlib
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import psutil


@dataclass
class RunSummary:
    framework: str
    benchmark: str
    seed: int
    status: str
    best_search_score: float | None
    raw_objective_mean: float | None
    runtime_sec: float
    peak_rss_mb: float | None
    candidates_evaluated: int
    artifact_dir: str
    notes: str = ""


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: str | Path, row: dict[str, Any], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if fieldnames is None:
        fieldnames = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_json_file(path: str | Path) -> Any:
    import json
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_file(path: str | Path, payload: Any) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_repo_import_root(repo_path: str | Path, package_name: str) -> Path:
    repo_path = Path(repo_path).resolve()
    candidates = [repo_path, *[p for p in repo_path.iterdir() if p.is_dir()]]
    for candidate in candidates:
        if (candidate / package_name).exists():
            return candidate
    raise FileNotFoundError(f"Could not find package '{package_name}' below {repo_path}")


def import_from_repo(repo_path: str | Path, package_name: str) -> None:
    root = resolve_repo_import_root(repo_path, package_name)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    importlib.invalidate_caches()


@contextlib.contextmanager
def pushd(path: str | Path):
    old = Path.cwd()
    os.chdir(Path(path))
    try:
        yield
    finally:
        os.chdir(old)


class ResourceMonitor:
    def __init__(self, poll_interval: float = 0.25):
        self._process = psutil.Process()
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self.peak_rss_bytes = 0
        self._thread: threading.Thread | None = None
        self.start_time = 0.0
        self.end_time = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            except Exception:
                pass
            self._stop.wait(self._poll_interval)

    def __enter__(self):
        self.start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end_time = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def runtime_sec(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def case_insensitive_get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    lowered = key.lower()
    for k, v in mapping.items():
        if k.lower() == lowered:
            return v
    return default


def flatten(iterable: Iterable[Iterable[Any]]) -> list[Any]:
    out: list[Any] = []
    for part in iterable:
        out.extend(part)
    return out

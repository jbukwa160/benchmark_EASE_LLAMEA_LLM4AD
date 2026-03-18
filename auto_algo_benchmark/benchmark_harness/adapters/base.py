from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..tasks import BenchmarkTask
from ..utils import RunSummary


class FrameworkAdapter(ABC):
    def __init__(self, framework_name: str, framework_cfg: dict[str, Any], global_cfg: dict[str, Any], output_dir: str | Path):
        self.framework_name = framework_name
        self.framework_cfg = framework_cfg
        self.global_cfg = global_cfg
        self.output_dir = Path(output_dir)

    @abstractmethod
    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict[str, Any]]]:
        raise NotImplementedError

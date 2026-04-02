from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FrameworkAdapter(ABC):
    def __init__(self, framework_name: str, framework_cfg: dict[str, Any], global_cfg: dict[str, Any]):
        self.framework_name = framework_name
        self.framework_cfg = framework_cfg
        self.global_cfg = global_cfg

    @abstractmethod
    def run(self, task_name: str, seed: int) -> dict[str, Any]:
        raise NotImplementedError

"""
adapters/base.py — Abstract base class for all framework adapters.
"""

import abc
from typing import Any

from ..tasks import BenchmarkTask


class BaseAdapter(abc.ABC):
    """
    Common interface every framework adapter must implement.

    A single `run()` call should:
      1. Invoke the framework for ONE (task, seed) combination.
      2. Return a result dict that will be persisted by the harness.
    """

    name: str  # subclass must set this

    def __init__(self, framework_cfg: dict, global_cfg: dict):
        self.framework_cfg = framework_cfg
        self.global_cfg = global_cfg
        self.ollama_model: str = global_cfg["ollama"]["model"]
        self.ollama_base_url: str = global_cfg["ollama"]["base_url"]
        self.task_defaults: dict = global_cfg.get("task_defaults", {})

    @abc.abstractmethod
    def is_enabled(self) -> bool:
        ...

    @abc.abstractmethod
    def run(self, task: BenchmarkTask, seed: int) -> dict[str, Any]:
        """
        Execute the framework on the given task with the given random seed.

        Returns a dict with at minimum:
          - framework: str
          - task: str
          - seed: int
          - best_value: float   (raw objective, lower=better)
          - success: bool
          - error: str | None
          - elapsed_sec: float
          - extra: dict         (framework-specific metadata)
        """
        ...

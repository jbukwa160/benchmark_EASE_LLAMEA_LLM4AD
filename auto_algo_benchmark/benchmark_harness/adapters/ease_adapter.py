"""
adapters/ease_adapter.py — Adapter for frontEASE (TBU-AILab/frontEASE).

frontEASE is a web-based platform; we interact with it via its REST API.
The adapter:
  1. Clones a template task to create a per-run task.
  2. Uploads our benchmark problem definition.
  3. Polls until the run finishes.
  4. Downloads the best result.
  5. Optionally deletes the cloned task.

Config keys used (under frameworks.ease):
  api_base_url      - e.g. http://localhost:4000
  username          - login username
  password          - login password
  template_task_id  - ID of a pre-configured EASE task to clone
  poll_interval_sec - how often to poll (default 5)
  max_wait_sec      - give up after this many seconds (default 3600)
  delete_clones_after_run - whether to clean up (default False)
"""

import time
import traceback
from typing import Any

import requests

from .base import BaseAdapter
from ..tasks import BenchmarkTask
from ..utils import get_logger, timestamp

logger = get_logger(__name__)


class EASEAdapter(BaseAdapter):
    name = "ease"

    def is_enabled(self) -> bool:
        return self.framework_cfg.get("enabled", False)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_token(self, session: requests.Session) -> str:
        cfg = self.framework_cfg
        url = cfg["api_base_url"].rstrip("/") + "/api/auth/login"
        resp = session.post(url, json={
            "username": cfg["username"],
            "password": cfg["password"],
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        token = data.get("token") or data.get("access_token") or data.get("accessToken")
        if not token:
            raise RuntimeError(f"Could not extract token from login response: {data}")
        return token

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _clone_task(self, session: requests.Session, headers: dict, template_id: str,
                    task_name: str) -> str:
        cfg = self.framework_cfg
        url = cfg["api_base_url"].rstrip("/") + f"/api/tasks/{template_id}/clone"
        resp = session.post(url, json={"name": task_name}, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cloned_id = data.get("id") or data.get("task_id") or data.get("taskId")
        if not cloned_id:
            raise RuntimeError(f"Could not get cloned task ID from: {data}")
        return str(cloned_id)

    def _upload_problem(self, session: requests.Session, headers: dict,
                        task_id: str, task: BenchmarkTask):
        """Upload our problem definition so EASE knows what to optimise."""
        cfg = self.framework_cfg
        url = cfg["api_base_url"].rstrip("/") + f"/api/tasks/{task_id}/problem"
        payload = {
            "name": task.name,
            "description": task.description,
            "dimension": task.dimension,
            "lower_bound": task.lower_bound,
            "upper_bound": task.upper_bound,
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
        }
        resp = session.put(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()

    def _start_run(self, session: requests.Session, headers: dict, task_id: str):
        cfg = self.framework_cfg
        url = cfg["api_base_url"].rstrip("/") + f"/api/tasks/{task_id}/run"
        resp = session.post(url, headers=headers, timeout=30)
        resp.raise_for_status()

    def _poll_status(self, session: requests.Session, headers: dict, task_id: str) -> str:
        cfg = self.framework_cfg
        url = cfg["api_base_url"].rstrip("/") + f"/api/tasks/{task_id}"
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("status") or data.get("state") or "unknown"

    def _get_best_result(self, session: requests.Session, headers: dict,
                         task_id: str) -> float | None:
        cfg = self.framework_cfg
        url = cfg["api_base_url"].rstrip("/") + f"/api/tasks/{task_id}/results"
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Try common result shapes
        if isinstance(data, list) and data:
            best = min(
                r.get("fitness") or r.get("score") or r.get("value") or float("inf")
                for r in data
            )
            return float(best) if best < float("inf") else None
        if isinstance(data, dict):
            for key in ["best_fitness", "best_score", "best_value", "fitness"]:
                if key in data:
                    return float(data[key])
        return None

    def _delete_task(self, session: requests.Session, headers: dict, task_id: str):
        try:
            cfg = self.framework_cfg
            url = cfg["api_base_url"].rstrip("/") + f"/api/tasks/{task_id}"
            session.delete(url, headers=headers, timeout=30)
        except Exception as exc:
            logger.warning(f"Could not delete EASE task {task_id}: {exc}")

    # ------------------------------------------------------------------

    def run(self, task: BenchmarkTask, seed: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        cfg = self.framework_cfg
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

        cloned_id = None
        try:
            session = requests.Session()
            token = self._get_token(session)
            headers = {"Authorization": f"Bearer {token}",
                       "Content-Type": "application/json"}

            template_id = cfg.get("template_task_id", "")
            if not template_id or "PUT-YOUR" in template_id:
                raise RuntimeError(
                    "ease.template_task_id is not configured in benchmark_config.json. "
                    "Please set it to a valid frontEASE task ID."
                )

            task_name = f"bench_{task.name}_seed{seed}_{int(time.time())}"
            logger.info(f"[EASE] Cloning template task {template_id!r} → {task_name!r}")
            cloned_id = self._clone_task(session, headers, template_id, task_name)

            self._upload_problem(session, headers, cloned_id, task)
            self._start_run(session, headers, cloned_id)

            poll_interval = cfg.get("poll_interval_sec", 5)
            max_wait = cfg.get("max_wait_sec", 3600)
            deadline = time.time() + max_wait
            status = "running"

            while time.time() < deadline:
                time.sleep(poll_interval)
                status = self._poll_status(session, headers, cloned_id)
                logger.debug(f"[EASE] task {cloned_id} status: {status}")
                if status in ("completed", "done", "finished", "success"):
                    break
                if status in ("failed", "error", "cancelled"):
                    raise RuntimeError(f"EASE task ended with status: {status}")
            else:
                raise TimeoutError(f"EASE task did not finish within {max_wait}s")

            best_value = self._get_best_result(session, headers, cloned_id)
            result["best_value"] = best_value
            result["success"] = True
            result["extra"]["ease_task_id"] = cloned_id
            result["extra"]["final_status"] = status

        except Exception:
            tb = traceback.format_exc()
            logger.error(f"EASE run failed:\n{tb}")
            result["error"] = tb
        finally:
            if cloned_id and cfg.get("delete_clones_after_run", False):
                try:
                    session = requests.Session()
                    token = self._get_token(session)
                    headers = {"Authorization": f"Bearer {token}"}
                    self._delete_task(session, headers, cloned_id)
                except Exception:
                    pass

        result["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return result

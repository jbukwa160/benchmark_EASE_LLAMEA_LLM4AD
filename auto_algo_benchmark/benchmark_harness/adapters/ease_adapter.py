from __future__ import annotations

import time
import traceback
from pathlib import Path

import requests

from .base import FrameworkAdapter
from ..tasks import BenchmarkTask
from ..utils import RunSummary, case_insensitive_get, ensure_dir


class EaseAdapter(FrameworkAdapter):
    RUN_STATE = 2
    FINISH_STATE = 5
    BREAK_STATE = 6

    def __init__(self, framework_cfg: dict, global_cfg: dict, output_dir: str | Path):
        super().__init__("ease", framework_cfg, global_cfg, output_dir)
        self.base_url = self.framework_cfg["api_base_url"].rstrip("/")
        self.session = requests.Session()

    def _login(self) -> None:
        response = self.session.post(
            f"{self.base_url}/login",
            json={
                "email": self.framework_cfg["username"],
                "password": self.framework_cfg["password"],
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        token = case_insensitive_get(payload, "accessToken") or case_insensitive_get(payload, "token")
        if not token:
            raise RuntimeError("Login succeeded but no access token was returned")
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _clone_task(self, template_task_id: str, new_name: str) -> str:
        response = self.session.post(
            f"{self.base_url}/api/tasks/{template_task_id}/clone",
            json={"name": new_name, "copies": 1},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("Clone endpoint did not return a task list")
        task = payload[0]
        task_id = case_insensitive_get(task, "id")
        if not task_id:
            raise RuntimeError("Cloned task has no ID")
        return str(task_id)

    def _change_state(self, task_id: str, state: int) -> None:
        response = self.session.patch(
            f"{self.base_url}/api/tasks/change-state/{state}",
            params={"taskIDs": [task_id]},
            timeout=120,
        )
        response.raise_for_status()

    def _load_task(self, task_id: str) -> dict:
        response = self.session.get(f"{self.base_url}/api/tasks/{task_id}", timeout=120)
        response.raise_for_status()
        return response.json()

    def _delete_task(self, task_id: str) -> None:
        response = self.session.delete(f"{self.base_url}/api/tasks", params={"taskIDs": [task_id]}, timeout=120)
        response.raise_for_status()

    def run_one(self, task: BenchmarkTask, seed: int) -> tuple[RunSummary, list[dict]]:
        run_dir = ensure_dir(self.output_dir / "artifacts" / "ease" / task.name / f"seed_{seed}")
        del run_dir  # artifact folder reserved for symmetry
        progress_rows: list[dict] = []
        status = "success"
        notes = ""
        best_score = None
        raw_best = None
        runtime_sec = 0.0
        peak_rss_mb = None
        candidates_evaluated = 0
        task_id = None
        started = time.perf_counter()

        try:
            self._login()
            template_task_id = self.framework_cfg["template_task_id"]
            clone_name = f"{task.name}_seed_{seed}_{int(time.time())}"
            task_id = self._clone_task(template_task_id, clone_name)
            self._change_state(task_id, self.RUN_STATE)

            poll_interval = float(self.framework_cfg.get("poll_interval_sec", 5))
            max_wait = float(self.framework_cfg.get("max_wait_sec", 3600))
            last_iteration = -1

            while True:
                task_payload = self._load_task(task_id)
                state = case_insensitive_get(task_payload, "state")
                current_iteration = int(case_insensitive_get(task_payload, "currentIteration", 0))
                solutions = case_insensitive_get(task_payload, "solutions", []) or []
                solution_scores = []
                for sol in solutions:
                    score = case_insensitive_get(sol, "fitness")
                    try:
                        solution_scores.append(float(score))
                    except Exception:
                        continue
                current_best = max(solution_scores) if solution_scores else None

                if current_iteration != last_iteration or current_best is not None:
                    progress_rows.append({
                        "framework": "ease",
                        "benchmark": task.name,
                        "seed": seed,
                        "sample_index": current_iteration,
                        "elapsed_sec": time.perf_counter() - started,
                        "candidate_score": current_best,
                        "best_so_far": current_best,
                    })
                    last_iteration = current_iteration

                if state in (self.FINISH_STATE, "FINISH"):
                    break
                if state in (self.BREAK_STATE, "BREAK"):
                    status = "failed"
                    notes = "EASE task ended in BREAK state"
                    break
                if time.perf_counter() - started > max_wait:
                    status = "failed"
                    notes = "Timed out while waiting for EASE task"
                    break
                time.sleep(poll_interval)

            runtime_sec = time.perf_counter() - started
            if progress_rows:
                last = progress_rows[-1]
                best_score = last["best_so_far"]
                raw_best = last["best_so_far"]
                candidates_evaluated = max(0, int(last["sample_index"]))
        except Exception as exc:
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            runtime_sec = time.perf_counter() - started
        finally:
            if task_id and self.framework_cfg.get("delete_clones_after_run", False):
                try:
                    self._delete_task(task_id)
                except Exception:
                    pass

        summary = RunSummary(
            framework="ease",
            benchmark=task.name,
            seed=seed,
            status=status,
            best_search_score=best_score,
            raw_objective_mean=raw_best,
            runtime_sec=runtime_sec,
            peak_rss_mb=peak_rss_mb,
            candidates_evaluated=candidates_evaluated,
            artifact_dir=str((self.output_dir / "artifacts" / "ease" / task.name / f"seed_{seed}").resolve()),
            notes=notes,
        )
        return summary, progress_rows

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class OllamaConfig:
    model: str
    base_url: str
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 90.0
    hard_timeout_seconds: float = 120.0
    retries: int = 1
    temperature: float = 0.2


def _chat_worker(queue: mp.Queue, payload: dict[str, Any], url: str, connect_timeout: float, read_timeout: float) -> None:
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(connect_timeout, read_timeout),
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            queue.put((False, f"Empty Ollama response: {json.dumps(data)[:500]}"))
            return
        queue.put((True, content))
    except Exception as exc:  # pragma: no cover - exercised only at runtime with network access
        queue.put((False, f"{type(exc).__name__}: {exc}"))


class RobustOllamaClient:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self._url = cfg.base_url.rstrip("/") + "/api/chat"

    def chat(self, messages: list[dict[str, str]], *, hard_timeout_seconds: float | None = None) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
            },
        }
        timeout = hard_timeout_seconds or self.cfg.hard_timeout_seconds
        attempts = max(1, int(self.cfg.retries) + 1)
        last_error = "Unknown Ollama error"
        for attempt in range(1, attempts + 1):
            queue: mp.Queue = mp.Queue()
            process = mp.Process(
                target=_chat_worker,
                args=(queue, payload, self._url, self.cfg.connect_timeout_seconds, self.cfg.read_timeout_seconds),
                daemon=True,
            )
            process.start()
            process.join(timeout)
            if process.is_alive():
                process.kill()
                process.join()
                last_error = f"Ollama hard timeout after {timeout:.1f}s"
            else:
                try:
                    ok, value = queue.get_nowait()
                except Exception:
                    ok, value = False, "Ollama worker exited without returning data"
                if ok:
                    return value
                last_error = str(value)
            if attempt < attempts:
                time.sleep(min(3.0 * attempt, 10.0))
        raise TimeoutError(last_error)

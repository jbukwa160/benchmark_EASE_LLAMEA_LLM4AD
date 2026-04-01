"""Minimal pytz compatibility shim for llm4ad on Python 3.11+."""

from __future__ import annotations

from datetime import timezone as _timezone
from zoneinfo import ZoneInfo

UTC = _timezone.utc


def timezone(name: str):
    if name.upper() == "UTC":
        return UTC
    return ZoneInfo(name)

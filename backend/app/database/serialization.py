"""Sanitize and serialize JSON documents before storing investigation data."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "bearer_token",
    "authorization",
    "credential",
)
_EXCLUDED_KEYS = {"scenario_label", "is_anomaly"}
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_ -]?key|password|access[_ -]?token|secret)\s*[:=]\s*"
    r"(['\"]?)[^\s,;}'\"]+"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")


def sanitize_json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _EXCLUDED_KEYS or any(
                sensitive in normalized for sensitive in _SENSITIVE_KEY_PARTS
            ):
                continue
            cleaned[str(key)] = sanitize_json(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        return _BEARER_TOKEN.sub("Bearer [REDACTED]", _SECRET_ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}=[REDACTED]", value
        ))
    return value

"""Check connectivity to the configured NVIDIA chat model."""

from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1"
TEST_MESSAGE = "Hello! Reply with exactly: NVIDIA API connection successful."


def _read_setting(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value.strip().strip("\"'")

    env_path = BACKEND_ROOT / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        setting, separator, configured_value = line.partition("=")
        if separator and setting.strip() == name:
            return configured_value.strip().rstrip(",").strip().strip("\"'")
    return ""


def _http_status(error: Exception) -> Any | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status


def _safe_error_message(error: Exception, api_key: str) -> str:
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return re.sub(r"(?i)nvapi-[A-Za-z0-9_-]+", "[REDACTED]", message)


def main() -> int:
    """Send one small test request when the provider configuration is available."""
    model = _read_setting("NVIDIA_MODEL")
    api_key = _read_setting("NVIDIA_API_KEY")
    problems = []
    if not model:
        problems.append("NVIDIA_MODEL is not configured.")
    if not api_key:
        problems.append("NVIDIA_API_KEY is not configured.")

    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except ImportError:
        ChatNVIDIA = None
        problems.append("langchain-nvidia-ai-endpoints is not installed.")

    if problems:
        print("API connection: FAILED")
        for problem in problems:
            print(f"Error: {problem}")
        return 1

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    rf"Found {re.escape(model)} in available_models, but type is "
                    r"unknown and inference may fail\."
                ),
                category=UserWarning,
                module=r"langchain_nvidia_ai_endpoints\._common",
            )
            client = ChatNVIDIA(
                model=model,
                api_key=api_key,
                base_url=NVIDIA_ENDPOINT,
                temperature=0.1,
                max_completion_tokens=64,
            )
        start_time = time.perf_counter()
        response = client.invoke(TEST_MESSAGE)
        elapsed_seconds = time.perf_counter() - start_time
    except Exception as error:
        print("API connection: FAILED")
        status = _http_status(error)
        if status is not None:
            print(f"HTTP status code: {status}")
        print(f"Error: {_safe_error_message(error, api_key)}")
        return 1

    print("API connection: SUCCESS")
    print(f"Model name: {model}")
    print(f"Response from NVIDIA: {response.content}")
    print(f"Response time: {elapsed_seconds:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
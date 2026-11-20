"""Safe PostgreSQL configuration loaded from the process or backend .env."""

from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy.engine import URL, make_url

BACKEND_ROOT = Path(__file__).resolve().parents[2]
_SETTING_NAMES = {
    "DATABASE_URL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
}


def _load_database_environment() -> None:
    env_path = BACKEND_ROOT / ".env"
    if not env_path.exists():
        return
    selected_lines = []
    with env_path.open(encoding="utf-8") as env_file:
        for line in env_file:
            assignment = re.match(
                r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=",
                line,
            )
            if assignment and assignment.group(1) in _SETTING_NAMES:
                selected_lines.append(line)
    values = dotenv_values(stream=StringIO("".join(selected_lines)))
    for name in _SETTING_NAMES:
        value = values.get(name)
        if value is not None:
            os.environ.setdefault(name, value)


_load_database_environment()


def database_url() -> URL:
    """Build the PostgreSQL URL without exposing it in errors or logs."""
    configured_url = os.getenv("DATABASE_URL", "").strip()
    if configured_url:
        try:
            url = make_url(configured_url)
        except Exception:
            raise ValueError("DATABASE_URL must be a valid PostgreSQL URL.") from None
        if url.get_backend_name() != "postgresql":
            raise ValueError("DATABASE_URL must use PostgreSQL.")
        return url.set(drivername="postgresql+psycopg")

    host = os.getenv("POSTGRES_HOST", "").strip()
    database = os.getenv("POSTGRES_DB", "").strip()
    username = os.getenv("POSTGRES_USER", "").strip()
    password = os.getenv("POSTGRES_PASSWORD", "")
    port_text = os.getenv("POSTGRES_PORT", "").strip() or "5432"
    if not all((host, database, username, password)):
        raise ValueError("PostgreSQL connection settings are incomplete.")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError("POSTGRES_PORT must be an integer.") from None
    if not 1 <= port <= 65535:
        raise ValueError("POSTGRES_PORT must be between 1 and 65535.")
    return URL.create(
        "postgresql+psycopg",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
    )

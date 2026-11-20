"""Environment-backed application configuration."""

import os


def _integer_setting(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _boolean_setting(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


RAG_CHUNK_SIZE = _integer_setting("RAG_CHUNK_SIZE", 224)
RAG_CHUNK_OVERLAP = _integer_setting("RAG_CHUNK_OVERLAP", 40)
RAG_EMBEDDING_MAX_TOKENS = _integer_setting("RAG_EMBEDDING_MAX_TOKENS", 256)
RAG_EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
RAG_LOCAL_FILES_ONLY = _boolean_setting(
    "RAG_LOCAL_FILES_ONLY",
    default=_boolean_setting("HF_HUB_OFFLINE", False)
    or _boolean_setting("TRANSFORMERS_OFFLINE", False),
)

APPLICATION_NAME = "Faultline Incident Response"
API_VERSION = os.getenv("API_VERSION", "1.0.0")
_DEFAULT_CORS_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173"
CORS_ALLOWED_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        _DEFAULT_CORS_ORIGINS,
    ).split(",")
    if origin.strip()
)

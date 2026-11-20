"""Shared validation helpers for investigation schemas."""

from datetime import datetime


def validate_time_range(start: datetime | None, end: datetime | None) -> None:
    """Require comparable, strictly increasing timestamps when both exist."""
    if start is None or end is None:
        return
    if (start.utcoffset() is None) != (end.utcoffset() is None):
        raise ValueError("start and end must both be timezone-aware or naive.")
    if start >= end:
        raise ValueError("end must be later than start.")

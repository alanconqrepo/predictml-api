"""
Shared project utilities
"""

from datetime import datetime, timezone


def _utcnow() -> datetime:
    """Return the current UTC datetime without timezone information.

    Returns:
        datetime: Current UTC datetime, naive (no tzinfo).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

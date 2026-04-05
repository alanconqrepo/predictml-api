"""
Utilitaires partagés du projet
"""

from datetime import datetime, timezone


def _utcnow() -> datetime:
    """Retourne la date/heure UTC courante sans information de timezone.

    Returns:
        datetime: Date/heure UTC courante, naive (sans tzinfo).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

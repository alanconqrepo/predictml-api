from urllib.parse import urlparse, urlunparse

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.config import settings


def _rate_limit_storage_uri() -> str:
    """Return a Redis URI for rate-limit counters on DB /1 (separate from the model cache on /0).

    When REDIS_SENTINEL_HOSTS is set, returns a redis+sentinel:// URI so the
    limits library connects via Sentinel instead of directly to the master.
    """
    if settings.REDIS_SENTINEL_HOSTS:
        parsed = urlparse(settings.REDIS_URL)
        auth = f":{parsed.password}@" if parsed.password else ""
        return f"redis+sentinel://{auth}{settings.REDIS_SENTINEL_HOSTS}/mymaster/1"
    parsed = urlparse(settings.REDIS_URL)
    return urlunparse(parsed._replace(path="/1"))


# Set RATELIMIT_ENABLED=0 to disable per-IP rate limiting (e.g. in tests).
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_rate_limit_storage_uri(),
)

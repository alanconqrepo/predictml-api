from urllib.parse import urlparse, urlunparse

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.config import settings


def _rate_limit_storage_uri() -> str:
    """Return a Redis URI for rate-limit counters on DB /1 (separate from the model cache on /0).

    When REDIS_SENTINEL_HOSTS is set, returns a redis+sentinel:// URI so the
    limits library connects via Sentinel instead of directly to the master.
    The password in the URI is for the Redis master only; sentinels have no auth,
    so sentinel_kwargs={"password": None} must be passed when building the Limiter.
    """
    if settings.REDIS_SENTINEL_HOSTS:
        parsed = urlparse(settings.REDIS_URL)
        auth = f":{parsed.password}@" if parsed.password else ""
        return f"redis+sentinel://{auth}{settings.REDIS_SENTINEL_HOSTS}/mymaster"
    parsed = urlparse(settings.REDIS_URL)
    return urlunparse(parsed._replace(path="/1"))


def _rate_limit_storage_options() -> dict:
    """Extra kwargs forwarded to the limits storage backend.

    When Sentinel is active the sentinels themselves have no auth, but the
    limits library would otherwise propagate the master password to sentinel
    connections, causing AuthenticationError and request timeouts.
    """
    if settings.REDIS_SENTINEL_HOSTS:
        return {"sentinel_kwargs": {"password": None}}
    return {}


# Set RATELIMIT_ENABLED=0 to disable per-IP rate limiting (e.g. in tests).
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_rate_limit_storage_uri(),
    storage_options=_rate_limit_storage_options(),
)

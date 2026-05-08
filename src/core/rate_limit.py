from slowapi import Limiter
from slowapi.util import get_remote_address

# Set RATELIMIT_ENABLED=0 to disable per-IP rate limiting (e.g. in tests).
limiter = Limiter(key_func=get_remote_address)

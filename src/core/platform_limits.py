"""
Process resource-limit utilities — cross-platform (Unix and Windows).

On Windows, ``resource`` does not exist; all operations are no-ops and
``preexec_fn`` (unsupported by asyncio.create_subprocess_exec) is omitted
from the kwargs dict.
"""

import sys

if sys.platform != "win32":
    import resource as _resource  # Unix-only module


def set_subprocess_limits() -> None:
    """
    Called in the child process after fork (preexec_fn) — reduces the attack surface.

    RLIMIT_NOFILE: 1024 minimum required — importlib_metadata opens the metadata
    of all installed packages when mlflow is imported (200+ simultaneous file
    descriptors).

    No-op on Windows: the ``resource`` module does not exist on that platform.
    Resource limit management is handled by the operating system.
    """
    if sys.platform == "win32":
        return
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _fd_limit = min(1024, _hard)
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_fd_limit, _fd_limit))


# Extra kwargs for asyncio.create_subprocess_exec.
# preexec_fn is not supported on Windows (raises ValueError).
SUBPROCESS_PREEXEC_KWARGS: dict = (
    {} if sys.platform == "win32" else {"preexec_fn": set_subprocess_limits}
)

"""
Utilitaires de limitation de ressources processus — cross-platform (Unix et Windows).

Sur Windows, ``resource`` n'existe pas ; toutes les opérations sont des no-ops et
``preexec_fn`` (non supporté par asyncio.create_subprocess_exec) est omis des kwargs.
"""
import sys

if sys.platform != "win32":
    import resource as _resource  # module Unix uniquement


def set_subprocess_limits() -> None:
    """
    Appelé dans le processus enfant après fork (preexec_fn) — réduit la surface d'attaque.

    RLIMIT_NOFILE : 1024 minimum requis — importlib_metadata ouvre les métadonnées de
    tous les paquets installés à l'import de mlflow (200+ fd simultanément).

    No-op sur Windows : le module ``resource`` n'existe pas sur cette plateforme.
    La gestion des limites de ressources est assurée par le système d'exploitation.
    """
    if sys.platform == "win32":
        return
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _fd_limit = min(1024, _hard)
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_fd_limit, _fd_limit))


# Kwargs supplémentaires pour asyncio.create_subprocess_exec.
# preexec_fn n'est pas supporté sur Windows (lève ValueError).
SUBPROCESS_PREEXEC_KWARGS: dict = (
    {} if sys.platform == "win32"
    else {"preexec_fn": set_subprocess_limits}
)

"""
Audit logging for sensitive admin operations.

Emits structured log lines tagged AUDIT so they can be filtered,
shipped to a SIEM, or stored in a dedicated log stream.
"""

import structlog

_audit_logger = structlog.get_logger("audit")


def audit_log(action: str, actor_id: int, resource: str, details: dict = {}) -> None:
    """Emit a structured audit log entry.

    Args:
        action:    Dot-namespaced action (e.g. "model.upload", "user.delete").
        actor_id:  ID of the authenticated user who triggered the action.
        resource:  Target resource identifier (e.g. "iris:1.0.0", "user:42").
        details:   Optional extra key/value pairs added to the log line.
    """
    _audit_logger.info("AUDIT", action=action, actor_id=actor_id, resource=resource, **details)

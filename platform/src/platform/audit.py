"""Audit log: append-only record of organisation-scoped actions (organisations plan, T9).

Services call :func:`record` inside their own transaction so the audit row commits or
rolls back together with the change it describes. Nothing here commits.

Reads go through ``GET /audit-log``, which is admin-only and always scoped to the
caller's organisation.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .context import AuthContext, require
from .db import get_db
from .models import AuditLog
from .validation import SECRET_KEY_PARTS

router = APIRouter(tags=["audit"])

MAX_LIMIT = 200
DEFAULT_LIMIT = 50


def _is_secret_key(key: Any) -> bool:
    text = str(key).casefold()
    return any(part in text for part in SECRET_KEY_PARTS)


def _strip_secrets(value: Any) -> Any:
    """Drop (not mask) dict entries whose key names a secret, recursing into nesting."""

    if isinstance(value, dict):
        return {
            key: _strip_secrets(nested)
            for key, nested in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, list):
        return [_strip_secrets(item) for item in value]
    return value


def record(
    db: Session,
    ctx: AuthContext,
    action: str,
    *,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditLog:
    """Add an audit row to the caller's transaction. The caller owns commit/rollback."""

    entry = AuditLog(
        organization_id=ctx.org_id,
        actor_type=ctx.principal,
        actor_id=ctx.key_id or ctx.session_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        metadata_json=_strip_secrets(dict(metadata or {})),
        request_id=ctx.request_id,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    return entry


def _serialize(entry: AuditLog) -> dict[str, Any]:
    return {
        "id": entry.id,
        "action": entry.action,
        "actor_type": entry.actor_type,
        "actor_id": entry.actor_id,
        "resource_type": entry.resource_type,
        "resource_id": entry.resource_id,
        "metadata": entry.metadata_json or {},
        "request_id": entry.request_id,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
    }


@router.get("/audit-log")
def list_audit_log(
    resource_type: str | None = Query(default=None, max_length=40),
    resource_id: str | None = Query(default=None, max_length=36),
    action: str | None = Query(default=None, max_length=80),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    before: datetime | None = Query(
        default=None, description="Return entries created strictly before this time."
    ),
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    statement = select(AuditLog).where(AuditLog.organization_id == ctx.org_id)
    if resource_type is not None:
        statement = statement.where(AuditLog.resource_type == resource_type)
    if resource_id is not None:
        statement = statement.where(AuditLog.resource_id == resource_id)
    if action is not None:
        statement = statement.where(AuditLog.action == action)
    if before is not None:
        if before.tzinfo is None:
            before = before.replace(tzinfo=timezone.utc)
        statement = statement.where(AuditLog.created_at < before)
    statement = statement.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).limit(
        limit
    )
    entries = db.scalars(statement).all()
    return {"items": [_serialize(entry) for entry in entries]}

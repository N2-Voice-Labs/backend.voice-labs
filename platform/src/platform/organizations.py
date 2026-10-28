"""Organisation and API-key management API (organisations plan, T7), plus helpers.

Org-scoped keys carry a role (owner > admin > member). Keys can only be minted or
revoked at or below the caller's own role; superadmin (``PLATFORM_API_KEYS``) and dev
callers may act at any role. Plaintext keys appear only in the create response.
"""

import dataclasses
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import audit
from .auth import generate_api_key
from .context import AuthContext, require, role_at_least
from .db import get_db
from .models import DEFAULT_ORGANIZATION_ID, ApiKey, Organization

router = APIRouter(tags=["organizations"])

_SUPERADMIN_PRINCIPALS = frozenset({"superadmin", "dev"})
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


def ensure_default_org(db: Session) -> Organization:
    """Create the default organisation if missing; safe to call repeatedly."""
    existing = db.get(Organization, DEFAULT_ORGANIZATION_ID)
    if existing is not None:
        return existing
    org = Organization(
        id=DEFAULT_ORGANIZATION_ID, name="Default", slug="default", status="active"
    )
    db.add(org)
    try:
        db.commit()
    except IntegrityError:
        # Concurrent startup (or a migration-seeded row under another id) won the race.
        db.rollback()
        found = db.get(Organization, DEFAULT_ORGANIZATION_ID) or db.scalar(
            select(Organization).where(Organization.slug == "default")
        )
        if found is None:
            raise
        return found
    return org


# --- schemas -----------------------------------------------------------------------


class OrganizationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Name


class OrganizationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Name
    slug: str = Field(min_length=3, max_length=63)
    create_owner_key: bool = False

    @field_validator("slug")
    @classmethod
    def _valid_slug(cls, value: str) -> str:
        if not _SLUG_RE.fullmatch(value):
            raise ValueError(
                "slug must be lowercase letters, digits and hyphens, "
                "starting and ending with a letter or digit"
            )
        return value


class OrganizationStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["active", "suspended"]


class ApiKeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Name
    role: Literal["owner", "admin", "member"]
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def _future(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        if value <= datetime.now(timezone.utc):
            raise ValueError("expires_at must be in the future")
        return value


# --- helpers -----------------------------------------------------------------------


def _is_superadmin(ctx: AuthContext) -> bool:
    return ctx.principal in _SUPERADMIN_PRINCIPALS


def require_superadmin(ctx: AuthContext = Depends(require("member"))) -> AuthContext:
    if not _is_superadmin(ctx):
        raise HTTPException(status_code=403, detail={"code": "superadmin_required"})
    return ctx


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _org_json(org: Organization) -> dict[str, Any]:
    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "status": org.status,
        "created_at": _iso(org.created_at),
    }


def _key_json(key: ApiKey) -> dict[str, Any]:
    return {
        "id": key.id,
        "name": key.name,
        "role": key.role,
        "key_prefix": key.key_prefix,
        "created_by_key_id": key.created_by_key_id,
        "created_at": _iso(key.created_at),
        "last_used_at": _iso(key.last_used_at),
        "expires_at": _iso(key.expires_at),
        "revoked_at": _iso(key.revoked_at),
    }


def _can_grant(ctx: AuthContext, role: str) -> bool:
    """True when the caller may mint/revoke a key with ``role`` (never above their own)."""

    return _is_superadmin(ctx) or role_at_least(ctx.role, role)


def _mint_key(
    db: Session, *, org_id: str, name: str, role: str, created_by: str | None,
    expires_at: datetime | None = None,
) -> tuple[ApiKey, str]:
    plaintext, prefix, digest = generate_api_key()
    key = ApiKey(
        organization_id=org_id,
        name=name,
        key_prefix=prefix,
        key_hash=digest,
        role=role,
        created_by_key_id=created_by,
        expires_at=expires_at,
    )
    db.add(key)
    db.flush()
    return key, plaintext


def _get_org_or_404(db: Session, org_id: str) -> Organization:
    org = db.get(Organization, org_id)
    if org is None:
        raise HTTPException(status_code=404, detail={"code": "organization_not_found"})
    return org


# --- own organisation --------------------------------------------------------------


@router.get("/organization")
def get_organization(
    ctx: AuthContext = Depends(require("member")), db: Session = Depends(get_db)
) -> dict[str, Any]:
    return _org_json(_get_org_or_404(db, ctx.org_id))


@router.patch("/organization")
def update_organization(
    body: OrganizationUpdate,
    ctx: AuthContext = Depends(require("owner")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    org = _get_org_or_404(db, ctx.org_id)
    previous = org.name
    org.name = body.name
    audit.record(
        db, ctx, "organization.updated", resource_type="organization", resource_id=org.id,
        metadata={"previous_name": previous, "name": body.name},
    )
    db.commit()
    db.refresh(org)
    return _org_json(org)


# --- cross-organisation (superadmin) -----------------------------------------------


@router.post("/organizations", status_code=201)
def create_organization(
    body: OrganizationCreate,
    ctx: AuthContext = Depends(require_superadmin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    conflict = HTTPException(status_code=409, detail={"code": "slug_taken"})
    if db.scalar(select(Organization.id).where(Organization.slug == body.slug)) is not None:
        raise conflict
    org = Organization(name=body.name, slug=body.slug, status="active")
    db.add(org)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise conflict from None
    # Audit rows belong to the new organisation, not the superadmin's working org.
    org_ctx = dataclasses.replace(ctx, org_id=org.id)
    audit.record(
        db, org_ctx, "organization.created", resource_type="organization",
        resource_id=org.id,
        metadata={"name": org.name, "slug": org.slug, "owner_key": body.create_owner_key},
    )
    owner_key = None
    plaintext = None
    if body.create_owner_key:
        owner_key, plaintext = _mint_key(
            db, org_id=org.id, name="Initial owner key", role="owner", created_by=ctx.key_id
        )
        audit.record(
            db, org_ctx, "api_key.created", resource_type="api_key",
            resource_id=owner_key.id,
            metadata={"name": owner_key.name, "role": "owner", "key_prefix": owner_key.key_prefix},
        )
    db.commit()
    db.refresh(org)
    result = _org_json(org)
    if owner_key is not None:
        db.refresh(owner_key)
        result["owner_key"] = {**_key_json(owner_key), "key": plaintext}
    return result


@router.get("/organizations")
def list_organizations(
    ctx: AuthContext = Depends(require_superadmin), db: Session = Depends(get_db)
) -> dict[str, Any]:
    orgs = db.scalars(
        select(Organization).order_by(Organization.created_at, Organization.id)
    ).all()
    return {"items": [_org_json(org) for org in orgs]}


@router.post("/organizations/{organization_id}/status")
def set_organization_status(
    organization_id: str,
    body: OrganizationStatusUpdate,
    ctx: AuthContext = Depends(require_superadmin),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    org = _get_org_or_404(db, organization_id)
    previous = org.status
    org.status = body.status
    audit.record(
        db, dataclasses.replace(ctx, org_id=org.id), "organization.status_changed",
        resource_type="organization", resource_id=org.id,
        metadata={"previous_status": previous, "status": body.status},
    )
    db.commit()
    db.refresh(org)
    return _org_json(org)


# --- API keys ----------------------------------------------------------------------


@router.post("/api-keys", status_code=201)
def create_api_key(
    body: ApiKeyCreate,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not _can_grant(ctx, body.role):
        raise HTTPException(status_code=403, detail={"code": "role_exceeds_caller"})
    key, plaintext = _mint_key(
        db, org_id=ctx.org_id, name=body.name, role=body.role,
        created_by=ctx.key_id, expires_at=body.expires_at,
    )
    audit.record(
        db, ctx, "api_key.created", resource_type="api_key", resource_id=key.id,
        metadata={
            "name": key.name,
            "role": key.role,
            "key_prefix": key.key_prefix,
            "expires_at": _iso(key.expires_at),
        },
    )
    db.commit()
    db.refresh(key)
    return {**_key_json(key), "key": plaintext}


@router.get("/api-keys")
def list_api_keys(
    ctx: AuthContext = Depends(require("admin")), db: Session = Depends(get_db)
) -> dict[str, Any]:
    keys = db.scalars(
        select(ApiKey)
        .where(ApiKey.organization_id == ctx.org_id)
        .order_by(ApiKey.created_at.desc(), ApiKey.id)
    ).all()
    return {"items": [_key_json(key) for key in keys]}


@router.delete("/api-keys/{key_id}")
def revoke_api_key(
    key_id: str,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    key = db.scalar(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.organization_id == ctx.org_id)
    )
    if key is None:
        raise HTTPException(status_code=404, detail={"code": "api_key_not_found"})
    if not _can_grant(ctx, key.role):
        raise HTTPException(status_code=403, detail={"code": "role_exceeds_caller"})
    if key.revoked_at is None:  # idempotent: a repeat revoke changes and audits nothing
        key.revoked_at = datetime.now(timezone.utc)
        audit.record(
            db, ctx, "api_key.revoked", resource_type="api_key", resource_id=key.id,
            metadata={"name": key.name, "role": key.role, "key_prefix": key.key_prefix},
        )
        db.commit()
        db.refresh(key)
    return _key_json(key)

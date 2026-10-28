"""Platform API authentication and authorization (RM-004).

Admin API keys come from ``PLATFORM_API_KEYS`` (comma-separated). When the variable is
unset or empty, authentication is disabled so local development keeps working. When it
is set, a global dependency protects every route by default; the only exceptions are
listed explicitly below:

* ``PUBLIC_ROUTES`` need no credentials.
* ``RUNTIME_TOKEN_ROUTES`` are Runtime callbacks that carry their own ``X-Runtime-Token``.
* ``SESSION_SCOPED_ROUTES`` accept the admin key or a session token for that session.

Session tokens are random, returned once when the session (or a guest participant) is
created, and only their SHA-256 digests are stored.

Every authenticated request also gets an ``AuthContext`` on ``request.state.ctx``
(see ``context.py``). Credential resolution order for a non-public, non-runtime route:

1. ``PLATFORM_API_KEYS`` match -> ``superadmin`` (org: ``X-Organization-Id`` or default).
2. ``vlk_`` bearer -> organisation API key looked up by SHA-256 in ``api_keys``.
3. ``vls_`` bearer -> session token, scoped by ``SESSION_SCOPED_ROUTES``.
4. Anything else -> 401 ``invalid_api_key``.

With ``PLATFORM_API_KEYS`` unset auth is disabled (principal ``dev``, default org, owner),
but a presented ``vlk_`` key is still validated and honoured.
"""

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, HTTPException, Request
from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from .context import AuthContext
from .db import get_db
from .models import DEFAULT_ORGANIZATION_ID, ApiKey, Organization, VoiceSession

logger = logging.getLogger(__name__)

SESSION_TOKEN_PREFIX = "vls_"
API_KEY_PREFIX = "vlk_"
ORGANIZATION_HEADER = "x-organization-id"
LAST_USED_THROTTLE = timedelta(minutes=1)

PUBLIC_ROUTES = frozenset({("GET", "/health")})
RUNTIME_TOKEN_ROUTES = frozenset(
    {
        ("POST", "/runtime/events"),
        ("POST", "/runtime/agents/{agent_id}/knowledge/search"),
    }
)


@dataclass(frozen=True)
class SessionScope:
    """How a route maps to a session; ``creator_only`` excludes guest tokens."""

    param: str
    by: Literal["session", "conversation"]
    creator_only: bool = False


SESSION_SCOPED_ROUTES: dict[tuple[str, str], SessionScope] = {
    ("GET", "/voice/sessions/{session_id}"): SessionScope("session_id", "session"),
    # Only the creator ends a session; a guest token must not end it for everyone.
    ("POST", "/voice/sessions/{session_id}/end"): SessionScope(
        "session_id", "session", creator_only=True
    ),
    ("POST", "/voice/sessions/{session_id}/ensure-running"): SessionScope(
        "session_id", "session"
    ),
    ("POST", "/voice/sessions/{session_id}/participants"): SessionScope(
        "session_id", "session", creator_only=True
    ),
    ("POST", "/voice/sessions/{session_id}/phone-participants"): SessionScope(
        "session_id", "session", creator_only=True
    ),
    ("GET", "/conversations/{conversation_id}"): SessionScope(
        "conversation_id", "conversation"
    ),
    ("GET", "/conversations/{conversation_id}/messages"): SessionScope(
        "conversation_id", "conversation"
    ),
    ("GET", "/conversations/{conversation_id}/runtime-state"): SessionScope(
        "conversation_id", "conversation"
    ),
    # Guest "Report a problem" button (ST-02b); reading reports/debug stays admin-style.
    ("POST", "/conversations/{conversation_id}/reports"): SessionScope(
        "conversation_id", "conversation"
    ),
}


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_session_token() -> tuple[str, str]:
    """Return ``(token, sha256_hex_digest)``; only the digest is stored."""

    token = SESSION_TOKEN_PREFIX + secrets.token_urlsafe(32)
    return token, hash_token(token)


@lru_cache(maxsize=8)
def _digests_for(raw: str) -> tuple[str, ...]:
    return tuple(hash_token(key) for key in (p.strip() for p in raw.split(",")) if key)


def api_key_digests() -> tuple[str, ...]:
    """SHA-256 digests of the configured admin keys; empty means auth is disabled."""

    return _digests_for(os.getenv("PLATFORM_API_KEYS", ""))


def log_auth_status() -> None:
    if api_key_digests():
        logger.info("Platform API authentication enabled (%d key(s))", len(api_key_digests()))
    else:
        logger.warning(
            "PLATFORM_API_KEYS is not set: Platform API authentication is DISABLED. "
            "Set it before exposing the API beyond local development."
        )


def generate_api_key() -> tuple[str, str, str]:
    """Return ``(plaintext, display_prefix, sha256_hex_digest)`` for a new ``vlk_`` key.

    The plaintext is shown once; only the prefix and digest are stored.
    """

    plaintext = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return plaintext, plaintext[:12], hash_token(plaintext)


def _error(status_code: int, detail: Any) -> HTTPException:
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _matches_any(digest: str, candidates: list[str] | tuple[str, ...]) -> bool:
    matched = False
    for candidate in candidates:  # no early exit: constant work per candidate
        matched |= hmac.compare_digest(digest, candidate)
    return matched


def _session_token_match(
    db: Session, scope: SessionScope, value: str, digest: str
) -> tuple[VoiceSession, bool] | None:
    """Return ``(session, is_guest)`` when the token belongs to a matching session."""

    column = (
        VoiceSession.id if scope.by == "session" else VoiceSession.conversation_id
    )
    for voice_session in db.query(VoiceSession).filter(column == str(value)).all():
        creator_hash = voice_session.session_token_hash
        if creator_hash and hmac.compare_digest(digest, creator_hash):
            return voice_session, False
        if not scope.creator_only:
            if _matches_any(digest, list(voice_session.guest_token_hashes or [])):
                return voice_session, True
    return None


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _lookup_api_key(db: Session, digest: str) -> ApiKey | None:
    """The key for ``digest`` if usable; ``None`` if unknown, revoked, expired, or suspended."""

    key = db.query(ApiKey).filter(ApiKey.key_hash == digest).one_or_none()
    if key is None or key.revoked_at is not None:
        return None
    if key.expires_at is not None and _as_utc(key.expires_at) <= datetime.now(timezone.utc):
        return None
    org = db.get(Organization, key.organization_id)
    if org is None or org.status != "active":
        return None
    return key


def _touch_last_used(db: Session, key: ApiKey) -> None:
    """Set ``last_used_at`` at most once per minute (atomic, safe across workers)."""

    now = datetime.now(timezone.utc)
    cutoff = now - LAST_USED_THROTTLE
    result = db.execute(
        update(ApiKey)
        .where(
            ApiKey.id == key.id,
            or_(ApiKey.last_used_at.is_(None), ApiKey.last_used_at < cutoff),
        )
        .values(last_used_at=now)
        .execution_options(synchronize_session=False)
    )
    db.commit()
    if result.rowcount:
        db.expire(key)


def _superadmin_org(db: Session, request: Request) -> str:
    requested = request.headers.get(ORGANIZATION_HEADER, "").strip()
    if not requested:
        return DEFAULT_ORGANIZATION_ID
    if db.get(Organization, requested) is None:
        raise _error(404, {"code": "organization_not_found"})
    return requested


def _api_key_context(
    db: Session, request: Request, token: str, request_id: str
) -> AuthContext:
    key = _lookup_api_key(db, hash_token(token))
    if key is None:
        raise _error(401, "invalid_api_key")  # same response as an unknown key
    requested = request.headers.get(ORGANIZATION_HEADER, "").strip()
    if requested and requested != key.organization_id:
        raise _error(403, {"code": "organization_mismatch"})
    _touch_last_used(db, key)
    return AuthContext(
        org_id=key.organization_id,
        principal="api_key",
        role=key.role,
        key_id=key.id,
        request_id=request_id,
    )


async def authenticate(request: Request, db: Session = Depends(get_db)) -> None:
    """Global dependency enforcing the route policy and building ``request.state.ctx``."""

    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    route = request.scope.get("route")
    route_key = (request.method, getattr(route, "path", request.url.path))
    if route_key in PUBLIC_ROUTES:
        return
    if route_key in RUNTIME_TOKEN_ROUTES:
        # The route checks X-Runtime-Token itself and resolves the org from the session.
        request.state.ctx = AuthContext(
            org_id=None, principal="runtime", role=None, request_id=request_id
        )
        return

    digests = api_key_digests()
    header = request.headers.get("authorization")
    scheme, _, token = (header or "").partition(" ")
    token = token.strip()
    bearer = scheme.lower() == "bearer" and bool(token)

    if not digests:
        # Auth disabled for local development; an explicit org key is still honoured.
        if bearer and token.startswith(API_KEY_PREFIX):
            request.state.ctx = _api_key_context(db, request, token, request_id)
        else:
            request.state.ctx = AuthContext(
                org_id=DEFAULT_ORGANIZATION_ID,
                principal="dev",
                role="owner",
                request_id=request_id,
            )
        return

    if not header:
        raise _error(401, "authentication_required")
    if not bearer:
        raise _error(401, "invalid_api_key")
    digest = hash_token(token)
    if _matches_any(digest, digests):
        request.state.ctx = AuthContext(
            org_id=_superadmin_org(db, request),
            principal="superadmin",
            role="owner",
            request_id=request_id,
        )
        return

    if token.startswith(API_KEY_PREFIX):
        request.state.ctx = _api_key_context(db, request, token, request_id)
        return

    if token.startswith(SESSION_TOKEN_PREFIX):
        # A session token never grants admin access, and only works for its own session.
        scope = SESSION_SCOPED_ROUTES.get(route_key)
        match = (
            _session_token_match(db, scope, request.path_params.get(scope.param, ""), digest)
            if scope is not None
            else None
        )
        if match is None:
            raise _error(403, "forbidden")
        voice_session, is_guest = match
        session_org_id = getattr(voice_session, "organization_id", None)
        if session_org_id:
            org = db.get(Organization, session_org_id)
            if org is not None and org.status != "active":
                # Suspending an org also stops its session tokens; same generic error as
                # any other bad token. (Runtime ingest for a suspended org is left as-is:
                # calls already running are allowed to finish.)
                raise _error(401, "invalid_api_key")
        request.state.ctx = AuthContext(
            org_id=getattr(voice_session, "organization_id", None)
            or DEFAULT_ORGANIZATION_ID,
            principal="session",
            role=None,
            session_id=voice_session.id,
            is_guest=is_guest,
            request_id=request_id,
        )
        return
    raise _error(401, "invalid_api_key")

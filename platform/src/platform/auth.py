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
"""

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import get_db
from .models import VoiceSession

logger = logging.getLogger(__name__)

SESSION_TOKEN_PREFIX = "vls_"

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
    ("POST", "/voice/sessions/{session_id}/end"): SessionScope("session_id", "session"),
    ("POST", "/voice/sessions/{session_id}/ensure-running"): SessionScope(
        "session_id", "session"
    ),
    ("POST", "/voice/sessions/{session_id}/participants"): SessionScope(
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


def _error(status_code: int, detail: str) -> HTTPException:
    headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
    return HTTPException(status_code=status_code, detail=detail, headers=headers)


def _matches_any(digest: str, candidates: list[str] | tuple[str, ...]) -> bool:
    matched = False
    for candidate in candidates:  # no early exit: constant work per candidate
        matched |= hmac.compare_digest(digest, candidate)
    return matched


def _session_token_allowed(
    db: Session, scope: SessionScope, value: str, digest: str
) -> bool:
    query = db.query(VoiceSession.session_token_hash, VoiceSession.guest_token_hashes)
    column = (
        VoiceSession.id if scope.by == "session" else VoiceSession.conversation_id
    )
    for creator_hash, guest_hashes in query.filter(column == str(value)).all():
        candidates = [creator_hash] if creator_hash else []
        if not scope.creator_only:
            candidates.extend(guest_hashes or [])
        if _matches_any(digest, candidates):
            return True
    return False


async def authenticate(request: Request, db: Session = Depends(get_db)) -> None:
    """Global dependency enforcing the route policy when API keys are configured."""

    digests = api_key_digests()
    route = request.scope.get("route")
    route_key = (request.method, getattr(route, "path", request.url.path))
    if not digests or route_key in PUBLIC_ROUTES or route_key in RUNTIME_TOKEN_ROUTES:
        return

    header = request.headers.get("authorization")
    if not header:
        raise _error(401, "authentication_required")
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise _error(401, "invalid_api_key")
    digest = hash_token(token)
    if _matches_any(digest, digests):
        return

    if token.startswith(SESSION_TOKEN_PREFIX):
        # A session token never grants admin access, and only works for its own session.
        scope = SESSION_SCOPED_ROUTES.get(route_key)
        if scope is not None and _session_token_allowed(
            db, scope, request.path_params.get(scope.param, ""), digest
        ):
            return
        raise _error(403, "forbidden")
    raise _error(401, "invalid_api_key")

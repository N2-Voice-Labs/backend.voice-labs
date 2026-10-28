"""Per-request authorization context and role guards (organisations plan, T3).

``auth.authenticate`` builds an :class:`AuthContext` for every non-public request and
stores it on ``request.state.ctx``. Route handlers read it with :func:`get_ctx` and gate
on role with :func:`require`.

``org_id`` is ``None`` only for the ``runtime`` principal: Runtime callbacks carry a
shared ``X-Runtime-Token`` and no tenant, so the routes resolve the organisation later
from the VoiceSession (never from the caller).
"""

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, Request

from .models import ORGANIZATION_ROLES

Principal = Literal["superadmin", "api_key", "user", "session", "runtime", "system", "dev"]

# Principals that bypass role checks (trusted operators / internal callers).
_ALWAYS_PASS = frozenset({"superadmin", "dev", "system"})
# Principals that are never granted an organisation role.
_NEVER_PASS = frozenset({"session", "runtime"})


@dataclass(frozen=True)
class AuthContext:
    org_id: str | None
    principal: Principal
    role: str | None
    key_id: str | None = None
    session_id: str | None = None
    is_guest: bool = False
    request_id: str | None = None


def role_at_least(role: str | None, minimum: str) -> bool:
    """True when ``role`` ranks at or above ``minimum`` (owner > admin > member)."""

    if minimum not in ORGANIZATION_ROLES:
        raise ValueError(f"unknown role: {minimum!r}")
    if role not in ORGANIZATION_ROLES:
        return False
    # ORGANIZATION_ROLES is ordered most to least privileged.
    return ORGANIZATION_ROLES.index(role) <= ORGANIZATION_ROLES.index(minimum)


def get_ctx(request: Request) -> AuthContext:
    ctx = getattr(request.state, "ctx", None)
    if ctx is None:
        # Public routes never get a context; reaching here means a misconfigured route.
        raise HTTPException(
            status_code=401,
            detail="authentication_required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ctx


def require(minimum_role: str):
    """Dependency factory: 403 ``insufficient_role`` unless the caller has the role."""

    if minimum_role not in ORGANIZATION_ROLES:
        raise ValueError(f"unknown role: {minimum_role!r}")

    def dependency(request: Request) -> AuthContext:
        ctx = get_ctx(request)
        if ctx.principal in _NEVER_PASS:
            allowed = False
        elif ctx.principal in _ALWAYS_PASS:
            allowed = True
        else:
            allowed = role_at_least(ctx.role, minimum_role)
        if not allowed:
            raise HTTPException(status_code=403, detail={"code": "insufficient_role"})
        return ctx

    return dependency

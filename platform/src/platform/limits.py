"""Admission limits for new voice sessions (ST-13).

* Per-organisation cap on concurrently active sessions
  (``PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG``, default 10; ``0`` disables) ->
  429 ``too_many_active_sessions``. Checked in ``sessions.open_voice_session`` before any
  row is created, so it also covers the SIP bridge and campaign dispatcher. The count and
  the insert are not one atomic step: concurrent requests can overshoot the cap by the
  number of requests in flight.
* Per-organisation bound on NEW hidden translation configurations per rolling 24 h
  (``PLATFORM_MAX_TRANSLATION_CONFIGS``, default 50; ``0`` disables) -> 429
  ``too_many_translation_configs`` when creating one more would exceed it (only system
  agents created in the last 24 h count; reusing an existing one always works; checked in ``translations.system_agent_for``). Like the session
  cap, concurrent creators can overshoot by the number of requests in flight.
* Token bucket per API key (or per principal+org when there is no key) on
  ``POST /voice/sessions`` (``PLATFORM_SESSION_CREATE_RATE``, default ``30/min``; ``0``
  disables) -> 429 ``rate_limited`` with a ``Retry-After`` header. **The buckets live in
  process memory**: each worker process counts on its own, so with N workers the
  effective limit is up to N times the configured rate, and a restart resets them. Use an
  edge/gateway limiter when a hard global limit is needed.

0 disables a limit (tests/conftest.py does this). Both read their environment on every call so tests and operators can change them live.
"""

from __future__ import annotations

import math
import os
import threading
import time

from fastapi import Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .context import AuthContext, get_ctx
from .models import VoiceSession

# Statuses of a VoiceSession that still holds (or may hold) a Runtime slot.
ACTIVE_SESSION_STATUSES: tuple[str, ...] = ("created", "running")

DEFAULT_MAX_ACTIVE_SESSIONS_PER_ORG = 10
DEFAULT_SESSION_CREATE_RATE = 30  # per minute
# Hidden translation (system) agents: one row per distinct interpreter configuration.
NEW_TRANSLATION_CONFIGS_PER_DAY = 50
TRANSLATION_CONFIG_WINDOW_SECONDS = 24 * 3600


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    # Tolerate "30/min" style values.
    raw = raw.split("/", 1)[0].strip()
    try:
        return max(int(raw), 0)
    except ValueError:
        return default


def max_active_sessions_per_org() -> int:
    return _env_int("PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG", DEFAULT_MAX_ACTIVE_SESSIONS_PER_ORG)


def max_new_translation_configs_per_day() -> int:
    """Per-org bound on new hidden translation configs per rolling 24 h
    (``PLATFORM_MAX_TRANSLATION_CONFIGS``; 0 disables)."""

    return _env_int("PLATFORM_MAX_TRANSLATION_CONFIGS", NEW_TRANSLATION_CONFIGS_PER_DAY)


def session_create_rate() -> int:
    return _env_int("PLATFORM_SESSION_CREATE_RATE", DEFAULT_SESSION_CREATE_RATE)


def count_active_sessions(db: Session, org_id: str) -> int:
    return db.scalar(
        select(func.count())
        .select_from(VoiceSession)
        .where(
            VoiceSession.organization_id == org_id,
            VoiceSession.status.in_(ACTIVE_SESSION_STATUSES),
        )
    ) or 0


def check_active_session_cap(db: Session, ctx: AuthContext) -> None:
    """429 ``too_many_active_sessions`` when the org is at its concurrent-session cap."""

    cap = max_active_sessions_per_org()
    if cap <= 0 or ctx.org_id is None:
        return
    if count_active_sessions(db, ctx.org_id) >= cap:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "too_many_active_sessions",
                "message": (
                    f"This organization already has {cap} active voice sessions. "
                    "End a session or wait for one to finish."
                ),
            },
        )


class TokenBucket:
    """Thread-safe token buckets keyed by string; capacity = burst = ``per_minute``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, updated)

    def take(self, key: str, per_minute: int, now: float | None = None) -> float:
        """Spend one token; returns 0.0 when allowed, else seconds until one is available."""

        now = time.monotonic() if now is None else now
        refill_per_sec = per_minute / 60.0
        with self._lock:
            tokens, updated = self._state.get(key, (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - updated) * refill_per_sec)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return 0.0
            self._state[key] = (tokens, now)
            return (1.0 - tokens) / refill_per_sec

    def reset(self) -> None:
        with self._lock:
            self._state.clear()


session_create_bucket = TokenBucket()


def rate_limit_key(ctx: AuthContext) -> str:
    return ctx.key_id or f"{ctx.principal}:{ctx.org_id}"


def limit_session_creation(ctx: AuthContext = Depends(get_ctx)) -> None:
    """Dependency for ``POST /voice/sessions``: 429 ``rate_limited`` + ``Retry-After``."""

    rate = session_create_rate()
    if rate <= 0:
        return
    wait = session_create_bucket.take(rate_limit_key(ctx), rate)
    if wait > 0:
        retry_after = max(1, math.ceil(wait))
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limited",
                "message": f"Too many session requests. Retry in {retry_after}s.",
            },
            headers={"Retry-After": str(retry_after)},
        )

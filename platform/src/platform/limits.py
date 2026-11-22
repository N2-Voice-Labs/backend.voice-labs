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
  edge/gateway limiter when a hard global limit is needed. Buckets that have refilled to
  full are dropped (a full bucket and a missing one behave the same) once the table grows
  past ``TokenBucket.max_keys`` entries, so the table does not grow without bound.
* Per-organisation bucket on outbound phone dials (``PLATFORM_DIAL_PER_ORG_HOUR``, default
  30; ``0`` disables) -> 429 ``rate_limited`` with ``Retry-After``. Charged once per
  ``POST /voice/sessions/{id}/phone-participants`` or ``POST /deployments/{id}/bridge-call``
  request that gets as far as dialling.

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
DEFAULT_DIAL_PER_ORG_HOUR = 30
DEFAULT_BUCKET_MAX_KEYS = 10_000
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


def dial_per_org_hour() -> int:
    return _env_int("PLATFORM_DIAL_PER_ORG_HOUR", DEFAULT_DIAL_PER_ORG_HOUR)


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
    """Thread-safe token buckets keyed by string; capacity = burst = ``per_minute``.

    Once more than ``max_keys`` keys are held, buckets that have refilled to full are
    dropped: that loses nothing (a missing key starts full) and bounds memory when keys
    are attacker-chosen, such as client addresses.
    """

    def __init__(self, max_keys: int = DEFAULT_BUCKET_MAX_KEYS) -> None:
        self._lock = threading.Lock()
        self._max_keys = max_keys
        self._prune_at = max_keys
        # key -> (tokens, updated, per_minute)
        self._state: dict[str, tuple[float, float, int]] = {}

    def take(self, key: str, per_minute: int, now: float | None = None) -> float:
        """Spend one token; returns 0.0 when allowed, else seconds until one is available."""

        now = time.monotonic() if now is None else now
        refill_per_sec = per_minute / 60.0
        with self._lock:
            if key not in self._state and len(self._state) >= self._prune_at:
                self._prune(now)
            tokens, updated, _ = self._state.get(key, (float(per_minute), now, per_minute))
            tokens = min(float(per_minute), tokens + (now - updated) * refill_per_sec)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now, per_minute)
                return 0.0
            self._state[key] = (tokens, now, per_minute)
            return (1.0 - tokens) / refill_per_sec

    def _prune(self, now: float) -> None:
        """Drop buckets that would be full by ``now`` (caller holds the lock)."""

        for key, (tokens, updated, per_minute) in list(self._state.items()):
            if tokens + (now - updated) * per_minute / 60.0 >= per_minute:
                del self._state[key]
        # Grow geometrically while everything is still live, so a flood of distinct
        # keys costs amortised O(1) per request instead of a scan each time.
        self._prune_at = max(self._max_keys, 2 * len(self._state))

    def reset(self) -> None:
        with self._lock:
            self._state.clear()
            self._prune_at = self._max_keys


session_create_bucket = TokenBucket()
dial_bucket = TokenBucket()


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


def check_dial_rate(ctx: AuthContext) -> None:
    """Charge one outbound dial to the organisation: 429 ``rate_limited`` + ``Retry-After``.

    Called by the routes right before they dial, so refused or invalid requests cost nothing.
    """

    per_hour = dial_per_org_hour()
    if per_hour <= 0:
        return
    # Capacity is per_hour tokens refilling at per_hour/hour: the bucket's clock runs in
    # minutes (``take`` divides its rate by 60), as in ``calls.limit_create``.
    wait = dial_bucket.take(str(ctx.org_id), per_hour, now=time.monotonic() / 60.0)
    if wait > 0:
        retry_after = max(1, math.ceil(wait * 60.0))
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limited",
                "message": f"Too many outbound calls for this organization. Retry in {retry_after}s.",
            },
            headers={"Retry-After": str(retry_after)},
        )

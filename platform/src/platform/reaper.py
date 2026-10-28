"""Stale voice session reaper (ST-13).

A session whose Runtime died (or never started) would otherwise stay ``created`` /
``running`` forever, holding a slot against the per-organisation cap. The reaper marks
such sessions ``failed`` with end reason ``stale`` when the newest of ``started_at`` (set
at creation) and ``last_event_at`` (set by every accepted Runtime event) is older than
``PLATFORM_STALE_SESSION_SECS`` (default 300).

``reap_stale_sessions`` is one pass and runs across all organisations (tenant filtering
bypassed). ``reaper_loop`` repeats it every ``PLATFORM_REAPER_INTERVAL_SECS`` (default 60;
``0`` disables; tests/conftest.py does).
Runs in every Platform process; passes are idempotent, so several workers are harmless.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .limits import ACTIVE_SESSION_STATUSES
from .models import Conversation, RuntimeEvent, VoiceSession, utcnow
from .tenancy import bypass

logger = logging.getLogger("platform.reaper")

STALE_REASON = "stale"
DEFAULT_INTERVAL_SECS = 60
DEFAULT_STALE_SECS = 300


def reaper_interval() -> int:
    """Seconds between passes; 0 means disabled."""

    raw = os.getenv("PLATFORM_REAPER_INTERVAL_SECS", "").strip()
    if not raw:
        return DEFAULT_INTERVAL_SECS
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_INTERVAL_SECS


def stale_after_secs() -> int:
    try:
        return max(int(os.getenv("PLATFORM_STALE_SESSION_SECS", "")), 1)
    except ValueError:
        return DEFAULT_STALE_SECS


def _aware(value: datetime) -> datetime:
    # SQLite returns naive datetimes; they were stored as UTC.
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def reap_stale_sessions(db: Session, now: datetime | None = None) -> list[str]:
    """Fail every active session with no activity for the stale window; returns their ids.

    Also fails the conversation and records a ``SessionFailed`` event with reason
    ``stale`` (what the debug view shows as the end reason). Each session is reaped in
    its own transaction with a *conditional* UPDATE (still active and with unchanged
    ``last_event_at``), so a session that ended or received an event meanwhile is left
    alone, and a concurrent worker or a lost race never aborts the rest of the pass.
    """

    now = _aware(now) if now is not None else utcnow()
    cutoff = now - timedelta(seconds=stale_after_secs())
    reaped: list[str] = []
    with bypass(db):
        rows = db.execute(
            select(
                VoiceSession.id,
                VoiceSession.organization_id,
                VoiceSession.conversation_id,
                VoiceSession.started_at,
                VoiceSession.last_event_at,
            ).where(VoiceSession.status.in_(ACTIVE_SESSION_STATUSES))
        ).all()
        db.rollback()  # end the read transaction; each session gets its own below
        for session_id, org_id, conversation_id, started_at, last_event_at in rows:
            seen = _aware(started_at)
            if last_event_at is not None:
                seen = max(seen, _aware(last_event_at))
            if seen >= cutoff:
                continue
            try:
                if _reap_one(
                    db, now, session_id, org_id, conversation_id, last_event_at
                ):
                    reaped.append(session_id)
            except Exception:
                db.rollback()
                logger.exception("failed to reap stale session %s", session_id)
    return reaped


def _reap_one(
    db: Session,
    now: datetime,
    session_id: str,
    org_id: str,
    conversation_id: str,
    observed_last_event_at: datetime | None,
) -> bool:
    """Fail one session if it is still active and unchanged since it was judged stale."""

    last_event_guard = (
        VoiceSession.last_event_at.is_(None)
        if observed_last_event_at is None
        else VoiceSession.last_event_at == observed_last_event_at
    )
    result = db.execute(
        update(VoiceSession)
        .where(
            VoiceSession.id == session_id,
            VoiceSession.status.in_(ACTIVE_SESSION_STATUSES),
            last_event_guard,
        )
        .values(status="failed", ended_at=now)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:  # ended, failed or got an event meanwhile
        db.rollback()
        return False
    agent_id = db.scalar(
        select(Conversation.agent_id).where(Conversation.id == conversation_id)
    )
    db.execute(
        update(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.status.not_in(("ended", "failed")),
        )
        .values(status="failed", ended_at=now)
        .execution_options(synchronize_session=False)
    )
    try:
        with db.begin_nested():  # an event another worker already wrote is not an error
            db.add(
                RuntimeEvent(
                    organization_id=org_id,
                    event_id=f"platform-stale-{session_id}",
                    event_type="SessionFailed",
                    session_id=session_id,
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    payload={"reason": STALE_REASON, "source": "platform_reaper"},
                )
            )
    except IntegrityError:
        pass
    db.commit()
    return True


async def _end_in_runtime(session_ids: list[str]) -> None:
    """Best-effort: tell the Runtime to drop the sessions. Failures are ignored."""

    runtime_url = os.getenv("RUNTIME_API_URL")
    runtime_token = os.getenv("PLATFORM_RUNTIME_TOKEN", "")
    if not runtime_url or not runtime_token or not session_ids:
        return
    async with httpx.AsyncClient(timeout=5.0) as client:
        for session_id in session_ids:
            try:
                await client.post(
                    f"{runtime_url.rstrip('/')}/sessions/{session_id}/end",
                    headers={"X-Runtime-Token": runtime_token},
                )
            except httpx.HTTPError:
                continue


async def reap_once() -> list[str]:
    """One pass with its own DB session, then the best-effort Runtime end calls."""

    from . import db as _db  # late: tests rebind ``SessionLocal``

    with _db.SessionLocal() as db:
        reaped = await asyncio.to_thread(reap_stale_sessions, db)
    if reaped:
        logger.warning("reaped %d stale voice session(s): %s", len(reaped), reaped)
        await _end_in_runtime(reaped)
    return reaped


async def reaper_loop(interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await reap_once()
        except asyncio.CancelledError:
            raise
        except Exception:  # keep the loop alive whatever one pass hits
            logger.exception("stale session reaper pass failed")


def start_reaper() -> asyncio.Task | None:
    interval = reaper_interval()
    if interval <= 0:
        return None
    return asyncio.create_task(reaper_loop(interval), name="session-reaper")


async def stop_reaper(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

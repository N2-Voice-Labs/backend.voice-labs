"""Provider health (ST-00): runtime provider status, session preflight and alerting.

``GET /health/providers`` proxies Runtime's ``GET /health/providers`` (30 s cache), adds the
database state and reports ``runtime: not_configured`` when ``RUNTIME_API_URL`` is unset.

``create_session`` calls :func:`required_providers` and :func:`assert_providers_ready` before
it persists anything: a required provider that is ``down`` (rejected key, no credit) refuses
the session with 503 ``provider_unavailable`` instead of creating a mute call. ``degraded``
never blocks, and an unreachable health endpoint never blocks (the runtime handoff failure
path still covers a dead Runtime).

``PLATFORM_ALERT_WEBHOOK_URL`` (Discord/Slack-compatible ``{"content": ...}``) receives one
message when a provider becomes degraded/down or recovers, at most one per provider per
5 minutes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .context import AuthContext, get_ctx, role_at_least
from .db import get_db

_logger = logging.getLogger(__name__)

router = APIRouter()

CACHE_TTL_SECONDS = 30.0
# Forced refreshes run paid live probes: admins only, at most once per interval per process.
FORCED_REFRESH_MIN_INTERVAL_SECONDS = 10.0
FETCH_TIMEOUT_SECONDS = 5.0
ALERT_DEBOUNCE_SECONDS = 300.0
DEFAULT_PROVIDERS = {"stt": "deepgram", "llm": "openai", "tts": "cartesia"}
PROVIDER_ALIASES = {"openai-compatible": "openai"}  # same key, same health entry

_cache: dict[str, Any] = {"at": None, "report": None}
_last_forced_at: float = float("-inf")
_alert_state: dict[tuple[str, str], str] = {}
_alert_sent_at: dict[tuple[str, str], float] = {}
# Tests inject an ``httpx.MockTransport`` (runtime fetch and webhook).
_transport: httpx.AsyncBaseTransport | None = None


def _client_kwargs() -> dict[str, Any]:
    return {"transport": _transport} if _transport is not None else {}


def reset() -> None:
    """Forget the cache and alert state (tests)."""

    global _last_forced_at
    _cache.update(at=None, report=None)
    _last_forced_at = float("-inf")
    _alert_state.clear()
    _alert_sent_at.clear()


def runtime_unavailable_detail(message: str = "The voice runtime is unavailable.") -> dict[str, str]:
    """The structured 503 body replacing the bare ``"runtime_unavailable"`` string."""

    return {"code": "runtime_unavailable", "message": message}


def runtime_unavailable(exc_message: str | None = None) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=runtime_unavailable_detail(
            exc_message or "The voice runtime is unavailable. Try again shortly."
        ),
    )


# --- fetching -----------------------------------------------------------------------


def _valid_report(body: Any) -> bool:
    return (
        isinstance(body, dict)
        and isinstance(body.get("providers"), list)
        and all(isinstance(p, dict) for p in body["providers"])
    )


async def fetch_runtime_health(*, force: bool = False) -> dict[str, Any] | None:
    """Runtime's provider report (cached 30 s); ``None`` when Runtime is not configured.

    Raises on a transport/HTTP/shape error (callers decide whether that blocks).
    """

    runtime_url = os.getenv("RUNTIME_API_URL")
    if not runtime_url:
        return None
    now = time.monotonic()
    if not force and _cache["at"] is not None and now - _cache["at"] < CACHE_TTL_SECONDS:
        return _cache["report"]
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, **_client_kwargs()) as client:
        response = await client.get(
            f"{runtime_url.rstrip('/')}/health/providers",
            headers={"X-Runtime-Token": os.getenv("PLATFORM_RUNTIME_TOKEN", "")},
            params={"refresh": "true"} if force else None,
        )
        response.raise_for_status()
        report = response.json()
    if not _valid_report(report):
        raise ValueError("runtime provider health response has an unexpected shape")
    _cache.update(at=time.monotonic(), report=report)
    await notify_changes(report["providers"])
    return report


# --- preflight ----------------------------------------------------------------------


def required_providers(
    mode: str, interpreter: dict[str, Any] | None, providers: dict[str, str] | None
) -> list[tuple[str, str]]:
    """The ``(kind, name)`` pairs a session needs: stt/llm/tts for the cascaded pipeline
    (agent and cascaded interpreter), the realtime OpenAI engine for realtime interpreters.

    ``providers`` is the per-kind selection from the agent's provider records; an absent kind
    uses Runtime's default (deepgram / openai / cartesia).
    """

    if mode == "interpreter" and (interpreter or {}).get("engine") == "realtime":
        return [("realtime", "openai")]
    chosen = providers or {}
    return [
        (kind, PROVIDER_ALIASES.get(name, name))
        for kind, name in ((k, chosen.get(k) or DEFAULT_PROVIDERS[k]) for k in DEFAULT_PROVIDERS)
    ]


def assert_providers_ready(
    report: dict[str, Any] | None, required: list[tuple[str, str]]
) -> None:
    """503 ``provider_unavailable`` when a required provider is ``down`` in ``report``."""

    if not report:
        return
    entries = {(p.get("kind"), p.get("name")): p for p in report.get("providers", [])}
    for kind, name in required:
        entry = entries.get((kind, name))
        if entry is None or entry.get("status") != "down":
            continue
        reason = entry.get("reason") or "unavailable"
        raise HTTPException(
            status_code=503,
            detail={
                "code": "provider_unavailable",
                "provider": name,
                "kind": kind,
                "category": entry.get("category") or "unavailable",
                "message": f"Voice service unavailable: {name} {reason}",
            },
        )


async def preflight_providers(required: list[tuple[str, str]]) -> None:
    """Refuse (503) a session whose required provider is down; never block on a failed check."""

    try:
        report = await fetch_runtime_health()
    except Exception as exc:  # noqa: BLE001 - health must never take sessions down
        _logger.warning("provider health unavailable, not blocking: %s", type(exc).__name__)
        return
    assert_providers_ready(report, required)


# --- alerting -----------------------------------------------------------------------


async def notify_changes(providers: list[dict[str, Any]]) -> None:
    """Post a webhook message for providers whose status changed (debounced per provider)."""

    url = os.getenv("PLATFORM_ALERT_WEBHOOK_URL")
    now = time.monotonic()
    for entry in providers:
        key = (str(entry.get("kind")), str(entry.get("name")))
        status = str(entry.get("status", "ok"))
        previous = _alert_state.get(key, "ok")
        _alert_state[key] = status
        if status == previous or not url:
            continue
        sent = _alert_sent_at.get(key)
        if sent is not None and now - sent < ALERT_DEBOUNCE_SECONDS:
            continue
        if status == "ok":
            content = f"Voice Labs: {key[0]}/{key[1]} recovered"
        else:
            detail = entry.get("reason") or entry.get("category") or ""
            content = f"Voice Labs: {key[0]}/{key[1]} is {status}" + (
                f" ({detail})" if detail else ""
            )
        _alert_sent_at[key] = now
        try:
            async with httpx.AsyncClient(timeout=5.0, **_client_kwargs()) as client:
                (await client.post(url, json={"content": content})).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("provider alert webhook failed: %s", type(exc).__name__)


# --- route --------------------------------------------------------------------------


@router.get("/health/providers")
async def provider_health(
    refresh: bool = False,
    db: Session = Depends(get_db),
    ctx: AuthContext = Depends(get_ctx),
) -> dict[str, Any]:
    global _last_forced_at
    force = False
    if refresh and (ctx.principal in {"superadmin", "dev", "system"} or role_at_least(ctx.role, "admin")):
        now = time.monotonic()
        if now - _last_forced_at >= FORCED_REFRESH_MIN_INTERVAL_SECONDS:
            _last_forced_at = now
            force = True
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError:
        database = "down"
    providers: list[dict[str, Any]] = []
    runtime_overall = "unknown"
    try:
        report = await fetch_runtime_health(force=force)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("runtime provider health fetch failed: %s", type(exc).__name__)
        runtime = "unreachable"
    else:
        if report is None:
            runtime = "not_configured"
        else:
            runtime = "ok"
            providers = report["providers"]
            runtime_overall = str(report.get("overall", "unknown"))
    states = {runtime_overall, "down" if database == "down" or runtime == "unreachable" else "ok"}
    overall = "down" if "down" in states else "degraded" if "degraded" in states else "ok"
    return {
        "overall": overall,
        "database": database,
        "runtime": runtime,
        "providers": providers,
    }

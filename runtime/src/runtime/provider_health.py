"""Provider health (ST-00): cheap auth/quota probes for every configured provider.

A provider with a rejected key, no credit or an outage used to be found only when a caller
hit a mute session. This module probes each configured provider concurrently with a read-only
request (no synthesis, no tokens), keeps the result for :data:`CACHE_TTL_SECONDS`, and lets
the live failure path degrade it immediately (:func:`record_provider_failure`; only a
probe can mark a provider ``down``).

Probes (3 s timeout each, all concurrent):

* ``stt/deepgram``    ``GET https://api.deepgram.com/v1/projects``
* ``llm/openai``      ``GET {OPENAI_BASE_URL or https://api.openai.com/v1}/models/{model}``
  (also reported as ``realtime/openai``: the realtime engine uses the same key)
* ``tts/cartesia``    ``GET https://api.cartesia.ai/voices?limit=1``
* ``tts/elevenlabs``  ``GET https://api.elevenlabs.io/v1/user/subscription``

A provider is probed only when its API key is set in the environment.

Status/category mirror ``provider_failures.classify_error``: 401/403 ``auth`` and 402 /
``insufficient_quota`` ``quota`` are ``down``; 408/425/429/5xx and network errors are
``degraded`` (``unavailable``), a timeout is ``degraded`` (``timeout``). Reasons are fixed
phrases: never a key and never a provider response body.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import httpx

from .speech_providers import provider_names
from .tts_preview import CARTESIA_BASE_URL, CARTESIA_VERSION

_logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 3.0
CACHE_TTL_SECONDS = 60.0
REFRESH_INTERVAL_SECONDS = 45.0
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_HEALTH_MODEL = "gpt-4.1-mini"
DEEPGRAM_PROJECTS_URL = "https://api.deepgram.com/v1/projects"
ELEVENLABS_SUBSCRIPTION_URL = "https://api.elevenlabs.io/v1/user/subscription"

OK = "ok"
DEGRADED = "degraded"
DOWN = "down"
CATEGORIES = frozenset({"auth", "quota", "unavailable", "timeout", "other"})

_REASONS = {
    "auth": "credentials were rejected",
    "quota": "out of credit or quota",
    "unavailable": "service unavailable",
    "timeout": "health check timed out",
    "other": "unexpected response",
}
_QUOTA_BODY = re.compile(r"insufficient[_ ]quota|quota[_ ]exceeded|out of credit", re.I)


@dataclasses.dataclass
class ProviderStatus:
    kind: str
    name: str
    status: str = OK
    category: str | None = None
    latency_ms: int | None = None
    checked_at: str = ""
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class _Outcome:
    status: str
    category: str | None = None
    reason: str | None = None


def _outcome_for_status(code: int, body: str = "") -> _Outcome:
    if 200 <= code < 300:
        return _Outcome(OK)
    if code in (401, 403):
        return _Outcome(DOWN, "auth", _REASONS["auth"])
    if code == 402 or (code in (429, 400) and _QUOTA_BODY.search(body)):
        return _Outcome(DOWN, "quota", _REASONS["quota"])
    if code in (408, 425, 429) or 500 <= code < 600:
        reason = "rate limited" if code == 429 else _REASONS["unavailable"]
        return _Outcome(DEGRADED, "unavailable", reason)
    return _Outcome(DEGRADED, "other", f"unexpected response (HTTP {code})")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- probes -------------------------------------------------------------------------

Inspect = Callable[[httpx.Response], _Outcome | None]


async def _get(
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str],
    *,
    params: Mapping[str, Any] | None = None,
    inspect: Inspect | None = None,
) -> _Outcome:
    try:
        response = await client.get(url, headers=dict(headers), params=params)
    except httpx.TimeoutException:
        return _Outcome(DEGRADED, "timeout", _REASONS["timeout"])
    except httpx.HTTPError:
        return _Outcome(DEGRADED, "unavailable", "could not reach the service")
    outcome = _outcome_for_status(
        response.status_code, response.text[:2000] if response.status_code >= 400 else ""
    )
    if outcome.status == OK and inspect is not None:
        return inspect(response) or outcome
    return outcome


def _elevenlabs_quota(response: httpx.Response) -> _Outcome | None:
    try:
        data = response.json()
        used, limit = data["character_count"], data["character_limit"]
    except (ValueError, KeyError, TypeError):
        return None
    if isinstance(used, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
        if used >= limit:
            return _Outcome(DOWN, "quota", _REASONS["quota"])
    return None


@dataclasses.dataclass(frozen=True)
class _Probe:
    kind: str
    name: str
    key_env: str
    run: Callable[[httpx.AsyncClient, str], Awaitable[_Outcome]]
    mirror_kinds: tuple[str, ...] = ()  # same result reported under these kinds too


async def _probe_deepgram(client: httpx.AsyncClient, key: str) -> _Outcome:
    return await _get(client, DEEPGRAM_PROJECTS_URL, {"Authorization": f"Token {key}"})


async def _probe_openai(client: httpx.AsyncClient, key: str) -> _Outcome:
    base = (os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    model = os.getenv("OPENAI_HEALTH_MODEL") or DEFAULT_OPENAI_HEALTH_MODEL
    return await _get(
        client, f"{base}/models/{model}", {"Authorization": f"Bearer {key}"}
    )


async def _probe_cartesia(client: httpx.AsyncClient, key: str) -> _Outcome:
    return await _get(
        client,
        f"{CARTESIA_BASE_URL}/voices",
        {"X-API-Key": key, "Cartesia-Version": CARTESIA_VERSION},
        params={"limit": 1},
    )


async def _probe_elevenlabs(client: httpx.AsyncClient, key: str) -> _Outcome:
    return await _get(
        client, ELEVENLABS_SUBSCRIPTION_URL, {"xi-api-key": key}, inspect=_elevenlabs_quota
    )


_PROBES: tuple[_Probe, ...] = (
    _Probe("stt", "deepgram", "DEEPGRAM_API_KEY", _probe_deepgram),
    _Probe("llm", "openai", "OPENAI_API_KEY", _probe_openai, mirror_kinds=("realtime",)),
    _Probe("tts", "cartesia", "CARTESIA_API_KEY", _probe_cartesia),
    _Probe("tts", "elevenlabs", "ELEVENLABS_API_KEY", _probe_elevenlabs),
)


async def _run_probe(
    probe: _Probe, client: httpx.AsyncClient, key: str
) -> list[ProviderStatus]:
    started = time.monotonic()
    try:
        outcome = await asyncio.wait_for(
            probe.run(client, key), timeout=PROBE_TIMEOUT_SECONDS + 1.0
        )
    except asyncio.TimeoutError:
        outcome = _Outcome(DEGRADED, "timeout", _REASONS["timeout"])
    except Exception as exc:  # a probe must never raise into the health endpoint
        _logger.warning("provider health probe failed: %s", type(exc).__name__)
        outcome = _Outcome(DEGRADED, "other", _REASONS["other"])
    latency = int((time.monotonic() - started) * 1000)
    checked_at = _now_iso()
    return [
        ProviderStatus(
            kind=kind,
            name=probe.name,
            status=outcome.status,
            category=outcome.category,
            latency_ms=latency,
            checked_at=checked_at,
            reason=outcome.reason,
        )
        for kind in (probe.kind, *probe.mirror_kinds)
    ]


async def probe_providers(
    *, transport: httpx.AsyncBaseTransport | None = None, only: str | None = None
) -> list[ProviderStatus]:
    """Probe every configured provider (or just ``only``) concurrently (no caching)."""

    selected = [(p, os.getenv(p.key_env, "")) for p in _PROBES if only in (None, p.name)]
    selected = [(p, key) for p, key in selected if key]
    if not selected:
        return []
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS, transport=transport) as client:
        groups = await asyncio.gather(*(_run_probe(p, client, key) for p, key in selected))
    return [status for group in groups for status in group]


# --- cache and passive updates ------------------------------------------------------

_cache: dict[tuple[str, str], ProviderStatus] = {}
_cached_at: float | None = None
_lock: asyncio.Lock | None = None
_transport: httpx.AsyncBaseTransport | None = None  # tests inject a MockTransport
REPROBE_DEBOUNCE_SECONDS = 30.0
_reprobe_last: dict[str, float] = {}
_reprobe_tasks: set[asyncio.Task[None]] = set()


def reset() -> None:
    """Forget the cache (tests)."""

    global _cached_at, _lock
    _cache.clear()
    _cached_at = None
    _lock = None
    _reprobe_last.clear()
    _reprobe_tasks.clear()


def _cache_fresh() -> bool:
    return _cached_at is not None and (time.monotonic() - _cached_at) < CACHE_TTL_SECONDS


async def refresh() -> list[ProviderStatus]:
    """Probe now and replace the cache."""

    global _cached_at, _lock
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        statuses = await probe_providers(transport=_transport)
        _cache.clear()
        _cache.update({(s.kind, s.name): s for s in statuses})
        _cached_at = time.monotonic()
        return list(_cache.values())


async def get_provider_health(*, force: bool = False) -> list[ProviderStatus]:
    """The cached statuses; probes when the cache is older than 60 s (or ``force``)."""

    if force or not _cache_fresh():
        return await refresh()
    return list(_cache.values())


def _schedule_reprobe(name: str) -> None:
    """Re-probe ``name`` in the background (at most once per debounce window)."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # not on the event loop (sync caller): the periodic refresh will catch up
    now = time.monotonic()
    last = _reprobe_last.get(name)
    if last is not None and now - last < REPROBE_DEBOUNCE_SECONDS:
        return
    _reprobe_last[name] = now
    task = loop.create_task(_reprobe(name))
    _reprobe_tasks.add(task)
    task.add_done_callback(_reprobe_tasks.discard)


async def _reprobe(name: str) -> None:
    try:
        statuses = await probe_providers(transport=_transport, only=name)
    except Exception as exc:  # pragma: no cover - probes swallow their own errors
        _logger.warning("provider health re-probe failed: %s", type(exc).__name__)
        return
    for status in statuses:  # not configured here -> nothing to say, stay degraded
        _cache[(status.kind, status.name)] = status


def record_provider_failure(provider: str, category: str) -> None:
    """Passive update from the live failure path: mark ``provider`` ``degraded`` now.

    ``category`` is ``provider_failures.classify_error``'s result (``auth|quota|unavailable|
    other``) or ``timeout``. A passive failure reflects one tenant's request (a model their
    project cannot use, their own rate limit), so it never marks the provider ``down`` for
    everyone: it is ``degraded`` (non-blocking), and for ``auth``/``quota`` an immediate
    background re-probe of that provider (debounced) lets the real key/quota check decide.
    Applies to every kind the provider is reported under; the next probe overwrites it.
    """

    if category not in CATEGORIES:
        category = "other"
    status = DEGRADED
    name = "openai" if provider == "openai-compatible" else provider
    keys = [key for key in _cache if key[1] == name]
    if not keys:
        kinds = [k for k in ("stt", "llm", "tts") if provider in provider_names(k)]
        if name == "openai":
            kinds = ["llm", "realtime"]
        keys = [(kind, name) for kind in kinds]
    for key in keys:
        _cache[key] = ProviderStatus(
            kind=key[0],
            name=key[1],
            status=status,
            category=category,
            latency_ms=None,
            checked_at=_now_iso(),
            reason=_REASONS[category],
        )
    if category in ("auth", "quota"):
        _schedule_reprobe(name)


def overall_status(statuses: list[ProviderStatus]) -> str:
    if not statuses:
        return "unknown"
    if any(s.status == DOWN for s in statuses):
        return DOWN
    if any(s.status == DEGRADED for s in statuses):
        return DEGRADED
    return OK


async def health_report(*, force: bool = False) -> dict[str, Any]:
    statuses = await get_provider_health(force=force)
    return {
        "overall": overall_status(statuses),
        "providers": [s.as_dict() for s in statuses],
    }


async def refresh_forever(interval: float = REFRESH_INTERVAL_SECONDS) -> None:
    """Keep the cache warm (the lifespan task). The first request probes on demand."""

    while True:
        await asyncio.sleep(interval)
        try:
            await refresh()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - refresh() swallows probe errors
            _logger.warning("provider health refresh failed: %s", type(exc).__name__)

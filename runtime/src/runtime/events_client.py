"""Authenticated HTTP transport for runtime events."""

import asyncio
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from .events import RuntimeEvent, is_critical_event

# Best-effort telemetry gets a short retry; critical events keep retrying with exponential
# backoff for about a minute before delivery is reported as failed.
BEST_EFFORT_ATTEMPTS = 3
CRITICAL_RETRY_BUDGET_SECONDS = 60.0
_MAX_BACKOFF_SECONDS = 8.0
_RETRYABLE_STATUSES = frozenset({408, 425, 429})


def is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUSES or status_code >= 500


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    """Choose a bounded delay from Retry-After or the local backoff schedule."""

    fallback = min(0.1 * (2 ** min(attempt, 10)), _MAX_BACKOFF_SECONDS)
    if not retry_after:
        return fallback
    try:
        delay = float(retry_after)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return fallback
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return max(min(max(delay, 0.0), 1.0), fallback)


class PlatformEventClient:
    def __init__(self, platform_url: str, runtime_token: str) -> None:
        self._client = httpx.AsyncClient(base_url=platform_url.rstrip("/"), timeout=10.0)
        self._runtime_token = runtime_token

    async def emit(self, event: RuntimeEvent) -> dict[str, str]:
        """Deliver one event.

        A non-retryable status (4xx other than 408/425/429) is raised immediately as
        ``httpx.HTTPStatusError``. Transient failures are retried: best-effort events a few
        times, critical events (see ``events.CRITICAL_EVENT_TYPES``) with exponential backoff
        until ``CRITICAL_RETRY_BUDGET_SECONDS`` have elapsed; then the last error is raised.
        """

        payload = event.model_dump(mode="json")
        critical = is_critical_event(event.event_type)
        started = time.monotonic()
        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    "/runtime/events",
                    json=payload,
                    headers={"X-Runtime-Token": self._runtime_token},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if not is_retryable_status(exc.response.status_code):
                    raise
                retry_after = exc.response.headers.get("Retry-After")
                error: Exception = exc
            except httpx.TransportError as exc:
                retry_after = None
                error = exc
            else:
                return response.json()
            delay = _retry_delay(retry_after, attempt)
            attempt += 1
            if critical:
                if time.monotonic() - started + delay > CRITICAL_RETRY_BUDGET_SECONDS:
                    raise error
            elif attempt >= BEST_EFFORT_ATTEMPTS:
                raise error
            await asyncio.sleep(delay)

    async def aclose(self) -> None:
        await self._client.aclose()

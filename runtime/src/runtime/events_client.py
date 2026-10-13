"""Authenticated HTTP transport for runtime events."""

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from .events import RuntimeEvent


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    """Choose a bounded delay from Retry-After or the local backoff schedule."""

    fallback = 0.1 * (2**attempt)
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
    return min(max(delay, 0.0), 1.0)


class PlatformEventClient:
    def __init__(self, platform_url: str, runtime_token: str) -> None:
        self._client = httpx.AsyncClient(base_url=platform_url.rstrip("/"), timeout=10.0)
        self._runtime_token = runtime_token

    async def emit(self, event: RuntimeEvent) -> dict[str, str]:
        payload = event.model_dump(mode="json")
        for attempt in range(3):
            try:
                response = await self._client.post(
                    "/runtime/events",
                    json=payload,
                    headers={"X-Runtime-Token": self._runtime_token},
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code in {408, 425, 429} or exc.response.status_code >= 500
                if not retryable or attempt == 2:
                    raise
                retry_after = exc.response.headers.get("Retry-After")
            except httpx.TransportError:
                if attempt == 2:
                    raise
                retry_after = None
            else:
                return response.json()
            await asyncio.sleep(_retry_delay(retry_after, attempt))
        raise RuntimeError("runtime event delivery exhausted retries")

    async def aclose(self) -> None:
        await self._client.aclose()

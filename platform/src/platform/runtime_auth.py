"""Runtime authentication utilities."""

import hmac
import os

from fastapi import Header, HTTPException, status


async def require_runtime_token(
    x_runtime_token: str | None = Header(default=None),
) -> None:
    """Validate the runtime token from the X-Runtime-Token header.

    Raises:
        HTTPException: 503 if PLATFORM_RUNTIME_TOKEN env var is not configured
        HTTPException: 401 if token is missing or invalid
    """
    expected = os.getenv("PLATFORM_RUNTIME_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="runtime_auth_not_configured")
    if not x_runtime_token or not hmac.compare_digest(x_runtime_token, expected):
        raise HTTPException(status_code=401, detail="invalid_runtime_token")

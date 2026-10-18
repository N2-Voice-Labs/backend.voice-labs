"""Outbound SIP through LiveKit, plus the Platform self-call used to open sessions.

Everything here is gated by ``LIVEKIT_SIP_TRUNK_ID`` (the id of a LiveKit *outbound*
SIP trunk) together with the normal ``LIVEKIT_URL`` / ``LIVEKIT_API_KEY`` /
``LIVEKIT_API_SECRET``. Without them :func:`sip_configured` is false and callers must
not dial. Phone calls were never exercised against a real trunk; the request shape
follows ``livekit-api`` (``SipService.create_sip_participant``).

Optional tuning:

``LIVEKIT_SIP_RINGING_TIMEOUT_S``
    Seconds to let the callee ring (LiveKit default is 30).
``LIVEKIT_SIP_KRISP``
    ``1`` enables Krisp noise cancellation on the SIP leg.
``PLATFORM_API_URL``
    Where the Platform reaches itself to open voice sessions (default
    ``http://localhost:8000``).
"""

import os
from typing import Any

import httpx

SIP_IDENTITY_PREFIX = "sip-"
# SIP final responses that mean "nobody took the call": request timeout, temporarily
# unavailable, busy, request terminated, busy everywhere, declined.
NO_ANSWER_SIP_CODES = frozenset({408, 480, 486, 487, 600, 603})
_DEFAULT_PLATFORM_URL = "http://localhost:8000"
_SESSION_TIMEOUT_S = 15.0

# Test seam: ``httpx`` transport used for calls the Platform makes to itself.
http_transport: httpx.AsyncBaseTransport | None = None


class SipError(Exception):
    """A SIP operation could not be completed."""

    def __init__(
        self,
        message: str,
        *,
        sip_status_code: int | None = None,
        sip_status: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.sip_status_code = sip_status_code
        self.sip_status = sip_status
        self.code = code


class SipDialError(SipError):
    """The outbound call did not connect."""

    @property
    def no_answer(self) -> bool:
        """True when the callee did not pick up (ring timeout, busy, declined)."""
        return (
            self.sip_status_code in NO_ANSWER_SIP_CODES
            or self.code == "deadline_exceeded"
        )


class SessionCreateError(Exception):
    """``POST /voice/sessions`` failed when the Platform called itself."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def sip_trunk_id() -> str | None:
    return os.getenv("LIVEKIT_SIP_TRUNK_ID") or None


def sip_configured() -> bool:
    return bool(
        sip_trunk_id()
        and os.getenv("LIVEKIT_URL")
        and os.getenv("LIVEKIT_API_KEY")
        and os.getenv("LIVEKIT_API_SECRET")
    )


def api_url(url: str) -> str:
    """The LiveKit server API speaks http(s); the client URL is ws(s)."""
    if url.startswith("wss://"):
        return "https://" + url[len("wss://") :]
    if url.startswith("ws://"):
        return "http://" + url[len("ws://") :]
    return url


def _new_client() -> Any:
    """Build a ``livekit.api.LiveKitAPI`` (patched in tests)."""
    try:
        from livekit import api
    except ImportError as exc:  # pragma: no cover - livekit-api is a dependency
        raise SipError("livekit-api is not installed") from exc
    return api.LiveKitAPI(
        api_url(os.environ["LIVEKIT_URL"]),
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )


def _build_request(
    trunk_id: str,
    phone_number: str,
    room_name: str,
    participant_identity: str,
    display_name: str | None,
) -> Any:
    from livekit import api

    request = api.CreateSIPParticipantRequest(
        sip_trunk_id=trunk_id,
        sip_call_to=phone_number,
        room_name=room_name,
        participant_identity=participant_identity,
        wait_until_answered=True,
        krisp_enabled=os.getenv("LIVEKIT_SIP_KRISP") == "1",
    )
    if display_name:
        request.participant_name = display_name
    ringing = os.getenv("LIVEKIT_SIP_RINGING_TIMEOUT_S")
    if ringing and ringing.isdigit() and int(ringing) > 0:
        request.ringing_timeout.seconds = int(ringing)
    return request


async def dial(
    phone_number: str,
    room_name: str,
    participant_identity: str,
    *,
    display_name: str | None = None,
) -> str:
    """Dial ``phone_number`` into ``room_name``; return the SIP participant id.

    Blocks until the callee answers (``wait_until_answered``). ``display_name`` is the
    LiveKit participant name (not the SIP From display). Raises :class:`SipDialError`
    if the call does not connect, and :class:`SipError` if SIP is not configured.
    """

    trunk_id = sip_trunk_id()
    if not sip_configured() or trunk_id is None:
        raise SipError("sip_not_configured")
    try:
        from livekit import api
    except ImportError as exc:  # pragma: no cover
        raise SipError("livekit-api is not installed") from exc
    request = _build_request(
        trunk_id, phone_number, room_name, participant_identity, display_name
    )
    try:
        async with _new_client() as lkapi:
            info = await lkapi.sip.create_sip_participant(request)
    except api.TwirpError as exc:
        raise SipDialError(
            exc.message,
            sip_status_code=getattr(exc, "sip_status_code", None),
            sip_status=getattr(exc, "sip_status", None),
            code=exc.code,
        ) from exc
    except Exception as exc:
        raise SipDialError(f"{type(exc).__name__}: {exc}") from exc
    return str(info.participant_id)


async def hangup(room_name: str, participant_identity: str) -> None:
    """Best-effort removal of a participant (ends its SIP call). Never raises."""

    if not sip_configured():
        return
    try:
        from livekit import api

        async with _new_client() as lkapi:
            await lkapi.room.remove_participant(
                api.RoomParticipantIdentity(
                    room=room_name, identity=participant_identity
                )
            )
    except Exception:
        return


async def create_voice_session(agent_id: str, deployment_id: str | None) -> dict[str, Any]:
    """Open a voice session through the Platform's own ``POST /voice/sessions``.

    Going through HTTP keeps every session validation (agent enabled, deployment
    active, provider checks, runtime hand-off) in one place.
    """

    base = os.getenv("PLATFORM_API_URL", _DEFAULT_PLATFORM_URL).rstrip("/")
    body: dict[str, Any] = {"agent_id": agent_id}
    if deployment_id is not None:
        body["deployment_id"] = deployment_id
    try:
        async with httpx.AsyncClient(
            timeout=_SESSION_TIMEOUT_S, transport=http_transport
        ) as client:
            response = await client.post(f"{base}/voice/sessions", json=body)
    except httpx.HTTPError as exc:
        raise SessionCreateError(f"platform_unreachable: {type(exc).__name__}") from exc
    if response.status_code != 201:
        raise SessionCreateError(
            f"session_create_http_{response.status_code}", response.status_code
        )
    try:
        data = response.json()
        data["session_id"], data["conversation_id"], data["room_name"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SessionCreateError("session_create_bad_response") from exc
    return data


async def get_session_status(session_id: str) -> str | None:
    """Current ``status`` of a voice session, or None if it cannot be read."""

    base = os.getenv("PLATFORM_API_URL", _DEFAULT_PLATFORM_URL).rstrip("/")
    try:
        async with httpx.AsyncClient(
            timeout=_SESSION_TIMEOUT_S, transport=http_transport
        ) as client:
            response = await client.get(f"{base}/voice/sessions/{session_id}")
        if response.status_code != 200:
            return None
        status = response.json().get("status")
        return status if isinstance(status, str) else None
    except (httpx.HTTPError, ValueError, AttributeError):
        return None


async def end_voice_session(session_id: str) -> None:
    """Best-effort ``POST /voice/sessions/{id}/end``. Never raises."""

    base = os.getenv("PLATFORM_API_URL", _DEFAULT_PLATFORM_URL).rstrip("/")
    try:
        async with httpx.AsyncClient(
            timeout=_SESSION_TIMEOUT_S, transport=http_transport
        ) as client:
            await client.post(f"{base}/voice/sessions/{session_id}/end")
    except httpx.HTTPError:
        return

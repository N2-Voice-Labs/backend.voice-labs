"""Outbound SIP through LiveKit, plus the in-process helpers that open and end sessions.

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

Sessions are opened and ended by calling :mod:`.sessions` directly under a ``system``
context for the deployment's / campaign's organisation; the Platform no longer calls its
own HTTP API (which would need credentials and carry no organisation).
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .context import AuthContext
from .tenancy import scoped_session_for

SIP_IDENTITY_PREFIX = "sip-"
# SIP final responses that mean "nobody took the call": request timeout, temporarily
# unavailable, busy, request terminated, busy everywhere, declined.
NO_ANSWER_SIP_CODES = frozenset({408, 480, 486, 487, 600, 603})


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
    """Opening a voice session in-process failed (carries the HTTP-style status)."""

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


def system_ctx(org_id: str) -> AuthContext:
    """Context for the Platform's own background work (bridge calls, campaign dialer)."""

    return AuthContext(org_id=org_id, principal="system", role="admin")


@contextmanager
def _db_for(org_id: str, db: Session | None) -> Iterator[Session]:
    """The caller's session, or a short-lived one pinned to ``org_id``."""

    if db is not None:
        yield db
    else:
        with scoped_session_for(org_id) as scoped_db:
            yield scoped_db


async def create_voice_session(
    agent_id: str,
    deployment_id: str | None,
    *,
    org_id: str,
    db: Session | None = None,
) -> dict[str, Any]:
    """Open a voice session in-process through :func:`sessions.open_voice_session`.

    Same validation and Runtime hand-off as ``POST /voice/sessions`` without an HTTP
    round trip (which would need credentials and an organisation). ``HTTPException``
    failures surface as :class:`SessionCreateError` carrying the status code.
    """

    from . import sessions  # sessions imports this module

    ctx = system_ctx(org_id)
    try:
        with _db_for(org_id, db) as session_db:
            data = await sessions.open_voice_session(
                session_db, ctx, agent_id=agent_id, deployment_id=deployment_id
            )
    except HTTPException as exc:
        raise SessionCreateError(
            f"session_create_http_{exc.status_code}", exc.status_code
        ) from exc
    try:
        data["session_id"], data["conversation_id"], data["room_name"]
    except (KeyError, TypeError) as exc:
        raise SessionCreateError("session_create_bad_response") from exc
    return data


async def get_session_status(
    session_id: str, *, org_id: str, db: Session | None = None
) -> str | None:
    """Current ``status`` of an organisation's voice session, or None if unreadable."""

    from . import sessions

    try:
        with _db_for(org_id, db) as session_db:
            value = await sessions.get_session_status(
                session_db, system_ctx(org_id), session_id
            )
    except Exception:
        return None
    return value if isinstance(value, str) else None


async def end_voice_session(
    session_id: str, *, org_id: str, db: Session | None = None
) -> None:
    """Best-effort end of an organisation's session. Never raises."""

    from . import sessions

    try:
        with _db_for(org_id, db) as session_db:
            await sessions.end_session(session_db, system_ctx(org_id), session_id)
    except Exception:
        return

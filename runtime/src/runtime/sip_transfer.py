"""Transfer a live SIP call to a human through LiveKit (``TransferSIPParticipant``).

Plugs into :data:`runtime.tool_runtime.transfer_handler`. When the LLM calls
``transfer_to_human`` the tool runtime has already emitted ``HandoffRequested``; this
handler then, if the room holds a SIP participant, asks LiveKit to transfer it to the
configured destination. Web sessions have no SIP participant, so nothing happens beyond
the recorded handoff.

Where the pieces come from:

* Room: the session's ``room_name`` attribute if present, else the accepted session
  contract (``runtime.api._session_contracts``), else the Platform naming convention
  ``voice-labs-<session_id>``.
* SIP participant: a participant of kind ``SIP`` or whose identity starts with
  ``sip-`` (the prefix the Platform dials with).
* Destination: ``transfer_to`` on the session or its contract (the Platform sends the
  deployment's ``configuration.transfer_to`` in the session contract), else the
  ``SIP_TRANSFER_TO`` fallback. Bare E.164 numbers become ``tel:+...``;
  ``tel:``/``sip:``/``sips:`` URIs pass through.

The handler is installed whenever the LiveKit URL/key/secret are set, because the
destination may come per deployment. If a SIP call needs a transfer and no destination
resolves, :func:`transfer` raises ``TransferError("transfer_not_configured")`` and the
tool reports that error to the LLM instead of claiming success.
Never exercised against a real trunk.
"""

import logging
import os
import re
from typing import TYPE_CHECKING, Any

from . import tool_runtime

if TYPE_CHECKING:
    from .session import RuntimeSession

_logger = logging.getLogger(__name__)

SIP_IDENTITY_PREFIX = "sip-"
_E164 = re.compile(r"^\+[1-9]\d{6,14}$")
_URI_SCHEMES = ("tel:", "sip:", "sips:")


class TransferError(Exception):
    """The SIP transfer could not be started or did not complete."""


def configured() -> bool:
    """True when LiveKit credentials exist; the destination is resolved per call."""

    return bool(
        os.getenv("LIVEKIT_URL")
        and os.getenv("LIVEKIT_API_KEY")
        and os.getenv("LIVEKIT_API_SECRET")
    )


def api_url(url: str) -> str:
    if url.startswith("wss://"):
        return "https://" + url[len("wss://") :]
    if url.startswith("ws://"):
        return "http://" + url[len("ws://") :]
    return url


def normalize_destination(value: str) -> str:
    value = value.strip()
    if value.lower().startswith(_URI_SCHEMES):
        return value
    if _E164.match(value):
        return f"tel:{value}"
    raise TransferError("invalid_transfer_destination")


def _contract(session: "RuntimeSession") -> dict[str, Any]:
    from . import api  # lazy: api imports this module's installer

    return api._session_contracts.get(str(session.session_id)) or {}


def resolve_room(session: "RuntimeSession") -> str:
    room = getattr(session, "room_name", None) or _contract(session).get("room_name")
    return str(room or f"voice-labs-{session.session_id}")


def resolve_destination(session: "RuntimeSession") -> str | None:
    value = (
        getattr(session, "transfer_to", None)
        or _contract(session).get("transfer_to")
        or os.getenv("SIP_TRANSFER_TO")
    )
    return normalize_destination(value) if isinstance(value, str) and value.strip() else None


def _new_client() -> Any:
    """Build a ``livekit.api.LiveKitAPI`` (patched in tests)."""
    from livekit import api

    return api.LiveKitAPI(
        api_url(os.environ["LIVEKIT_URL"]),
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    )


def _is_sip(participant: Any) -> bool:
    from livekit import api

    kind_sip = api.ParticipantInfo.Kind.SIP  # type: ignore[attr-defined]
    return participant.kind == kind_sip or str(participant.identity).startswith(
        SIP_IDENTITY_PREFIX
    )


async def transfer(session: "RuntimeSession", reason: str | None) -> str:
    """Transfer the session's SIP participant to the configured human destination.

    Returns ``"no_sip_participant"`` (nothing to do, e.g. a web call) or
    ``"transferred"``. Raises :class:`TransferError` on any failure (including
    ``transfer_not_configured`` when a SIP call has no resolvable destination), which the
    tool runtime reports to the LLM as an error.
    """

    try:
        from livekit import api
    except ImportError as exc:
        raise TransferError("livekit_api_not_installed") from exc
    room = resolve_room(session)
    try:
        async with _new_client() as lkapi:
            listing = await lkapi.room.list_participants(
                api.ListParticipantsRequest(room=room)
            )
            sip_participants = [p for p in listing.participants if _is_sip(p)]
            if not sip_participants:
                _logger.info("No SIP participant in %s; handoff recorded only", room)
                return "no_sip_participant"
            destination = resolve_destination(session)
            if destination is None:
                raise TransferError("transfer_not_configured")
            response = await lkapi.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    participant_identity=sip_participants[0].identity,
                    room_name=room,
                    transfer_to=destination,
                )
            )
    except TransferError:
        raise
    except Exception as exc:
        raise TransferError(f"{type(exc).__name__}: {exc}") from exc
    # A failed transfer is reported in the response, not raised.
    if getattr(response, "status", None) == api.SIPTransferStatus.STS_TRANSFER_FAILED:
        raise TransferError(f"transfer_failed: {getattr(response, 'reason', '')}")
    _logger.info("Transferred SIP participant in %s (reason=%r)", room, reason)
    return "transferred"


def install_if_configured() -> bool:
    """Register :func:`transfer` as the tool runtime's transfer handler when enabled."""

    if not configured():
        return False
    tool_runtime.set_transfer_handler(transfer)
    return True

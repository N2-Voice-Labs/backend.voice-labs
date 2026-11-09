"""LiveKit access-token generation using the official server SDK."""

import os
from datetime import timedelta

LISTEN_ROOM_SEPARATOR = "--listen--"


def public_livekit_url() -> str:
    """LiveKit URL handed to browsers/clients in session responses.

    ``LIVEKIT_PUBLIC_URL`` when set (e.g. ``ws://localhost:7880`` while the server-side
    ``LIVEKIT_URL`` is the in-network ``ws://livekit:7880``); otherwise ``LIVEKIT_URL``.
    """

    return os.getenv("LIVEKIT_PUBLIC_URL") or os.getenv("LIVEKIT_URL", "")


def listen_room_name(room_name: str, identity: str) -> str:
    """Name of the room only ``identity`` and the Runtime join to hear translations.

    The Runtime derives the same name (``runtime.participants.listen_room_name``);
    ``docs/contracts.md`` documents the convention.
    """

    return f"{room_name}{LISTEN_ROOM_SEPARATOR}{identity}"


PHONE_ROOM_SEPARATOR = "--phone--"


def phone_room_name(room_name: str, identity: str) -> str:
    """Name of the room a phone participant is dialled into under private listening.

    Only the phone and the Runtime are in it. The Runtime derives the same name
    (``runtime.participants.phone_room_name``); ``docs/contracts.md`` documents it.
    """

    return f"{room_name}{PHONE_ROOM_SEPARATOR}{identity}"


def private_listening_enabled() -> bool:
    """Whether interpreter participants get the isolated listening scheme.

    Default off until the web app joins the listen room (``connection.listen``); set
    ``PLATFORM_PRIVATE_LISTENING=1`` to turn it on.
    """

    return os.getenv("PLATFORM_PRIVATE_LISTENING", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _token(room_name: str, identity: str, **grants: bool) -> str:
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    if not key or not secret:
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be configured")
    try:
        from livekit import api
    except ImportError as exc:
        raise RuntimeError("install the livekit-api package to generate LiveKit tokens") from exc
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name, **grants))
        .with_ttl(timedelta(minutes=10))
        .to_jwt()
    )


def create_participant_token(room_name: str, identity: str, *, can_subscribe: bool = True) -> str:
    """Token for the session room.

    ``can_subscribe=False`` is used for interpreter participants that must never receive
    anybody's audio from this room (their translation comes from their listen room).
    LiveKit enforces it for the whole connection: auto-subscription and subscriptions
    requested by the client or forced through the server API are all refused.
    """

    return _token(room_name, identity, can_publish=True, can_subscribe=can_subscribe)


def create_listener_token(room_name: str, identity: str) -> str:
    """Subscribe-only token for a participant's listen room (see :func:`listen_room_name`)."""

    return _token(
        listen_room_name(room_name, identity),
        identity,
        can_publish=False,
        can_publish_data=False,
        can_subscribe=True,
    )

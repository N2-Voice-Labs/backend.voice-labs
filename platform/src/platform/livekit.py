"""LiveKit access-token generation using the official server SDK."""

import os
from datetime import timedelta


def create_participant_token(room_name: str, identity: str) -> str:
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
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_ttl(timedelta(minutes=10))
        .to_jwt()
    )

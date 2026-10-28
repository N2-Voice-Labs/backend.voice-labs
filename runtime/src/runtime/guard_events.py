"""Events of the answer guard and empty translations (ST-12), plus ``turn_id`` lookup.

No Pipecat dependency. Payloads never carry transcript text (privacy, size): the text is
already in the ``TranscriptFinal`` events of the same turn.

* ``AnswerEmpty`` ``{turn_id?, language, fallback: "repeat"}``
* ``AnswerTruncated`` ``{turn_id?, spoken_chars, max_spoken_chars}``
* ``TranslationEmpty`` ``{turn_id?, stage, source_language, target_language, user_id?}``
  with ``stage`` one of ``interpreter|inbound|outbound``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .turn_ids import turn_state

_logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"


def current_turn_id(session: Any | None, timeline: Any | None = None) -> int | None:
    """The turn an event belongs to: the utterance's own id, else the session's current turn."""

    turn_id = getattr(timeline, "turn_id", None)
    if turn_id is not None:
        return turn_id
    state = turn_state(session)
    return (state.current or None) if state is not None else None


def emit_session_event(session: Any | None, event_type: str, payload: dict[str, object]) -> None:
    """Queue a Platform event; telemetry failures never break the conversation."""

    if session is None or not getattr(session, "event_client", None):
        return
    try:
        session.enqueue_event(session.event(event_type, payload))
    except RuntimeError:
        _logger.warning("Dropped %s event for session %s", event_type, session.session_id)


def _with_turn(session: Any | None, payload: dict[str, object]) -> dict[str, object]:
    turn_id = current_turn_id(session)
    return {"turn_id": turn_id, **payload} if turn_id is not None else payload


def answer_empty_hook(session: Any | None) -> Callable[[str | None], None]:
    def on_empty(language: str | None) -> None:
        emit_session_event(
            session,
            "AnswerEmpty",
            _with_turn(session, {"language": language or DEFAULT_LANGUAGE, "fallback": "repeat"}),
        )

    return on_empty


def answer_truncated_hook(
    session: Any | None, max_spoken_chars: int
) -> Callable[[int], None]:
    def on_truncated(spoken_chars: int) -> None:
        emit_session_event(
            session,
            "AnswerTruncated",
            _with_turn(
                session,
                {"spoken_chars": spoken_chars, "max_spoken_chars": max_spoken_chars},
            ),
        )

    return on_truncated


def translation_empty_hook(
    session: Any | None, stage: str
) -> Callable[[dict[str, object]], None]:
    """Hook for ``on_empty_translation``; drops the ``text`` the processors pass along."""

    def on_empty(info: dict[str, object]) -> None:
        payload: dict[str, object] = {
            "stage": stage,
            "source_language": info.get("source_language"),
            "target_language": info.get("target_language"),
        }
        if info.get("user_id"):
            payload["user_id"] = info["user_id"]
        turn_id = info.get("turn_id")
        if turn_id is not None:
            payload = {"turn_id": turn_id, **payload}
        else:
            payload = _with_turn(session, payload)
        emit_session_event(session, "TranslationEmpty", payload)

    return on_empty

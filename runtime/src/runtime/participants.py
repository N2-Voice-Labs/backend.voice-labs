"""Participant mapping for two-person interpreter sessions (TR-020).

The Platform sends ``participants``: ``{livekit_identity: {language, voice_id,
display_name}}`` in the session contract and, when the second person joins, through
``POST /sessions/{id}/participants``. This module holds the live mapping.

It is pure Python (no Pipecat import) so the HTTP layer can use it without the voice
extras. The pipeline reads the *same* :class:`ParticipantDirectory` instance, so a
participant added after the pipeline started is visible to it immediately.

Lanes
-----
LiveKit delivers one audio stream per remote participant (``UserAudioRawFrame.user_id``
is the LiveKit identity), interleaved on a single Pipecat stream. One speech-to-text
connection cannot transcribe two interleaved voices, so each participant gets a *lane*:
its own VAD, Deepgram connection (in the participant's declared language) and turn
detection. A participant's lane is its position in the mapping (insertion order); at
most :data:`MAX_PARTICIPANTS` are admitted.

All room participants hear the translated audio (it is published to the LiveKit room).
Routing each translation only to the other person is future work.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_PARTICIPANTS = 2
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")


class ParticipantError(ValueError):
    """Invalid participant update; ``code`` is the API error detail."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def normalize_language(value: object) -> str | None:
    """Return the lowercase base code (``"en-US"`` -> ``"en"``) or None."""

    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip().lower().replace("_", "-")
    base = text.split("-", 1)[0]
    return base or None


@dataclass(frozen=True)
class Participant:
    identity: str
    language: str
    voice_id: str | None = None
    display_name: str | None = None


Listener = Callable[[Participant], Awaitable[None]]


class ParticipantDirectory:
    """Live ``identity -> Participant`` mapping with stable lane numbers."""

    def __init__(self, languages: Sequence[str]) -> None:
        pair = tuple(normalize_language(code) or "" for code in languages)
        if len(pair) != 2 or not all(pair) or pair[0] == pair[1]:
            raise ParticipantError("invalid_interpreter_languages")
        self.languages: tuple[str, str] = (pair[0], pair[1])
        self._participants: dict[str, Participant] = {}
        self._listeners: list[Listener] = []

    @classmethod
    def from_mapping(
        cls, languages: Sequence[str], mapping: Mapping[str, Mapping[str, Any]] | None
    ) -> "ParticipantDirectory":
        directory = cls(languages)
        for identity, spec in (mapping or {}).items():
            directory.set(identity, spec)
        return directory

    def __len__(self) -> int:
        return len(self._participants)

    def __contains__(self, identity: object) -> bool:
        return identity in self._participants

    def set(self, identity: str, spec: Mapping[str, Any]) -> Participant:
        """Add or replace a participant without notifying listeners."""

        if not isinstance(identity, str) or not identity:
            raise ParticipantError("invalid_participant")
        language = normalize_language(spec.get("language"))
        if not language or not _LANGUAGE.match(language):
            raise ParticipantError("invalid_participant_language")
        if identity not in self._participants and len(self._participants) >= MAX_PARTICIPANTS:
            raise ParticipantError("session_full")
        voice_id = spec.get("voice_id") or None
        display_name = spec.get("display_name") or None
        participant = Participant(identity, language, voice_id, display_name)
        self._participants[identity] = participant
        return participant

    async def update(self, identity: str, spec: Mapping[str, Any]) -> Participant:
        """Add or replace a participant and tell listeners (the pipeline lanes)."""

        participant = self.set(identity, spec)
        for listener in list(self._listeners):
            await listener(participant)
        return participant

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def get(self, identity: str | None) -> Participant | None:
        return self._participants.get(identity) if identity else None

    def as_contract(self) -> dict[str, dict[str, str | None]]:
        return {
            p.identity: {
                "language": p.language,
                "voice_id": p.voice_id,
                "display_name": p.display_name,
            }
            for p in self._participants.values()
        }

    # -- lanes -----------------------------------------------------------------

    def lane_of(self, identity: str | None) -> int | None:
        """Lane number of a known participant, else None (audio is ignored)."""

        if not identity or identity not in self._participants:
            return None
        return list(self._participants).index(identity)

    def identity_for_lane(self, lane: int) -> str | None:
        identities = list(self._participants)
        return identities[lane] if lane < len(identities) else None

    def language_for_lane(self, lane: int) -> str:
        """Speech language of a lane; a pair language nobody else uses while it is empty."""

        participant = self.get(self.identity_for_lane(lane))
        if participant is not None:
            return participant.language
        taken = {p.language for p in self._participants.values()}
        return next((code for code in self.languages if code not in taken), self.languages[0])

    # -- translation direction ---------------------------------------------------

    def target_for(self, identity: str) -> str:
        """Language a speaker's words are translated into.

        The other participant's language when present; before the second person has
        joined, the other language of the agent's pair.
        """

        speaker = self._participants[identity]
        for other in self._participants.values():
            if other.identity != identity and other.language != speaker.language:
                return other.language
        return next(
            (code for code in self.languages if code != speaker.language), self.languages[1]
        )

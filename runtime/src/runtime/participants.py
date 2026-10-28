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

Audio ownership (RM-001 / RM-002)
---------------------------------
Each participant also has a :class:`ParticipantContext`: who they are, what they speak,
which language their words are translated into, their voice, their lane, and the
Runtime-published track *they* hear (``translation-for-<identity>``). The translation of
A's speech is synthesised in B's language and A's voice and goes only to B's track;
see :mod:`runtime.audio_routing` for how that and "nobody hears the other's original
microphone" are enforced server-side.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MAX_PARTICIPANTS = 2
_LANGUAGE = re.compile(r"^[a-z]{2,3}$")
TRANSLATION_TRACK_PREFIX = "translation-for-"


def translation_track_name(identity: str) -> str:
    """LiveKit track name of the translation a listener hears."""

    return f"{TRANSLATION_TRACK_PREFIX}{identity}"


LISTEN_ROOM_SEPARATOR = "--listen--"


def listen_room_name(room_name: str, identity: str) -> str:
    """Private room a listener hears their translation in (``private_listening``).

    Only that listener (subscribe-only) and the Runtime join it. The Platform derives the
    same name (``platform.livekit.listen_room_name``); ``docs/contracts.md`` documents it.
    """

    return f"{room_name}{LISTEN_ROOM_SEPARATOR}{identity}"


PHONE_ROOM_SEPARATOR = "--phone--"


def phone_room_name(room_name: str, identity: str) -> str:
    """Private room a phone (SIP) participant is dialled into (``private_listening``).

    The phone and the Runtime are the only members: the phone's microphone is published
    there and its translation (``translation-for-<identity>``) comes back on the same
    connection, so it can never hear another person's original voice. The Platform
    derives the same name (``platform.livekit.phone_room_name``).
    """

    return f"{room_name}{PHONE_ROOM_SEPARATOR}{identity}"


def translation_destination(lane: int) -> str:
    """Pipecat output destination feeding the track of the listener in ``lane``."""

    return f"translation-lane-{lane}"


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
    # A phone (SIP) participant dialled into its own :func:`phone_room_name` room instead
    # of the session room. Only with ``private_listening``.
    phone_room: bool = False


@dataclass(frozen=True)
class ParticipantContext:
    """Everything the pipeline and the audio router need to know about one participant.

    ``source_language`` is what the participant speaks, ``target_language`` the language
    their words are translated into (the other participant's). ``voice_id`` is the voice
    their translated words are spoken in. ``output_track`` / ``output_destination`` are
    the track and Pipecat destination carrying what this participant *hears* (the other
    person's translated speech); ``listener`` is who hears *this* participant's
    translation, or None while the other person is not in the room.
    """

    identity: str
    lane: int
    display_name: str | None
    source_language: str
    target_language: str
    voice_id: str | None
    output_track: str
    output_destination: str
    listener: str | None


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
        # Who is in the LiveKit room right now. None until a router reports it, in
        # which case every directory entry counts as present.
        self._present: set[str] | None = None

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
        participant = Participant(
            identity, language, voice_id, display_name, bool(spec.get("phone_room"))
        )
        self._participants[identity] = participant
        return participant

    async def remove(self, identity: str) -> None:
        """Forget a phone-room participant whose call never connected (and tell listeners).

        Only the most recently added participant can leave, so the remaining lane
        numbers (insertion order) stay stable.
        """

        participant = self._participants.get(identity)
        if participant is None:
            raise ParticipantError("participant_not_found")
        if not participant.phone_room or identity != list(self._participants)[-1]:
            raise ParticipantError("participant_not_removable")
        del self._participants[identity]
        if self._present is not None:
            self._present.discard(identity)
        for listener in list(self._listeners):
            await listener(participant)

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

    def as_contract(self) -> dict[str, dict[str, Any]]:
        return {
            p.identity: {
                "language": p.language,
                "voice_id": p.voice_id,
                "display_name": p.display_name,
                **({"phone_room": True} if p.phone_room else {}),
            }
            for p in self._participants.values()
        }

    def phone_room_identities(self) -> list[str]:
        """Participants that live in their own phone room, not in the session room."""

        return [p.identity for p in self._participants.values() if p.phone_room]

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

    # -- per-participant context and presence (RM-001) --------------------------

    def set_present(self, identities: Sequence[str] | set[str]) -> None:
        """Record who is in the LiveKit room (reported by the audio router)."""

        self._present = set(identities)

    def is_present(self, identity: str | None) -> bool:
        if not identity or identity not in self._participants:
            return False
        return self._present is None or identity in self._present

    def listener_for(self, speaker: str) -> str | None:
        """Identity that hears ``speaker``'s translation: the other participant, if present."""

        for other in self._participants:
            if other != speaker and self.is_present(other):
                return other
        return None

    def context(self, identity: str) -> ParticipantContext:
        participant = self._participants[identity]
        lane = self.lane_of(identity)
        assert lane is not None
        return ParticipantContext(
            identity=identity,
            lane=lane,
            display_name=participant.display_name,
            source_language=participant.language,
            target_language=self.target_for(identity),
            voice_id=participant.voice_id,
            output_track=translation_track_name(identity),
            output_destination=translation_destination(lane),
            listener=self.listener_for(identity),
        )

    def contexts(self) -> list[ParticipantContext]:
        return [self.context(identity) for identity in self._participants]

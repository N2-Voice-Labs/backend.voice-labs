"""The small interface between the interpreter pipeline and a translation engine (RM-010).

An engine is "what turns the speaker's audio into the listener's audio" between
``transport.input()`` and ``transport.output()``. The pipeline owns everything around it
(LiveKit transport, per-participant routing, the event observer, the worker); an engine
only says what it can do and builds the processors that do it::

    transport.input -> engine.build_processors(...) -> transport.output

The interface is deliberately tiny and says nothing about providers: the cascaded engine
uses the STT and TTS providers the session names (``speech_providers``) and OpenAI, the
realtime engine OpenAI ``gpt-realtime-translate``; a Qwen or local-model engine plugs in by implementing the same
five members and calling :func:`register_engine`. ``interpreter.engine`` in the session
contract is the registry key.

* ``name``: the registry key and the ``engine`` value of the contract.
* ``supports_two_way``: serves ``direction: "auto"`` and per-participant sessions
  (several speakers, each translated for the other).
* ``uses_own_voice``: speaks with the agent's / participant's own voice through the
  configured STT and TTS providers (so their API keys are required). ``False`` means the
  provider speaks and needs only the translation credentials.
* ``validate(settings)``: raise ``RuntimeError`` when the engine cannot serve the settings.
* ``build_processors(settings, context)``: the processors, in order.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .speech_providers import build_stt, build_tts

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pipecat.processors.frame_processor import FrameProcessor

    from .interpreter import InterpreterSettings
    from .participants import ParticipantDirectory
    from .pipeline import PipelineConfig
    from .timeline import TimelineTracker


@dataclass(frozen=True)
class EngineContext:
    """What the pipeline gives an engine besides the settings."""

    config: "PipelineConfig"
    session: Any | None = None
    # Two-person sessions only (engines with ``supports_two_way``).
    directory: "ParticipantDirectory | None" = None
    # Utterance timeline recorder (RM-003); None without a session.
    timeline: "TimelineTracker | None" = None


@runtime_checkable
class TranslationEngine(Protocol):
    name: str
    supports_two_way: bool
    uses_own_voice: bool

    def validate(self, settings: "InterpreterSettings") -> None: ...

    def build_processors(
        self, settings: "InterpreterSettings", context: EngineContext
    ) -> "list[FrameProcessor]": ...


_ENGINES: dict[str, TranslationEngine] = {}


def register_engine(engine: TranslationEngine) -> TranslationEngine:
    """Make ``engine`` selectable as ``interpreter.engine`` (replaces the same name)."""

    _ENGINES[engine.name] = engine
    return engine


def get_engine(name: str) -> TranslationEngine:
    try:
        return _ENGINES[name]
    except KeyError:
        raise RuntimeError(f"unsupported interpreter engine: {name}") from None


def engine_names() -> tuple[str, ...]:
    return tuple(_ENGINES)


class CascadedEngine:
    """STT -> OpenAI translation -> TTS per utterance (the default engine)."""

    name = "cascaded"
    supports_two_way = True
    uses_own_voice = True

    def validate(self, settings: "InterpreterSettings") -> None:
        return None

    def build_processors(
        self, settings: "InterpreterSettings", context: EngineContext
    ) -> "list[FrameProcessor]":
        """``VAD -> STT -> user turns -> translation -> TTS -> events``.

        Interruptions are disabled (a translation is never cut off by the next speaker).
        Two-person sessions replace the single ``VAD -> STT -> turns`` stage with one lane
        per participant and the single TTS with one TTS per listener; see
        ``participant_lanes``. The latency profile (RM-011) sets the turn pause, the
        chunking, the translation model and the context length.
        """

        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.pipeline.parallel_pipeline import ParallelPipeline
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.turns.user_start import (
            TranscriptionUserTurnStartStrategy,
            VADUserTurnStartStrategy,
        )
        from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
        from pipecat.turns.user_turn_processor import UserTurnProcessor
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        from .glossary import glossary_keyterms
        from .guard_events import translation_empty_hook
        from .interpreter import OpenAITranslator, TranslationProcessor
        from .language_capabilities import stt_language_for
        from .participants import MAX_PARTICIPANTS, translation_destination

        config, directory = context.config, context.directory
        source, target = settings.languages
        profile = settings.profile

        def make_stt(language: str):
            keyterms = glossary_keyterms(settings, None if language == "multi" else language)
            return build_stt(config, language, keyterms)

        def make_tts(language: str, destination: str | None = None):
            extra = {"transport_destination": destination} if destination else {}
            return build_tts(config, language, **extra)

        model = config.provider_settings.get("llm", {}).get("model") or profile.translation_model
        translation = TranslationProcessor(
            settings=settings,
            translate=OpenAITranslator(
                api_key=config.openai_api_key,
                model=model,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            ),
            model=model,
            session=context.session,
            initial_tts_language=target,
            participants=directory,
            default_voice_id=config.tts_voice_id,
            route_by_listener=directory is not None,
            timeline=context.timeline,
            context_pairs=profile.context_pairs,
            clause_min_chars=profile.clause_min_chars,
            on_empty_translation=translation_empty_hook(context.session, "interpreter"),
        )

        def make_vad():
            return VADProcessor(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2))
            )

        def make_user_turns():
            return UserTurnProcessor(
                user_turn_strategies=UserTurnStrategies(
                    start=[
                        VADUserTurnStartStrategy(enable_interruptions=False),
                        TranscriptionUserTurnStartStrategy(enable_interruptions=False),
                    ],
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(
                            user_speech_timeout=profile.turn_pause_secs
                        )
                    ],
                )
            )

        if directory is None:
            # Deepgram "multi" detects the language per utterance; one_way fixes the source.
            stt_language = stt_language_for(settings.languages, settings.direction)
            if settings.direction == "auto" and stt_language != "multi":
                _logger.warning(
                    "languages %s are not all in Deepgram's multi set; "
                    "transcribing %s only",
                    settings.languages,
                    stt_language,
                )
            stt = make_stt(stt_language)
            listening = [make_vad(), stt, make_user_turns()]
            speaking = [make_tts(target)]
        else:
            # One lane per participant: its own VAD, Deepgram connection (in that person's
            # language) and turn detection, fed only that participant's audio.
            from .participant_lanes import LaneTurnTagger, ParticipantLaneGate, TTSLaneGate

            lanes = []
            for lane in range(MAX_PARTICIPANTS):
                language = directory.language_for_lane(lane)
                stt = make_stt(language)
                lanes.append(
                    [
                        ParticipantLaneGate(directory, lane, stt=stt, language=language),
                        make_vad(),
                        stt,
                        make_user_turns(),
                        LaneTurnTagger(directory, lane),
                    ]
                )
            listening = [ParallelPipeline(*lanes)]
            # One TTS per listener, each publishing to that listener's own destination:
            # the translation of A's speech is synthesised once, in B's language and A's
            # voice, on B's lane only.
            speaking = [
                ParallelPipeline(
                    *[
                        [
                            TTSLaneGate(lane),
                            make_tts(
                                directory.language_for_lane(lane),
                                translation_destination(lane),
                            ),
                        ]
                        for lane in range(MAX_PARTICIPANTS)
                    ]
                )
            ]
        processors = [*listening, translation, *speaking]
        if context.session is not None:
            from .frame_events import RuntimeEventProcessor

            # The observer after TTS also feeds the translated audio to the timeline.
            processors.append(RuntimeEventProcessor(context.session, timeline=context.timeline))
        return processors


class RealtimeEngine:
    """OpenAI ``gpt-realtime-translate``: streaming speech-to-speech, one way."""

    name = "realtime"
    supports_two_way = False
    uses_own_voice = False

    def validate(self, settings: "InterpreterSettings") -> None:
        from .realtime_translate import validate_realtime_settings

        validate_realtime_settings(settings)

    def build_processors(
        self, settings: "InterpreterSettings", context: EngineContext
    ) -> "list[FrameProcessor]":
        """``VAD -> RealtimeTranslateProcessor -> events``.

        No STT, LLM or TTS service is involved: the processor streams the (resampled) input
        audio to the translation socket and emits the translated audio itself, plus the
        user/assistant ``TranscriptFinal``, ``realtime_first_audio_ms`` and the utterance
        timeline. The VAD only marks speech boundaries (it does not gate the audio, which
        is streamed continuously). Interruptions never occur. Provider voices only.
        """

        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.processors.audio.vad_processor import VADProcessor

        from .realtime_translate import RealtimeTranslateProcessor

        translate = RealtimeTranslateProcessor(
            settings=settings,
            api_key=context.config.openai_api_key,
            session=context.session,
            timeline=context.timeline,
        )
        vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)))
        processors = [vad, translate]
        if context.session is not None:
            from .frame_events import RuntimeEventProcessor

            # The processor feeds the timeline itself (it knows the utterance boundaries).
            processors.append(RuntimeEventProcessor(context.session))
        return processors


register_engine(CascadedEngine())
register_engine(RealtimeEngine())

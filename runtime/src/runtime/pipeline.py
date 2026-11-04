"""Native Pipecat pipeline factory for the LiveKit runtime."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .provider_failures import watch_provider_failures
from .turn_debug import watch_turn_debug
from .speech_providers import (
    build_llm,
    build_stt,
    build_tts,
    provider_names,
    provider_setting_types,
)


# TTS provider -> the PipelineConfig credential fields it needs.
_TTS_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "cartesia": ("cartesia_api_key", "cartesia_voice_id"),
    "elevenlabs": ("elevenlabs_api_key", "elevenlabs_voice_id"),
}
_KNOWN_TTS_FIELDS = frozenset(
    {"cartesia_api_key", "cartesia_voice_id", "elevenlabs_api_key", "elevenlabs_voice_id"}
)


def register_tts_credentials(name: str, fields: tuple[str, ...] = ()) -> None:
    """Declare which PipelineConfig fields a TTS provider uses: ``(api_key_field, voice_field)``.

    A registered TTS provider MUST be declared here (session start fails closed otherwise).
    ``()`` is the explicit declaration that the provider needs no credentials and no default
    voice; ``tts_api_key`` / ``tts_voice_id`` are then ``""``.
    """

    fields = tuple(fields)
    if len(fields) > 2 or not set(fields) <= _KNOWN_TTS_FIELDS:
        raise ValueError(f"tts credential fields must be up to two of {sorted(_KNOWN_TTS_FIELDS)}")
    _TTS_CREDENTIALS[name] = fields


@dataclass(frozen=True)
class PipelineConfig:
    livekit_url: str
    livekit_token: str
    room_name: str
    deepgram_api_key: str
    openai_api_key: str
    cartesia_api_key: str
    cartesia_voice_id: str
    system_instruction: str
    providers: dict[str, str] = field(
        default_factory=lambda: {
            "stt": "deepgram",
            "llm": "openai-compatible",
            "tts": "cartesia",
        }
    )
    provider_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    knowledge_enabled: bool = False
    mode: str = "agent"
    interpreter: dict[str, Any] | None = None
    # Two-person interpreter sessions: LiveKit identity -> {language, voice_id,
    # display_name}. None (or empty) keeps the single-participant pipeline.
    participants: dict[str, dict[str, Any]] | None = None
    # Two-person sessions whose humans cannot subscribe in the session room: each hears its
    # translation in a private listen room (``participants.listen_room_name``).
    private_listening: bool = False
    # Agent + translator (TR-040): {working_language, caller_languages}. None keeps the
    # ordinary agent pipeline.
    translation: dict[str, Any] | None = None
    # ElevenLabs credentials/default voice (ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID). Only
    # required when providers["tts"] == "elevenlabs"; the cartesia_* fields keep their names
    # and env vars so existing constructors and stored contracts do not change.
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    @property
    def tts_api_key(self) -> str:
        """API key of the selected TTS provider (``providers["tts"]``)."""

        # Unmapped / credential-less providers yield "" (never another provider's key).
        fields = _TTS_CREDENTIALS.get(self.providers.get("tts", "cartesia"), ())
        return getattr(self, fields[0]) if fields else ""

    @property
    def tts_voice_id(self) -> str:
        """Default voice of the selected TTS provider (a session ``voice_id`` already replaced it)."""

        fields = _TTS_CREDENTIALS.get(self.providers.get("tts", "cartesia"), ())
        return getattr(self, fields[1]) if len(fields) > 1 else ""

    @classmethod
    def from_environment(
        cls, session: Mapping[str, Any] | None = None
    ) -> "PipelineConfig":
        session = session or {}
        names = {
            "livekit_url": "LIVEKIT_URL",
            "livekit_token": "RUNTIME_LIVEKIT_TOKEN",
            "room_name": "LIVEKIT_ROOM_NAME",
            "deepgram_api_key": "DEEPGRAM_API_KEY",
            "openai_api_key": "OPENAI_API_KEY",
            "cartesia_api_key": "CARTESIA_API_KEY",
            "cartesia_voice_id": "CARTESIA_VOICE_ID",
            "elevenlabs_api_key": "ELEVENLABS_API_KEY",
            "elevenlabs_voice_id": "ELEVENLABS_VOICE_ID",
        }
        values = {field: os.getenv(name, "") for field, name in names.items()}
        values["room_name"] = session.get("room_name", values["room_name"])
        if session.get("room_name"):
            values["livekit_token"] = session.get("livekit_token", "")
        providers = {
            "stt": "deepgram",
            "llm": "openai-compatible",
            "tts": "cartesia",
        }
        providers.update(session.get("providers", {}))
        # The session's provider voice id replaces the default voice of the selected TTS
        # provider only.
        # Fail at session start, not mid-call: every provider must be registered.
        for kind, provider in providers.items():
            if provider not in provider_names(kind):
                raise RuntimeError(f"unsupported {kind} provider: {provider}")
        # Fail closed: a TTS provider must declare its credentials (possibly none).
        if providers["tts"] not in _TTS_CREDENTIALS:
            raise RuntimeError(f"tts provider has no credential mapping: {providers['tts']}")
        selected = _TTS_CREDENTIALS[providers["tts"]]
        if len(selected) > 1:
            values[selected[1]] = session.get("voice_id") or values[selected[1]]
        provider_settings = session.get("provider_settings") or {}
        if not isinstance(provider_settings, dict):
            raise RuntimeError("provider settings must be an object")
        for kind, settings in provider_settings.items():
            if kind not in providers or not isinstance(settings, dict):
                raise RuntimeError(f"unsupported provider settings section: {kind}")
            allowed = provider_setting_types(kind, providers[kind])
            for name, value in settings.items():
                expected_type = allowed.get(name)
                if expected_type is None:
                    raise RuntimeError(f"unsupported {kind} provider setting: {name}")
                if isinstance(value, bool) and expected_type is not bool:
                    valid = False
                else:
                    valid = isinstance(value, expected_type)
                if not valid:
                    raise RuntimeError(f"invalid {kind} provider setting type: {name}")
        tools = session.get("tools") or []
        if not isinstance(tools, list):
            raise RuntimeError("session tools must be a list")
        mode = session.get("mode") or "agent"
        if mode not in {"agent", "interpreter"}:
            raise RuntimeError(f"unsupported session mode: {mode}")
        interpreter = session.get("interpreter") if mode == "interpreter" else None
        engine = None
        if mode == "interpreter":
            from .interpreter import InterpreterSettings
            from .translation_engine import get_engine

            engine = get_engine(InterpreterSettings.from_mapping(interpreter).engine)
        participants = session.get("participants") or None
        if participants is not None:
            if not isinstance(participants, dict) or not all(
                isinstance(identity, str) and isinstance(spec, dict) and spec.get("language")
                for identity, spec in participants.items()
            ):
                raise RuntimeError("participants must map identities to {language, ...}")
            if engine is None or not engine.supports_two_way:
                raise RuntimeError("participants require a cascaded interpreter")
            participants = {identity: dict(spec) for identity, spec in participants.items()}
        private_listening = bool(session.get("private_listening", False))
        if private_listening and participants is None:
            raise RuntimeError("private_listening requires participants")
        if not private_listening and any(
            spec.get("phone_room") for spec in (participants or {}).values()
        ):
            raise RuntimeError("phone_room participants require private_listening")
        translation = session.get("translation") or None
        if translation is not None:
            from .agent_translation import AgentTranslationSettings

            if mode != "agent":
                raise RuntimeError("translation is only supported for agent sessions")
            AgentTranslationSettings.from_mapping(translation)
            translation = dict(translation)
        if not values["livekit_token"]:
            values["livekit_token"] = create_runtime_token(values["room_name"])
        # An engine that speaks with the provider's voice (realtime) needs only LiveKit
        # and OpenAI: no STT/TTS credentials.
        # Only the selected TTS provider's credentials are required.
        tts_fields = {"cartesia_api_key", "cartesia_voice_id", "elevenlabs_api_key", "elevenlabs_voice_id"}
        unused = {"deepgram_api_key"} | tts_fields
        required = {
            field: name
            for field, name in names.items()
            if not (engine is not None and not engine.uses_own_voice and field in unused)
            and not (field in tts_fields and field not in selected)
        }
        missing = [name for field, name in required.items() if not values[field]]
        if missing:
            raise RuntimeError(
                f"runtime provider configuration is missing: {', '.join(missing)}"
            )
        return cls(
            **values,
            system_instruction=session.get(
                "system_prompt",
                "You are a helpful multilingual voice assistant. Respond briefly and naturally.",
            ),
            providers=providers,
            provider_settings=provider_settings,
            tools=tools,
            knowledge_enabled=bool(session.get("knowledge_enabled", False)),
            mode=mode,
            interpreter=dict(interpreter) if interpreter else None,
            participants=participants,
            private_listening=private_listening,
            translation=translation,
        )


def create_runtime_token(room_name: str) -> str:
    """Mint a short-lived agent token when the platform did not pass one."""

    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    if not key or not secret or not room_name:
        return ""
    try:
        from livekit import api
    except ImportError as exc:
        raise RuntimeError(
            "install the livekit-api package to mint a runtime token"
        ) from exc
    identity = os.getenv("RUNTIME_IDENTITY", f"voice-labs-agent-{room_name}")
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_kind("agent")
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


def listen_room_connector(config: PipelineConfig, room_name_for=None):
    """Connect the Runtime to one participant's private room (their listen room, or with
    ``room_name_for=phone_room_name`` the room a phone is dialled into)."""

    from livekit import rtc

    from .participants import listen_room_name

    async def connect(identity: str):
        name = (room_name_for or listen_room_name)(config.room_name, identity)
        room = rtc.Room()
        await room.connect(config.livekit_url, create_runtime_token(name))
        return room

    return connect


def require_pipecat() -> None:
    try:
        import pipecat  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "install the runtime voice extras before starting a Pipecat pipeline"
        ) from exc


def build_livekit_pipeline(config: PipelineConfig, session=None):
    """Build the native LiveKit -> STT -> LLM -> TTS pipeline.

    The returned objects are intentionally Pipecat-specific and stay inside the runtime
    process. Platform and browser contracts use IDs, events, and audio transport only.
    """

    require_pipecat()
    if config.mode == "interpreter":
        return build_interpreter_pipeline(config, session)
    from .frame_events import RuntimeEventProcessor

    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
        LocalSmartTurnAnalyzerV3,
    )
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    transport = LiveKitTransport(
        url=config.livekit_url,
        token=config.livekit_token,
        room_name=config.room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )
    translation = None
    if config.translation:
        from .agent_translation import AgentTranslationSettings

        translation = AgentTranslationSettings.from_mapping(config.translation)
    # "multi" detects the caller's language per utterance; otherwise it is fixed.
    stt_language = translation.stt_language if translation is not None else None
    keyterms = ()
    if translation is not None:
        from .glossary import glossary_keyterms

        language = None if stt_language == "multi" else stt_language
        keyterms = glossary_keyterms(translation, language)
    stt = build_stt(config, stt_language, keyterms)
    tool_runtime = None
    system_instruction = config.system_instruction
    if translation is not None:
        system_instruction = f"{system_instruction}\n\n{translation.instruction}"
    if session is not None:
        from .tool_runtime import INSTRUCTION_SUFFIX, build_tool_runtime

        tool_runtime = build_tool_runtime(
            config.tools, config.knowledge_enabled, session
        )
        if tool_runtime:
            system_instruction = f"{system_instruction}\n\n{INSTRUCTION_SUFFIX}"
    llm = build_llm(config, system_instruction)
    tts = build_tts(
        config, translation.initial_tts_language if translation is not None else None
    )
    if tool_runtime:
        tool_runtime.register(llm)
        context = LLMContext(tools=tool_runtime.tools_schema())
    else:
        context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )
    if translation is not None:
        processors = agent_translation_processors(
            config,
            session,
            translation,
            transport=transport,
            stt=stt,
            user_aggregator=user_aggregator,
            llm=llm,
            tts=tts,
            assistant_aggregator=assistant_aggregator,
        )
    else:
        processors = [transport.input(), stt]
        if session is not None:
            processors.append(RuntimeEventProcessor(session, control_events=False))
        processors.append(user_aggregator)
        processors.append(llm)
        if session is not None:
            processors.append(
                RuntimeEventProcessor(session, assistant_output=True, control_events=False)
            )
        processors.append(answer_guard(session, _agent_language(config, session)))
        processors.append(tts)
        if session is not None:
            processors.append(RuntimeEventProcessor(session))
        processors.extend([transport.output(), assistant_aggregator])
    pipeline = Pipeline(processors)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    # PR-30: a dead or out-of-credit provider ends the call cleanly instead of going silent.
    # ST-10a: also attaches the stall watchdog (STT/LLM/TTS hangs raise no error).
    watch_provider_failures(worker, session, config.providers)
    # ST-02a: one TurnDebug event per turn (agent mode only; not the interpreter pipelines).
    watch_turn_debug(worker, session, config.providers)
    return transport, worker


def _agent_language(config: PipelineConfig, session):
    """The language the answer guard's REPEAT line is spoken in (non-translation agent).

    The agent has no declared caller language; use what is configured for speech: the TTS
    language setting, else the STT one (``multi`` means undetermined). Unknown: English.
    """

    settings = config.provider_settings
    for kind in ("tts", "stt"):
        language = (settings.get(kind) or {}).get("language")
        if isinstance(language, str) and language and language != "multi":
            return language
    return getattr(session, "caller_language", None)


def answer_guard(session, language):
    """ST-12: the answer guard before TTS; reports ``AnswerEmpty``/``AnswerTruncated``."""

    from .answer_guard import MAX_SPOKEN_CHARS, AnswerGuard
    from .guard_events import answer_empty_hook, answer_truncated_hook

    return AnswerGuard(
        language=language,
        max_spoken_chars=MAX_SPOKEN_CHARS,
        on_empty=answer_empty_hook(session),
        on_truncated=answer_truncated_hook(session, MAX_SPOKEN_CHARS),
    )


def agent_translation_processors(
    config: PipelineConfig,
    session,
    translation,
    *,
    transport,
    stt,
    user_aggregator,
    llm,
    tts,
    assistant_aggregator,
) -> list:
    """Processor order of the agent + translator pipeline (TR-040).

    ``transport.input -> STT -> tap -> user aggregator -> InboundTranslator -> LLM ->
    OutboundTranslator -> TTS -> events -> transport.output -> assistant aggregator``.
    The translators emit the user and assistant ``TranscriptFinal`` events themselves
    (with the original and translated text), so the lifecycle/latency
    ``RuntimeEventProcessor`` after TTS has its transcript events disabled. See
    ``agent_translation`` for how the assistant context keeps the working language.
    """

    from .agent_translation import (
        CallerLanguageState,
        CallerLanguageTap,
        InboundTranslator,
        OutboundTranslator,
    )
    from .frame_events import RuntimeEventProcessor
    from .agent_translation import AGENT_TRANSLATION_MODEL
    from .guard_events import translation_empty_hook
    from .interpreter import OpenAITranslator

    # A small fast model keeps the two extra round trips short; the LLM's own model is
    # deliberately not reused. Override with provider_settings.llm.translation_model.
    model = config.provider_settings.get("llm", {}).get("translation_model") or AGENT_TRANSLATION_MODEL
    translator = OpenAITranslator(
        api_key=config.openai_api_key,
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    from .timeline import TimelineTracker

    # RM-003: caller speech start -> first audible translated reply.
    timeline = TimelineTracker(session, "agent_translation") if session is not None else None
    state = CallerLanguageState(translation, timeline=timeline)
    if session is not None:
        # ST-10b: the failure apology is spoken in the caller's current language.
        session.caller_language = lambda: state.language
    processors = [
        transport.input(),
        stt,
        CallerLanguageTap(state),
        user_aggregator,
        InboundTranslator(
            state=state,
            translate=translator,
            model=model,
            session=session,
            warm_up=translator.warm_up,
            on_empty_translation=translation_empty_hook(session, "inbound"),
        ),
        llm,
        OutboundTranslator(
            state=state,
            translate=translator,
            model=model,
            session=session,
            on_empty_translation=translation_empty_hook(session, "outbound"),
        ),
        # The outbound translator's spoken output is in the caller's current language.
        answer_guard(session, lambda: state.language),
        tts,
    ]
    if session is not None:
        processors.append(
            RuntimeEventProcessor(session, transcript_events=False, timeline=timeline)
        )
    processors.extend([transport.output(), assistant_aggregator])
    return processors


def participant_directory(config: PipelineConfig, session, languages):
    """The live participant mapping, or None for a single-participant session.

    A running session shares its :class:`RuntimeSession` directory with the HTTP layer so
    a participant added later reaches the pipeline; otherwise one is built from the
    config.
    """

    from .participants import ParticipantDirectory

    if not config.participants:
        return None
    shared = getattr(session, "participants", None)
    if isinstance(shared, ParticipantDirectory):
        return shared
    return ParticipantDirectory.from_mapping(languages, config.participants)


def build_interpreter_pipeline(config: PipelineConfig, session=None):
    """Build the interpreter pipeline of ``interpreter.engine`` (see ``translation_engine``).

    ``transport.input -> engine processors -> transport.output``. The engine is looked up
    in the registry; it builds everything between the transport's input and output (the
    cascaded engine: VAD, STT, turns, translation, TTS and the event observer; the
    realtime engine: VAD, streaming translation and the event observer). Interruptions
    are never enabled, so a translation is never cut off.

    Two-person sessions (``config.participants``, TR-020) need an engine with
    ``supports_two_way``; the transport then resamples each participant separately and
    publishes one output track per listener (RM-001/RM-002, see ``participant_lanes`` and
    ``audio_routing``). Every pipeline records the utterance timeline (RM-003) when there
    is a session to report to.
    """

    require_pipecat()
    from .interpreter import InterpreterSettings
    from .timeline import TimelineTracker
    from .translation_engine import EngineContext, get_engine

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

    settings = InterpreterSettings.from_mapping(config.interpreter)
    engine = get_engine(settings.engine)
    directory = participant_directory(config, session, settings.languages)
    if directory is None:
        transport = LiveKitTransport(
            url=config.livekit_url,
            token=config.livekit_token,
            room_name=config.room_name,
            params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
        )
    else:
        # Per-participant resampling (the stock input mixes interleaved speakers) and one
        # output track per listener (RM-001/RM-002): each destination is a lane whose TTS
        # audio is published only to that listener.
        from .audio_routing import ListenerAudioRouter, LiveKitApiSettings
        from .participant_lanes import PerParticipantLiveKitTransport
        from .participants import MAX_PARTICIPANTS, phone_room_name, translation_destination

        router = ListenerAudioRouter(
            directory,
            hear_original=settings.hear_original,
            api_settings=LiveKitApiSettings.from_environment(config.livekit_url),
            listen_room_factory=(
                listen_room_connector(config)
                if config.private_listening and not settings.hear_original
                else None
            ),
            phone_room_factory=(
                listen_room_connector(config, phone_room_name)
                if config.private_listening and not settings.hear_original
                else None
            ),
        )
        transport = PerParticipantLiveKitTransport(
            url=config.livekit_url,
            token=config.livekit_token,
            room_name=config.room_name,
            params=LiveKitParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_destinations=[
                    translation_destination(lane) for lane in range(MAX_PARTICIPANTS)
                ],
            ),
            router=router,
        )
    timeline = TimelineTracker(session, engine.name) if session is not None else None
    context = EngineContext(config=config, session=session, directory=directory, timeline=timeline)
    processors = [transport.input(), *engine.build_processors(settings, context)]
    processors.append(transport.output())
    worker = PipelineWorker(
        Pipeline(processors),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    # PR-30: a dead or out-of-credit provider ends the call cleanly instead of going silent.
    # ST-10a: plus the stall watchdog; the realtime engine has no STT/LLM/TTS to watch.
    watch_provider_failures(
        worker, session, config.providers, stall_watchdog=settings.engine != "realtime"
    )
    return transport, worker


async def run_livekit_pipeline(config: PipelineConfig) -> None:
    """Run one configured pipeline until its worker ends."""

    from pipecat.workers.runner import WorkerRunner

    _, worker = build_livekit_pipeline(config)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

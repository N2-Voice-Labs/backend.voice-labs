"""Native Pipecat pipeline factory for the LiveKit runtime."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .speech_providers import (
    build_llm,
    build_stt,
    build_tts,
    provider_names,
    provider_setting_types,
)


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
    # Agent + translator (TR-040): {working_language, caller_languages}. None keeps the
    # ordinary agent pipeline.
    translation: dict[str, Any] | None = None

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
        }
        values = {field: os.getenv(name, "") for field, name in names.items()}
        values["room_name"] = session.get("room_name", values["room_name"])
        if session.get("room_name"):
            values["livekit_token"] = session.get("livekit_token", "")
        values["cartesia_voice_id"] = (
            session.get("voice_id") or values["cartesia_voice_id"]
        )
        providers = {
            "stt": "deepgram",
            "llm": "openai-compatible",
            "tts": "cartesia",
        }
        providers.update(session.get("providers", {}))
        # Fail at session start, not mid-call: every provider must be registered.
        for kind, provider in providers.items():
            if provider not in provider_names(kind):
                raise RuntimeError(f"unsupported {kind} provider: {provider}")
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
        unused = {"deepgram_api_key", "cartesia_api_key", "cartesia_voice_id"}
        required = {
            field: name
            for field, name in names.items()
            if not (engine is not None and not engine.uses_own_voice and field in unused)
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
        processors.append(tts)
        if session is not None:
            processors.append(RuntimeEventProcessor(session))
        processors.extend([transport.output(), assistant_aggregator])
    pipeline = Pipeline(processors)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    return transport, worker


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
        ),
        llm,
        OutboundTranslator(state=state, translate=translator, model=model, session=session),
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
        from .participants import MAX_PARTICIPANTS, translation_destination

        router = ListenerAudioRouter(
            directory,
            hear_original=settings.hear_original,
            api_settings=LiveKitApiSettings.from_environment(config.livekit_url),
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
    return transport, worker


async def run_livekit_pipeline(config: PipelineConfig) -> None:
    """Run one configured pipeline until its worker ends."""

    from pipecat.workers.runner import WorkerRunner

    _, worker = build_livekit_pipeline(config)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

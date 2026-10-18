"""Native Pipecat pipeline factory for the LiveKit runtime."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any


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
        supported = {
            "stt": {"deepgram"},
            "llm": {"openai", "openai-compatible"},
            "tts": {"cartesia"},
        }
        for kind, provider in providers.items():
            if provider not in supported.get(kind, set()):
                raise RuntimeError(f"unsupported {kind} provider: {provider}")
        provider_setting_types: dict[str, dict[str, type | tuple[type, ...]]] = {
            "stt": {
                "model": str,
                "language": str,
                "interim_results": bool,
                "smart_format": bool,
            },
            "llm": {
                "model": str,
                "temperature": (int, float),
                "max_tokens": int,
                "top_p": (int, float),
                # Agent + translator: model of the two translation legs (not the LLM's).
                "translation_model": str,
            },
            "tts": {"model": str, "language": str},
        }
        provider_settings = session.get("provider_settings") or {}
        if not isinstance(provider_settings, dict):
            raise RuntimeError("provider settings must be an object")
        for kind, settings in provider_settings.items():
            if kind not in provider_setting_types or not isinstance(settings, dict):
                raise RuntimeError(f"unsupported provider settings section: {kind}")
            for name, value in settings.items():
                expected_type = provider_setting_types[kind].get(name)
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
        engine = "cascaded"
        if mode == "interpreter":
            from .interpreter import InterpreterSettings

            engine = InterpreterSettings.from_mapping(interpreter).engine
        participants = session.get("participants") or None
        if participants is not None:
            if not isinstance(participants, dict) or not all(
                isinstance(identity, str) and isinstance(spec, dict) and spec.get("language")
                for identity, spec in participants.items()
            ):
                raise RuntimeError("participants must map identities to {language, ...}")
            if mode != "interpreter" or engine != "cascaded":
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
        # The realtime interpreter engine needs only LiveKit and OpenAI (no STT/TTS).
        unused = {"deepgram_api_key", "cartesia_api_key", "cartesia_voice_id"}
        required = {
            field: name
            for field, name in names.items()
            if not (engine == "realtime" and field in unused)
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
        if (config.interpreter or {}).get("engine") == "realtime":
            return build_realtime_interpreter_pipeline(config, session)
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
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.llm import OpenAILLMService
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
    stt_services = {"deepgram": DeepgramSTTService}
    llm_services = {"openai": OpenAILLMService, "openai-compatible": OpenAILLMService}
    tts_services = {"cartesia": CartesiaTTSService}
    translation = None
    if config.translation:
        from .agent_translation import AgentTranslationSettings

        translation = AgentTranslationSettings.from_mapping(config.translation)
    stt_settings = config.provider_settings.get("stt", {}).copy()
    if translation is not None:
        # "multi" detects the caller's language per utterance; otherwise it is fixed.
        stt_settings["language"] = translation.stt_language
    stt = stt_services[config.providers["stt"]](
        api_key=config.deepgram_api_key,
        settings=DeepgramSTTService.Settings(**stt_settings),
    )
    llm_service = llm_services[config.providers["llm"]]
    tts_service = tts_services[config.providers["tts"]]
    llm_settings = config.provider_settings.get("llm", {}).copy()
    llm_settings.pop("translation_model", None)  # consumed by the translators only
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
    llm_settings["system_instruction"] = system_instruction
    llm = llm_service(
        api_key=config.openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        settings=llm_service.Settings(**llm_settings),
    )
    tts_settings = config.provider_settings.get("tts", {}).copy()
    tts_settings["voice"] = config.cartesia_voice_id
    if translation is not None:
        tts_settings["language"] = translation.initial_tts_language
    tts = tts_service(
        api_key=config.cartesia_api_key,
        settings=tts_service.Settings(**tts_settings),
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
    state = CallerLanguageState(translation)
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
        processors.append(RuntimeEventProcessor(session, transcript_events=False))
    processors.extend([transport.output(), assistant_aggregator])
    return processors


INTERPRETER_TURN_PAUSE_SECS = 0.3


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
    """Build the cascaded interpreter pipeline (no LLM context, tools or barge-in).

    ``transport.input -> VAD -> Deepgram STT -> user turn processor -> translation ->
    Cartesia TTS -> events -> transport.output``. Interruptions are disabled by
    constructing the turn start strategies with ``enable_interruptions=False``, so no
    ``InterruptionFrame`` is ever broadcast and a translation is never cut off.
    ``TranslationProcessor`` consumes the transcripts and emits the user and
    assistant ``TranscriptFinal`` events itself, so the single
    ``RuntimeEventProcessor`` (after TTS) only contributes lifecycle and latency events.

    Two-person sessions (``config.participants``, TR-020) replace the single
    ``VAD -> STT -> user turn`` stage with a native ``ParallelPipeline`` of one lane per
    participant (see ``participant_lanes``): LiveKit delivers each participant's audio
    separately (``UserAudioRawFrame.user_id``), so every lane gets its own VAD, Deepgram
    connection in that person's declared language, and turn detection. The translator
    then speaks each translation in the speaker's own voice.
    """

    require_pipecat()
    from .participants import MAX_PARTICIPANTS
    from .interpreter import (
        DEFAULT_TRANSLATION_MODEL,
        InterpreterSettings,
        OpenAITranslator,
        TranslationProcessor,
    )

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.cartesia.tts import CartesiaTTSService
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
    from pipecat.turns.user_start import (
        TranscriptionUserTurnStartStrategy,
        VADUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
    from pipecat.turns.user_turn_processor import UserTurnProcessor
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    settings = InterpreterSettings.from_mapping(config.interpreter)
    source, target = settings.languages
    directory = participant_directory(config, session, settings.languages)
    if directory is not None:
        # Per-participant resampling: the stock input mixes interleaved speakers.
        from .participant_lanes import PerParticipantLiveKitTransport as LiveKitTransport
    transport = LiveKitTransport(
        url=config.livekit_url,
        token=config.livekit_token,
        room_name=config.room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    def make_stt(language: str):
        stt_settings = config.provider_settings.get("stt", {}).copy()
        stt_settings["language"] = language
        return DeepgramSTTService(
            api_key=config.deepgram_api_key,
            settings=DeepgramSTTService.Settings(**stt_settings),
        )

    tts_settings = config.provider_settings.get("tts", {}).copy()
    tts_settings["voice"] = config.cartesia_voice_id
    tts_settings["language"] = target
    tts = CartesiaTTSService(
        api_key=config.cartesia_api_key,
        settings=CartesiaTTSService.Settings(**tts_settings),
    )
    model = config.provider_settings.get("llm", {}).get("model") or DEFAULT_TRANSLATION_MODEL
    translation = TranslationProcessor(
        settings=settings,
        translate=OpenAITranslator(
            api_key=config.openai_api_key,
            model=model,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        ),
        model=model,
        session=session,
        initial_tts_language=target,
        participants=directory,
        default_voice_id=config.cartesia_voice_id,
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
                        user_speech_timeout=INTERPRETER_TURN_PAUSE_SECS
                    )
                ],
            )
        )

    if directory is None:
        # Deepgram "multi" detects the language per utterance; one_way fixes the source.
        stt = make_stt("multi" if settings.direction == "auto" else source)
        listening = [make_vad(), stt, make_user_turns()]
    else:
        # One lane per participant: its own VAD, Deepgram connection (in that person's
        # language) and turn detection, fed only that participant's audio.
        from pipecat.pipeline.parallel_pipeline import ParallelPipeline

        from .participant_lanes import LaneTurnTagger, ParticipantLaneGate

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
    processors = [transport.input(), *listening, translation, tts]
    if session is not None:
        from .frame_events import RuntimeEventProcessor

        processors.append(RuntimeEventProcessor(session))
    processors.append(transport.output())
    worker = PipelineWorker(
        Pipeline(processors),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    return transport, worker


def build_realtime_interpreter_pipeline(config: PipelineConfig, session=None):
    """Build the streaming interpreter pipeline (OpenAI ``gpt-realtime-translate``).

    ``transport.input -> VAD -> RealtimeTranslateProcessor -> events -> transport.output``.
    No STT, LLM or TTS service is involved: the processor streams the (resampled) input
    audio to the translation socket and emits the translated audio itself, plus the
    user/assistant ``TranscriptFinal`` and ``realtime_first_audio_ms`` latency events. The
    VAD only marks speech boundaries (it does not gate the audio, which is streamed
    continuously); ``RuntimeEventProcessor`` adds lifecycle events and
    ``turn_to_first_audio``. Interruptions never occur. Provider voices only: the agent's
    voice profile is not used.
    """

    require_pipecat()
    from .realtime_translate import RealtimeTranslateProcessor

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

    from .interpreter import InterpreterSettings

    settings = InterpreterSettings.from_mapping(config.interpreter)
    transport = LiveKitTransport(
        url=config.livekit_url,
        token=config.livekit_token,
        room_name=config.room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    translate = RealtimeTranslateProcessor(
        settings=settings, api_key=config.openai_api_key, session=session
    )
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)))
    processors = [transport.input(), vad, translate]
    if session is not None:
        from .frame_events import RuntimeEventProcessor

        processors.append(RuntimeEventProcessor(session))
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

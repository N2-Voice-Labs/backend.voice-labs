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
        if not values["livekit_token"]:
            values["livekit_token"] = create_runtime_token(values["room_name"])
        missing = [name for field, name in names.items() if not values[field]]
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
    stt = stt_services[config.providers["stt"]](
        api_key=config.deepgram_api_key,
        settings=DeepgramSTTService.Settings(**config.provider_settings.get("stt", {})),
    )
    llm_service = llm_services[config.providers["llm"]]
    tts_service = tts_services[config.providers["tts"]]
    llm_settings = config.provider_settings.get("llm", {}).copy()
    llm_settings["system_instruction"] = config.system_instruction
    llm = llm_service(
        api_key=config.openai_api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        settings=llm_service.Settings(**llm_settings),
    )
    tts_settings = config.provider_settings.get("tts", {}).copy()
    tts_settings["voice"] = config.cartesia_voice_id
    tts = tts_service(
        api_key=config.cartesia_api_key,
        settings=tts_service.Settings(**tts_settings),
    )
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


async def run_livekit_pipeline(config: PipelineConfig) -> None:
    """Run one configured pipeline until its worker ends."""

    from pipecat.workers.runner import WorkerRunner

    _, worker = build_livekit_pipeline(config)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()

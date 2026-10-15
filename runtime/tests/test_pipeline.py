import base64
import json
from uuid import uuid4

import pytest

from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession


def test_pipeline_config_reports_missing_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        PipelineConfig.from_environment()


def test_pipeline_config_accepts_explicit_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "LIVEKIT_URL": "ws://localhost:7880",
        "RUNTIME_LIVEKIT_TOKEN": "token",
        "LIVEKIT_ROOM_NAME": "room",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    config = PipelineConfig.from_environment(
        {
            "room_name": "room",
            "livekit_token": "token",
            "providers": {
                "stt": "deepgram",
                "llm": "openai-compatible",
                "tts": "cartesia",
            },
            "provider_settings": {
                "stt": {"model": "nova-3", "language": "multi"},
                "llm": {"model": "gpt-4o-mini", "temperature": 0.4},
                "tts": {"model": "sonic-2", "language": "en"},
            },
        }
    )
    assert config.room_name == "room"
    assert config.providers == {
        "stt": "deepgram",
        "llm": "openai-compatible",
        "tts": "cartesia",
    }
    assert config.provider_settings == {
        "stt": {"model": "nova-3", "language": "multi"},
        "llm": {"model": "gpt-4o-mini", "temperature": 0.4},
        "tts": {"model": "sonic-2", "language": "en"},
    }


def test_pipeline_config_rejects_provider_without_a_native_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "LIVEKIT_URL": "ws://localhost:7880",
        "RUNTIME_LIVEKIT_TOKEN": "token",
        "LIVEKIT_ROOM_NAME": "room",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(RuntimeError, match="unsupported stt provider: faster-whisper"):
        PipelineConfig.from_environment(
            {
                "room_name": "room",
                "livekit_token": "token",
                "providers": {"stt": "faster-whisper"},
            }
        )


def test_pipeline_config_rejects_unsupported_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "LIVEKIT_URL": "ws://localhost:7880",
        "RUNTIME_LIVEKIT_TOKEN": "token",
        "LIVEKIT_ROOM_NAME": "room",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(RuntimeError, match="unsupported stt provider setting: api_key"):
        PipelineConfig.from_environment(
            {
                "room_name": "room",
                "livekit_token": "token",
                "provider_settings": {"stt": {"api_key": "not-a-secret"}},
            }
        )


def test_pipeline_config_mints_runtime_token_from_livekit_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_room = "requested-room-83"
    values = {
        "LIVEKIT_URL": "ws://localhost:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
        "LIVEKIT_ROOM_NAME": requested_room,
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("RUNTIME_LIVEKIT_TOKEN", raising=False)
    config = PipelineConfig.from_environment()
    claims = json.loads(
        base64.urlsafe_b64decode(config.livekit_token.split(".")[1] + "==")
    )

    assert claims["video"]["roomJoin"] is True
    assert claims["video"]["room"] == requested_room
    assert claims["video"]["canPublish"] is True
    assert claims["video"]["canSubscribe"] is True


def test_livekit_transport_receives_requested_room_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero
    from pipecat.services.cartesia import tts as cartesia_tts
    from pipecat.services.deepgram import stt as deepgram_stt
    from pipecat.services.openai import llm as openai_llm
    from pipecat.transports.livekit import transport as livekit_transport

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer
    )
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)
    real_transport = livekit_transport.LiveKitTransport
    transport_arguments: dict[str, object] = {}

    def capture_transport_arguments(*args, **kwargs):
        transport_arguments.update(kwargs)
        return real_transport(*args, **kwargs)

    monkeypatch.setattr(
        livekit_transport, "LiveKitTransport", capture_transport_arguments
    )
    provider_settings: dict[str, object] = {}
    for name, module, service_name in (
        ("stt", deepgram_stt, "DeepgramSTTService"),
        ("llm", openai_llm, "OpenAILLMService"),
        ("tts", cartesia_tts, "CartesiaTTSService"),
    ):
        service = getattr(module, service_name)

        def capture_settings(*args, _name=name, _service=service, **kwargs):
            provider_settings[_name] = kwargs["settings"]
            return _service(*args, **kwargs)

        capture_settings.Settings = service.Settings
        monkeypatch.setattr(module, service_name, capture_settings)
    requested_room = "requested-room-83"
    config = PipelineConfig(
        "ws://localhost:7880",
        "runtime-token",
        requested_room,
        "deepgram",
        "openai",
        "cartesia",
        "voice",
        "prompt",
        provider_settings={
            "stt": {"model": "nova-3", "language": "multi"},
            "llm": {"model": "gpt-4o-mini", "temperature": 0.4},
            "tts": {"model": "sonic-2", "language": "en"},
        },
    )

    build_livekit_pipeline(config)

    assert transport_arguments["room_name"] == requested_room
    assert provider_settings["stt"].model == "nova-3"
    assert provider_settings["stt"].language == "multi"
    assert provider_settings["llm"].model == "gpt-4o-mini"
    assert provider_settings["llm"].temperature == 0.4
    assert provider_settings["tts"].model == "sonic-2"
    assert provider_settings["tts"].language == "en"


def test_pipeline_places_frame_observers_at_pipecat_consumption_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero
    from pipecat.pipeline.pipeline import Pipeline

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer
    )
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)
    config = PipelineConfig(
        "ws://localhost:7880",
        "token",
        "room",
        "deepgram",
        "openai",
        "cartesia",
        "voice",
        "prompt",
        providers={
            "stt": "deepgram",
            "llm": "openai-compatible",
            "tts": "cartesia",
        },
        provider_settings={
            "stt": {"model": "nova-3", "language": "multi", "interim_results": True},
            "llm": {"model": "gpt-4o-mini", "temperature": 0.4, "max_tokens": 128},
            "tts": {"model": "sonic-2", "language": "en"},
        },
    )
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    _, task = build_livekit_pipeline(config, session)
    pipeline = next(
        processor
        for processor in task._pipeline._processors
        if isinstance(processor, Pipeline)
    )
    processors = pipeline._processors
    observer_indexes = [
        index
        for index, processor in enumerate(processors)
        if type(processor).__name__ == "RuntimeEventProcessor"
    ]
    observers = [processors[index] for index in observer_indexes]

    def index_named(name: str) -> int:
        return next(
            index
            for index, processor in enumerate(processors)
            if type(processor).__name__ == name
        )

    stt_index = index_named("DeepgramSTTService")
    llm_index = index_named("OpenAILLMService")
    tts_index = index_named("CartesiaTTSService")
    output_index = index_named("LiveKitOutputTransport")

    assert len(observers) == 3
    assert observer_indexes[0] == stt_index + 1
    assert observer_indexes[1] == llm_index + 1
    assert tts_index < observer_indexes[2] < output_index
    assert [
        (observer._assistant_output, observer._control_events) for observer in observers
    ] == [(False, False), (True, False), (False, True)]


def _pipeline_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        "ws://localhost:7880",
        "token",
        "room",
        "deepgram",
        "openai",
        "cartesia",
        "voice",
        "prompt",
        **kwargs,
    )


def test_pipeline_config_carries_tools_and_knowledge_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "LIVEKIT_URL": "ws://localhost:7880",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }.items():
        monkeypatch.setenv(key, value)
    tools = [{"name": "lookup", "tool_type": "http"}]
    config = PipelineConfig.from_environment(
        {
            "room_name": "room",
            "livekit_token": "token",
            "tools": tools,
            "knowledge_enabled": True,
        }
    )
    assert config.tools == tools
    assert config.knowledge_enabled is True
    legacy = PipelineConfig.from_environment({"room_name": "room", "livekit_token": "t"})
    assert legacy.tools == [] and legacy.knowledge_enabled is False


def test_pipeline_registers_tools_and_extends_system_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero
    from pipecat.processors.aggregators import llm_context
    from pipecat.services.openai import llm as openai_llm

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer
    )
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)
    captured: dict[str, object] = {}
    real_context = llm_context.LLMContext

    class capture_context(real_context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["context"] = self

    real_llm = openai_llm.OpenAILLMService

    def capture_llm(*args, **kwargs):
        service = real_llm(*args, **kwargs)
        captured["llm"] = service
        captured["settings"] = kwargs["settings"]
        return service

    capture_llm.Settings = real_llm.Settings
    monkeypatch.setattr(llm_context, "LLMContext", capture_context)
    monkeypatch.setattr(openai_llm, "OpenAILLMService", capture_llm)
    config = _pipeline_config(
        tools=[
            {
                "name": "lookup_order",
                "description": "Look up an order",
                "tool_type": "http",
                "input_schema": {"type": "object", "properties": {}},
                "configuration": {"url": "https://orders.test/lookup"},
            }
        ],
        knowledge_enabled=True,
    )

    build_livekit_pipeline(config, RuntimeSession(uuid4(), uuid4(), uuid4()))

    names = {tool.name for tool in captured["context"].tools.standard_tools}
    assert names == {"record_outcome", "search_knowledge", "lookup_order"}
    assert set(captured["llm"]._functions) >= names
    assert captured["settings"].system_instruction.endswith("call record_outcome.")
    assert captured["settings"].system_instruction.startswith("prompt")


def test_pipeline_without_session_registers_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pipecat")
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero
    from pipecat.processors.aggregators import llm_context

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer
    )
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)
    captured: dict[str, object] = {}
    real_context = llm_context.LLMContext

    class capture_context(real_context):
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(llm_context, "LLMContext", capture_context)
    build_livekit_pipeline(_pipeline_config())
    assert "tools" not in captured["kwargs"]

"""NS-06: the STT, LLM and TTS provider registries."""

import pytest

pytest.importorskip("pipecat")

from src.runtime import speech_providers
from src.runtime import pipeline as pipeline_module
from src.runtime.pipeline import PipelineConfig, register_tts_credentials
from src.runtime.speech_providers import (
    build_llm,
    build_stt,
    build_tts,
    llm_names,
    provider_capabilities,
    register_llm,
    register_stt,
    register_tts,
    stt_names,
    tts_names,
)


def _config(**overrides) -> PipelineConfig:
    values = dict(
        livekit_url="ws://localhost:7880",
        livekit_token="token",
        room_name="room",
        deepgram_api_key="dg-key",
        openai_api_key="oa-key",
        cartesia_api_key="ct-key",
        cartesia_voice_id="voice-1",
        system_instruction="prompt",
    )
    values.update(overrides)
    return PipelineConfig(**values)


class Recorder:
    """Stands in for a Pipecat service class: records constructor arguments."""

    class Settings(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def restore_registries():
    """Undo any ``register_*`` made by a test (factories, settings and capabilities)."""

    tables = [*speech_providers._REGISTRIES.values(), speech_providers._SETTING_TYPES,
              speech_providers._CAPABILITIES, pipeline_module._TTS_CREDENTIALS]
    saved = [dict(table) for table in tables]
    yield
    for table, snapshot in zip(tables, saved):
        table.clear()
        table.update(snapshot)


def test_builtin_providers_are_registered() -> None:
    assert stt_names()[:1] == ("deepgram",)
    assert tts_names()[:1] == ("cartesia",)


def test_unknown_providers_fail_with_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match="unsupported stt provider: whisper"):
        build_stt(_config(providers={"stt": "whisper", "llm": "openai", "tts": "cartesia"}))
    with pytest.raises(RuntimeError, match="unsupported tts provider: espeak"):
        build_tts(_config(providers={"stt": "deepgram", "llm": "openai", "tts": "espeak"}))


def test_session_start_rejects_unknown_providers_and_follows_the_registry(
    monkeypatch, restore_registries
) -> None:
    for key, value in {
        "LIVEKIT_URL": "ws://localhost:7880",
        "RUNTIME_LIVEKIT_TOKEN": "token",
        "LIVEKIT_ROOM_NAME": "room",
        "DEEPGRAM_API_KEY": "dg",
        "OPENAI_API_KEY": "oa",
        "CARTESIA_API_KEY": "ct",
        "CARTESIA_VOICE_ID": "voice",
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="unsupported tts provider: espeak"):
        PipelineConfig.from_environment({"providers": {"tts": "espeak"}})
    register_tts("espeak", lambda config, settings, **extra: None)
    # Registered but without a credential mapping: fail closed, never borrow cartesia's.
    with pytest.raises(RuntimeError, match="tts provider has no credential mapping: espeak"):
        PipelineConfig.from_environment({"providers": {"tts": "espeak"}})
    register_tts_credentials("espeak", ())
    config = PipelineConfig.from_environment({"providers": {"tts": "espeak"}})
    assert config.providers["tts"] == "espeak"
    # Explicitly credential-less: no cartesia fallback through the accessors.
    assert config.tts_api_key == ""
    assert config.tts_voice_id == ""


def test_register_tts_credentials_validates_fields_and_unmapped_accessors_are_empty(
    restore_registries,
) -> None:
    with pytest.raises(ValueError):
        register_tts_credentials("x", ("openai_api_key",))
    with pytest.raises(ValueError):
        register_tts_credentials("x", ("cartesia_api_key", "cartesia_voice_id", "elevenlabs_api_key"))
    config = _config(providers={"stt": "deepgram", "llm": "openai", "tts": "unmapped"})
    assert config.tts_api_key == ""
    assert config.tts_voice_id == ""


def test_deepgram_factory_builds_the_service_with_the_session_settings(monkeypatch) -> None:
    from pipecat.services.deepgram import stt

    monkeypatch.setattr(stt, "DeepgramSTTService", Recorder)
    config = _config(provider_settings={"stt": {"model": "nova-3", "interim_results": True}})

    service = build_stt(config, "ko")

    assert service.kwargs["api_key"] == "dg-key"
    assert service.kwargs["settings"] == {
        "model": "nova-3",
        "interim_results": True,
        "language": "ko",
    }
    # No language given: the settings' own (here none) are left alone, and the session's
    # provider_settings dict is never mutated.
    assert "language" not in build_stt(config).kwargs["settings"]
    assert "language" not in config.provider_settings["stt"]


def test_cartesia_factory_builds_the_service_with_voice_language_and_extras(
    monkeypatch,
) -> None:
    from pipecat.services.cartesia import tts

    monkeypatch.setattr(tts, "CartesiaTTSService", Recorder)
    config = _config(provider_settings={"tts": {"model": "sonic-2", "language": "en"}})

    service = build_tts(config, "ko", transport_destination="lane-1")

    assert service.kwargs["api_key"] == "ct-key"
    assert service.kwargs["transport_destination"] == "lane-1"
    assert service.kwargs["settings"] == {"model": "sonic-2", "language": "ko", "voice": "voice-1"}
    plain = build_tts(config).kwargs
    assert plain["settings"] == {"model": "sonic-2", "language": "en", "voice": "voice-1"}
    assert "transport_destination" not in plain
    assert config.provider_settings["tts"] == {"model": "sonic-2", "language": "en"}


def test_real_services_accept_the_factory_settings() -> None:
    stt_service = build_stt(_config(), "en")
    tts_service = build_tts(_config(), "en")
    assert type(stt_service).__name__ == "DeepgramSTTService"
    assert type(tts_service).__name__ == "CartesiaTTSService"


def test_registered_provider_is_used_by_name(restore_registries) -> None:
    seen = []

    def fake_tts(config, settings, **extra):
        seen.append((config.cartesia_voice_id, settings, extra))
        return "fake-tts"

    register_tts("fake", fake_tts)
    config = _config(
        providers={"stt": "deepgram", "llm": "openai", "tts": "fake"},
        provider_settings={"tts": {"model": "m"}},
    )
    assert build_tts(config, "ru", transport_destination="d") == "fake-tts"
    assert seen == [("voice-1", {"model": "m", "language": "ru"}, {"transport_destination": "d"})]


def test_register_stt_replaces_the_same_name(restore_registries) -> None:
    register_stt("deepgram", lambda config, settings: "replaced")
    assert build_stt(_config()) == "replaced"


def test_glossary_keyterms_reach_deepgram_only_when_present(monkeypatch) -> None:
    import pipecat.services.deepgram.stt as deepgram

    monkeypatch.setattr(deepgram, "DeepgramSTTService", Recorder)
    boosted = build_stt(_config(), "ru", ["вид на жительство"])
    assert boosted.kwargs["settings"]["keyterm"] == ["вид на жительство"]
    assert "keyterm" not in build_stt(_config(), "ru").kwargs["settings"]


def test_keyterms_are_not_passed_to_factories_without_them(restore_registries) -> None:
    register_stt("plain", lambda config, settings: settings)
    config = _config(providers={"stt": "plain", "llm": "openai", "tts": "cartesia"})
    assert build_stt(config, "en") == {"language": "en"}


def test_llm_registry_has_openai_and_openai_compatible() -> None:
    assert set(llm_names()) >= {"openai", "openai-compatible"}
    assert provider_capabilities("llm", "openai").supports("tools")
    assert provider_capabilities("stt", "deepgram").supports("keyterms")
    assert provider_capabilities("tts", "cartesia").kind == "tts"
    with pytest.raises(KeyError):
        provider_capabilities("tts", "nobody")


def test_unknown_llm_provider_fails_with_a_clear_error() -> None:
    config = _config(providers={"stt": "deepgram", "llm": "mistral", "tts": "cartesia"})
    with pytest.raises(RuntimeError, match="unsupported llm provider: mistral"):
        build_llm(config)


def test_openai_factory_builds_the_service_like_the_pipeline_used_to(monkeypatch) -> None:
    from pipecat.services.openai import llm

    monkeypatch.setattr(llm, "OpenAILLMService", Recorder)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://qwen.test/v1")
    config = _config(
        provider_settings={"llm": {"model": "gpt-x", "translation_model": "gpt-t"}},
    )

    service = build_llm(config, "be brief")

    assert service.kwargs["api_key"] == "oa-key"
    assert service.kwargs["base_url"] == "http://qwen.test/v1"
    # translation_model is for the translators only; the session dict is not mutated.
    assert service.kwargs["settings"] == {"model": "gpt-x", "system_instruction": "be brief"}
    assert config.provider_settings["llm"] == {"model": "gpt-x", "translation_model": "gpt-t"}
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert build_llm(config).kwargs["base_url"] is None
    assert "system_instruction" not in build_llm(config).kwargs["settings"]


def _session_env(monkeypatch) -> None:
    for key, value in {
        "LIVEKIT_URL": "ws://localhost:7880",
        "RUNTIME_LIVEKIT_TOKEN": "token",
        "LIVEKIT_ROOM_NAME": "room",
        "DEEPGRAM_API_KEY": "dg",
        "OPENAI_API_KEY": "oa",
        "CARTESIA_API_KEY": "ct",
        "CARTESIA_VOICE_ID": "voice",
    }.items():
        monkeypatch.setenv(key, value)


def test_session_start_rejects_unknown_llm_provider(monkeypatch) -> None:
    _session_env(monkeypatch)
    with pytest.raises(RuntimeError, match="unsupported llm provider: mistral"):
        PipelineConfig.from_environment({"providers": {"llm": "mistral"}})
    with pytest.raises(RuntimeError, match="unsupported voice provider: x"):
        PipelineConfig.from_environment({"providers": {"voice": "x"}})


def test_provider_settings_keys_follow_the_registered_provider(
    monkeypatch, restore_registries
) -> None:
    _session_env(monkeypatch)
    # Today's keys and types are unchanged.
    config = PipelineConfig.from_environment(
        {
            "provider_settings": {
                "stt": {"model": "nova-3", "language": "multi", "interim_results": True},
                "llm": {"model": "m", "temperature": 0.2, "translation_model": "t"},
                "tts": {"model": "sonic-2", "language": "en"},
            }
        }
    )
    assert config.provider_settings["llm"]["temperature"] == 0.2
    with pytest.raises(RuntimeError, match="unsupported tts provider setting: stability"):
        PipelineConfig.from_environment({"provider_settings": {"tts": {"stability": 0.5}}})
    with pytest.raises(RuntimeError, match="invalid llm provider setting type: temperature"):
        PipelineConfig.from_environment({"provider_settings": {"llm": {"temperature": "hot"}}})
    # A provider declares its own keys when it registers; nothing else changes.
    register_tts(
        "fake",
        lambda config, settings, **extra: None,
        settings={"model": str, "stability": (int, float)},
    )
    register_tts_credentials("fake", ())
    config = PipelineConfig.from_environment(
        {
            "providers": {"tts": "fake"},
            "provider_settings": {"tts": {"stability": 0.5}},
        }
    )
    assert config.provider_settings["tts"] == {"stability": 0.5}
    with pytest.raises(RuntimeError, match="unsupported tts provider setting: language"):
        PipelineConfig.from_environment(
            {"providers": {"tts": "fake"}, "provider_settings": {"tts": {"language": "en"}}}
        )


# -- every pipeline path asks the registry ---------------------------------------


@pytest.fixture
def light_models(monkeypatch):
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer)
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")


@pytest.fixture
def fake_providers(restore_registries):
    """Register ``fake`` as stt, llm and tts; every service built is recorded in ``calls``."""

    from pipecat.processors.frame_processor import FrameProcessor

    calls: list[tuple] = []

    class FakeSTT(FrameProcessor):
        pass

    class FakeLLM(FrameProcessor):
        def register_function(self, name, handler, **options) -> None:
            calls.append(("tool", name))

    class FakeTTS(FrameProcessor):
        pass

    def stt(config, settings, **extra):
        calls.append(("stt", dict(settings), extra))
        service = FakeSTT()
        service._settings = type("S", (), {"language": settings.get("language")})()
        return service

    def llm(config, settings):
        calls.append(("llm", dict(settings)))
        service = FakeLLM()
        service._settings = type("S", (), {"system_instruction": settings["system_instruction"]})()
        return service

    def tts(config, settings, **extra):
        calls.append(("tts", dict(settings), extra))
        return FakeTTS()

    register_stt("fake", stt, settings={"model": str, "language": str}, features=("keyterms",))
    register_llm("fake", llm, settings={"model": str})
    register_tts("fake", tts, settings={"language": str})
    register_tts_credentials("fake", ())
    return calls


def _fake_config(**kwargs) -> PipelineConfig:
    return _config(
        providers={"stt": "fake", "llm": "fake", "tts": "fake"},
        provider_settings={"stt": {"model": "m"}},
        **kwargs,
    )


def _inner(worker):
    from pipecat.pipeline.pipeline import Pipeline

    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    return [
        p for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]


def test_agent_pipeline_asks_the_registry_for_stt_llm_and_tts(light_models, fake_providers) -> None:
    from src.runtime.pipeline import build_livekit_pipeline

    _, worker = build_livekit_pipeline(_fake_config())

    names = [type(p).__name__ for p in _inner(worker)]
    assert names.count("FakeSTT") == names.count("FakeLLM") == names.count("FakeTTS") == 1
    assert "DeepgramSTTService" not in names and "CartesiaTTSService" not in names
    assert [call[0] for call in fake_providers] == ["stt", "llm", "tts"]
    assert fake_providers[1][1]["system_instruction"] == "prompt"


def test_agent_translator_pipeline_asks_the_registry(light_models, fake_providers) -> None:
    from uuid import uuid4

    from src.runtime.pipeline import build_livekit_pipeline
    from src.runtime.session import RuntimeSession

    config = _fake_config(translation={"working_language": "en", "caller_languages": ["ru", "es"]})
    _, worker = build_livekit_pipeline(config, RuntimeSession(uuid4(), uuid4(), uuid4()))

    names = [type(p).__name__ for p in _inner(worker)]
    assert {"FakeSTT", "FakeLLM", "FakeTTS"} <= set(names)
    kinds = {call[0]: call for call in fake_providers}
    assert kinds["stt"][1]["language"] == "multi"  # per-utterance detection, unchanged
    assert kinds["tts"][1]["language"] == "ru"
    assert "InboundTranslator" in names and "OutboundTranslator" in names


def test_cascaded_interpreter_asks_the_registry(light_models, fake_providers) -> None:
    from src.runtime.pipeline import build_livekit_pipeline

    config = _fake_config(
        mode="interpreter", interpreter={"languages": ["en", "ru"], "direction": "auto"}
    )
    _, worker = build_livekit_pipeline(config)

    names = [type(p).__name__ for p in _inner(worker)]
    assert "FakeSTT" in names and "FakeTTS" in names
    assert "DeepgramSTTService" not in names and "CartesiaTTSService" not in names
    assert [(c[0], c[1]["language"]) for c in fake_providers] == [("stt", "multi"), ("tts", "ru")]


def test_two_way_interpreter_lanes_and_listener_tts_ask_the_registry(
    light_models, fake_providers
) -> None:
    from uuid import uuid4

    from src.runtime.participants import ParticipantDirectory
    from src.runtime.pipeline import build_livekit_pipeline
    from src.runtime.session import RuntimeSession

    participants = {
        "alice": {"language": "en", "voice_id": "v-a"},
        "bob": {"language": "ko", "voice_id": "v-b"},
    }
    config = _fake_config(
        mode="interpreter",
        interpreter={"languages": ["en", "ko"], "direction": "one_way"},
        participants=participants,
    )
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    session.participants = ParticipantDirectory.from_mapping(("en", "ko"), participants)
    _, worker = build_livekit_pipeline(config, session)

    stt_calls = [c for c in fake_providers if c[0] == "stt"]
    tts_calls = [c for c in fake_providers if c[0] == "tts"]
    assert [c[1]["language"] for c in stt_calls] == ["en", "ko"]  # one STT lane each
    assert [c[1]["language"] for c in tts_calls] == ["en", "ko"]  # one TTS per listener
    assert [c[2] for c in tts_calls] == [
        {"transport_destination": "translation-lane-0"},
        {"transport_destination": "translation-lane-1"},
    ]
    assert "DeepgramSTTService" not in repr(worker._pipeline)


def test_pipelines_fail_for_unregistered_provider_before_building(light_models) -> None:
    from src.runtime.pipeline import build_livekit_pipeline

    config = _config(providers={"stt": "whisper", "llm": "openai", "tts": "cartesia"})
    with pytest.raises(RuntimeError, match="unsupported stt provider: whisper"):
        build_livekit_pipeline(config)


# -- the interpreter never uses the agent system prompt ----------------------------------

_SECRET_PROMPT = "SECRET-AGENT-PROMPT-DO-NOT-USE"


def _spy_translators(monkeypatch):
    """Record every message list handed to a translator, with no network involved."""

    from src.runtime import interpreter

    seen: list[list[dict[str, str]]] = []

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    seen.append(kwargs["messages"])
                    raise AssertionError("stop")

    original = interpreter.OpenAITranslator.__init__

    def init(self, **kwargs):
        original(self, **{**kwargs, "client": FakeClient()})

    monkeypatch.setattr(interpreter.OpenAITranslator, "__init__", init)
    return seen


def _translators(worker):
    from src.runtime.interpreter import TranslationProcessor

    return [p for p in _inner(worker) if isinstance(p, TranslationProcessor)]


def _assert_prompt_unused(fake_providers, worker, seen) -> None:
    import asyncio

    from src.runtime.interpreter import build_translation_messages

    # The LLM registry entry (fake_providers) was never asked for a service.
    assert "llm" not in [c[0] for c in fake_providers]
    # No factory call (stt/tts settings and extras) mentions the prompt.
    assert _SECRET_PROMPT not in repr(fake_providers)
    processors = _translators(worker)
    assert processors
    for processor in processors:
        async def run():
            async for _ in processor._translate(build_translation_messages("hello", "en", "ko")):
                pass

        with pytest.raises(AssertionError):
            asyncio.run(run())
    assert seen and _SECRET_PROMPT not in repr(seen)


def test_cascaded_interpreter_ignores_agent_system_prompt(
    light_models, fake_providers, monkeypatch
) -> None:
    from src.runtime import pipeline

    seen = _spy_translators(monkeypatch)
    llm_calls: list[tuple] = []
    monkeypatch.setattr(pipeline, "build_llm", lambda *a, **k: llm_calls.append(a))
    config = _fake_config(
        mode="interpreter",
        interpreter={"languages": ["en", "ru"], "direction": "auto"},
        system_instruction=_SECRET_PROMPT,
    )
    _, worker = pipeline.build_livekit_pipeline(config)

    assert llm_calls == []
    _assert_prompt_unused(fake_providers, worker, seen)


def test_two_participant_interpreter_ignores_agent_system_prompt(
    light_models, fake_providers, monkeypatch
) -> None:
    from uuid import uuid4

    from src.runtime import pipeline
    from src.runtime.participants import ParticipantDirectory
    from src.runtime.session import RuntimeSession

    seen = _spy_translators(monkeypatch)
    llm_calls: list[tuple] = []
    monkeypatch.setattr(pipeline, "build_llm", lambda *a, **k: llm_calls.append(a))
    participants = {
        "alice": {"language": "en", "voice_id": "v-a"},
        "bob": {"language": "ko", "voice_id": "v-b"},
    }
    config = _fake_config(
        mode="interpreter",
        interpreter={"languages": ["en", "ko"], "direction": "one_way"},
        participants=participants,
        system_instruction=_SECRET_PROMPT,
    )
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    session.participants = ParticipantDirectory.from_mapping(("en", "ko"), participants)
    _, worker = pipeline.build_livekit_pipeline(config, session)

    assert llm_calls == []
    _assert_prompt_unused(fake_providers, worker, seen)


# -- ElevenLabs STT + Gemini translation (no model is locked in code) -----------------


_ELEVEN_STACK = {"stt": "elevenlabs", "llm": "gemini", "tts": "elevenlabs"}
_ELEVEN_ENV = {
    "LIVEKIT_URL": "ws://localhost:7880",
    "RUNTIME_LIVEKIT_TOKEN": "token",
    "LIVEKIT_ROOM_NAME": "room",
    "ELEVENLABS_API_KEY": "el-key",
    "ELEVENLABS_VOICE_ID": "voice123",
    "GEMINI_API_KEY": "gm-key",
}


def _stack_env(monkeypatch, **extra) -> None:
    for name in (
        "DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID",
        "GEMINI_MODEL", "GEMINI_TRANSLATION_MODEL", "GEMINI_BASE_URL", "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**_ELEVEN_ENV, **extra}.items():
        monkeypatch.setenv(name, value)


def test_elevenlabs_and_gemini_are_registered() -> None:
    assert "elevenlabs" in stt_names()
    assert "gemini" in llm_names()
    assert provider_capabilities("stt", "elevenlabs").supports("keyterms")
    assert not provider_capabilities("stt", "elevenlabs").supports("multi_language")


def test_elevenlabs_stt_factory_uses_the_key_and_leaves_the_model_to_the_sdk(monkeypatch) -> None:
    from pipecat.services.elevenlabs import stt

    monkeypatch.setattr(stt, "ElevenLabsRealtimeSTTService", Recorder)
    monkeypatch.setattr(speech_providers, "_ELEVENLABS_STT_SERVICE", None)
    config = _config(
        elevenlabs_api_key="el-key", providers={"stt": "elevenlabs", "llm": "openai", "tts": "cartesia"}
    )

    service = build_stt(config, "ko")

    assert service.kwargs["api_key"] == "el-key"
    assert service.kwargs["settings"] == {"language": "ko"}  # no model chosen by Voice Labs


def test_elevenlabs_stt_maps_multi_to_auto_detection_and_passes_keyterms(monkeypatch) -> None:
    from pipecat.services.elevenlabs import stt

    monkeypatch.setattr(stt, "ElevenLabsRealtimeSTTService", Recorder)
    monkeypatch.setattr(speech_providers, "_ELEVENLABS_STT_SERVICE", None)
    config = _config(
        elevenlabs_api_key="el-key",
        providers={"stt": "elevenlabs", "llm": "openai", "tts": "cartesia"},
        provider_settings={"stt": {"model": "custom-model"}},
    )

    service = build_stt(config, "multi", ["Voice Labs"])

    assert service.kwargs["settings"] == {
        "model": "custom-model",
        "language": None,
        "keyterms": ["Voice Labs"],
    }


def test_elevenlabs_stack_needs_only_elevenlabs_and_gemini_keys(monkeypatch) -> None:
    _stack_env(monkeypatch)
    config = PipelineConfig.from_environment({"providers": _ELEVEN_STACK})
    assert (config.stt_api_key, config.llm_api_key, config.tts_api_key) == (
        "el-key", "gm-key", "el-key",
    )
    assert config.deepgram_api_key == "" and config.openai_api_key == ""


@pytest.mark.parametrize("missing", ["ELEVENLABS_API_KEY", "GEMINI_API_KEY"])
def test_elevenlabs_stack_reports_a_missing_key(monkeypatch, missing) -> None:
    _stack_env(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        PipelineConfig.from_environment({"providers": _ELEVEN_STACK})


def test_default_stack_still_requires_deepgram_and_openai(monkeypatch) -> None:
    _stack_env(monkeypatch, CARTESIA_API_KEY="c", CARTESIA_VOICE_ID="v")
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY.*OPENAI_API_KEY|OPENAI_API_KEY.*DEEPGRAM_API_KEY"):
        PipelineConfig.from_environment({})


def test_stt_and_llm_providers_without_credential_mapping_fail_closed(
    monkeypatch, restore_registries
) -> None:
    _stack_env(monkeypatch)
    register_stt("orphan", lambda config, settings: None)
    with pytest.raises(RuntimeError, match="stt provider has no credential mapping: orphan"):
        PipelineConfig.from_environment({"providers": {**_ELEVEN_STACK, "stt": "orphan"}})
    register_llm("orphan", lambda config, settings: None)
    with pytest.raises(RuntimeError, match="llm provider has no credential mapping: orphan"):
        PipelineConfig.from_environment({"providers": {**_ELEVEN_STACK, "llm": "orphan"}})


def test_gemini_translation_has_no_built_in_model(monkeypatch) -> None:
    _stack_env(monkeypatch)
    config = PipelineConfig.from_environment({"providers": _ELEVEN_STACK})
    with pytest.raises(RuntimeError, match="gemini translation needs a model"):
        speech_providers.resolve_translation_model(config, default="gpt-4.1-mini")
    # The record's setting, then the environment, supply it; the OpenAI default is ignored.
    monkeypatch.setenv("GEMINI_TRANSLATION_MODEL", "env-model")
    assert speech_providers.resolve_translation_model(config, default="gpt-4.1-mini") == "env-model"
    with_setting = PipelineConfig.from_environment(
        {"providers": _ELEVEN_STACK, "provider_settings": {"llm": {"model": "rec-model"}}}
    )
    assert speech_providers.resolve_translation_model(with_setting, default="x") == "rec-model"


def test_gemini_translator_uses_its_own_endpoint_not_openai_base_url(monkeypatch) -> None:
    _stack_env(monkeypatch, OPENAI_BASE_URL="https://openai.example/v1")
    config = PipelineConfig.from_environment({"providers": _ELEVEN_STACK})
    assert speech_providers.translation_endpoint(config) == (
        "gm-key", speech_providers.GEMINI_DEFAULT_BASE_URL,
    )
    monkeypatch.setenv("GEMINI_BASE_URL", "https://proxy.example/openai/")
    assert speech_providers.translation_endpoint(config)[1] == "https://proxy.example/openai/"
    openai_config = _config(providers={"stt": "deepgram", "llm": "openai-compatible", "tts": "cartesia"})
    assert speech_providers.translation_endpoint(openai_config) == ("oa-key", "https://openai.example/v1")


def test_openai_translation_keeps_its_profile_default(monkeypatch) -> None:
    config = _config(providers={"stt": "deepgram", "llm": "openai-compatible", "tts": "cartesia"})
    assert speech_providers.resolve_translation_model(config, default="profile-model") == "profile-model"
    with_setting = _config(provider_settings={"llm": {"model": "mine"}})
    assert speech_providers.resolve_translation_model(with_setting, default="profile-model") == "mine"


def test_translation_failures_are_reported_under_the_real_provider() -> None:
    assert speech_providers.translation_provider(_config(providers=_ELEVEN_STACK)) == "gemini"
    assert speech_providers.translation_provider(_config()) == "openai"


def test_gemini_without_a_model_fails_at_session_start_for_translation(monkeypatch) -> None:
    _stack_env(monkeypatch)
    interpreter = {"mode": "interpreter", "interpreter": {"languages": ["en", "ko"], "direction": "one_way"}}
    with pytest.raises(RuntimeError, match="gemini translation needs a model"):
        PipelineConfig.from_environment({"providers": _ELEVEN_STACK, **interpreter})
    monkeypatch.setenv("GEMINI_TRANSLATION_MODEL", "env-model")
    assert PipelineConfig.from_environment({"providers": _ELEVEN_STACK, **interpreter}).providers == _ELEVEN_STACK


def test_gemini_nameless_tool_calls_are_dropped_before_they_run() -> None:
    """Gemini's stream yields a phantom call with no name next to the real one; running it 400s the LLM."""
    import asyncio
    from types import SimpleNamespace

    real = SimpleNamespace(function_name="transfer_to_human", tool_call_id="call_1", arguments={"reason": "asked"})
    phantom = SimpleNamespace(function_name="", tool_call_id="", arguments={})
    assert speech_providers.drop_nameless_function_calls([phantom, real]) == [real]

    cls = speech_providers._gemini_service_class()
    ran: list[list] = []

    async def fake_super_run(self, calls):
        ran.append(list(calls))

    parent = cls.__mro__[1]
    original = parent.run_function_calls
    parent.run_function_calls = fake_super_run
    try:
        service = object.__new__(cls)
        asyncio.run(service.run_function_calls([phantom, real]))
        asyncio.run(service.run_function_calls([phantom]))
    finally:
        parent.run_function_calls = original
    assert ran == [[real]]


def test_gemini_replayed_tool_calls_get_a_placeholder_thought_signature() -> None:
    messages = [
        {"role": "user", "content": "Can I talk to a person?"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "transfer_to_human", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "x", "arguments": "{}"},
             "extra_content": {"google": {"thought_signature": "real"}}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    out = speech_providers.add_gemini_thought_signatures(messages)
    calls = out[1]["tool_calls"]
    assert calls[0]["extra_content"] == {"google": {"thought_signature": "skip_thought_signature_validator"}}
    assert calls[1]["extra_content"]["google"]["thought_signature"] == "real"
    assert out[0] == messages[0] and out[2] == messages[2]
    assert "extra_content" not in messages[1]["tool_calls"][0]  # the context itself is not mutated

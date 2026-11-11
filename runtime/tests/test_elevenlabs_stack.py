"""ElevenLabs STT + Gemini translation + ElevenLabs voice: wiring, classification, health."""

import asyncio
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("pipecat")

from src.runtime import provider_health
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.provider_failures import cascaded_fallback_config, provider_kind
from src.runtime.session import RuntimeSession

STACK = {"stt": "elevenlabs", "llm": "gemini", "tts": "elevenlabs"}


def stack_config(**overrides) -> PipelineConfig:
    values = dict(
        livekit_url="ws://localhost:7880",
        livekit_token="token",
        room_name="room",
        deepgram_api_key="",
        openai_api_key="",
        cartesia_api_key="",
        cartesia_voice_id="",
        system_instruction="prompt",
        providers=STACK,
        provider_settings={"llm": {"model": "any-model"}},
        elevenlabs_api_key="el-key",
        elevenlabs_voice_id="voice123",
        gemini_api_key="gm-key",
        mode="interpreter",
        interpreter={"languages": ["en", "ru"], "direction": "auto"},
    )
    values.update(overrides)
    return PipelineConfig(**values)


def test_cascaded_interpreter_builds_with_elevenlabs_stt_gemini_and_elevenlabs_voice(monkeypatch) -> None:
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.pipeline.pipeline import Pipeline

    class Light:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", Light)
    _, worker = build_livekit_pipeline(stack_config(), RuntimeSession(uuid4(), uuid4(), uuid4()))
    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    processors = [
        p for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]
    names = [type(p).__name__ for p in processors]
    assert "VoiceLabsElevenLabsSTTService" in names and "VoiceLabsElevenLabsTTSService" in names
    translation = next(p for p in processors if type(p).__name__ == "TranslationProcessor")
    assert translation._provider == "gemini" and translation._model == "any-model"
    # Failures of the new services are attributed to the right slots (no "Realtime" mix-up).
    kinds = {type(p).__name__: provider_kind(p) for p in processors}
    assert kinds["VoiceLabsElevenLabsSTTService"] == "stt"
    assert kinds["VoiceLabsElevenLabsTTSService"] == "tts"
    assert kinds["TranslationProcessor"] == "translation"


def test_cascaded_fallback_uses_the_selected_providers_credentials() -> None:
    realtime = stack_config(
        interpreter={"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"}
    )
    assert cascaded_fallback_config(realtime).interpreter["engine"] == "cascaded"
    assert cascaded_fallback_config(stack_config(gemini_api_key="", interpreter=realtime.interpreter)) is None
    assert cascaded_fallback_config(stack_config(elevenlabs_api_key="", interpreter=realtime.interpreter)) is None


def _probe(monkeypatch, env, handler):
    provider_health.reset()
    for name in env:
        monkeypatch.setenv(name, "secret")
    try:
        return asyncio.run(provider_health.probe_providers(transport=httpx.MockTransport(handler)))
    finally:
        provider_health.reset()


def test_gemini_probe_lists_models_on_its_own_endpoint(monkeypatch) -> None:
    for name in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(401 if len(seen) == 2 else 200, json={})

    ok = _probe(monkeypatch, ["GEMINI_API_KEY"], handler)
    assert [(s.kind, s.name, s.status) for s in ok] == [("llm", "gemini", "ok")]
    assert seen[0] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/models",
        "Bearer secret",
    )
    bad = _probe(monkeypatch, ["GEMINI_API_KEY"], handler)
    assert (bad[0].status, bad[0].category) == ("down", "auth")

"""RM-010: the small TranslationEngine interface and its registry."""

from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.processors.frame_processor import FrameProcessor

from src.runtime import translation_engine
from src.runtime.interpreter import InterpreterSettings
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession
from src.runtime.translation_engine import (
    CascadedEngine,
    EngineContext,
    RealtimeEngine,
    TranslationEngine,
    engine_names,
    get_engine,
    register_engine,
)


def test_both_builtin_engines_are_registered_and_satisfy_the_interface() -> None:
    assert engine_names()[:2] == ("cascaded", "realtime")
    for name, engine_type in (("cascaded", CascadedEngine), ("realtime", RealtimeEngine)):
        engine = get_engine(name)
        assert isinstance(engine, engine_type) and isinstance(engine, TranslationEngine)
        assert engine.name == name


def test_capabilities_of_the_builtin_engines() -> None:
    cascaded, realtime = get_engine("cascaded"), get_engine("realtime")
    # Cascaded: two-way / per-participant sessions, speaks with the agent's or the
    # participant's own voice. Realtime: one way, provider voice.
    assert (cascaded.supports_two_way, cascaded.uses_own_voice) == (True, True)
    assert (realtime.supports_two_way, realtime.uses_own_voice) == (False, False)


def test_unknown_engine_is_rejected_with_the_established_message() -> None:
    with pytest.raises(RuntimeError, match="unsupported interpreter engine: whisper"):
        get_engine("whisper")
    with pytest.raises(RuntimeError, match="unsupported interpreter engine: whisper"):
        InterpreterSettings.from_mapping(
            {"languages": ["en", "ru"], "direction": "one_way", "engine": "whisper"}
        )


def test_engines_validate_their_own_settings() -> None:
    pair = ("en", "ru")
    get_engine("cascaded").validate(InterpreterSettings(languages=pair, direction="auto"))
    get_engine("realtime").validate(InterpreterSettings(languages=pair, direction="one_way", engine="realtime"))
    with pytest.raises(RuntimeError, match="one_way"):
        get_engine("realtime").validate(
            InterpreterSettings(languages=pair, direction="auto", engine="realtime")
        )


@pytest.fixture
def light_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.audio.vad import silero

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)


@pytest.fixture
def fake_engine():
    """A third engine registered for one test, removed afterwards."""

    class Passthrough(FrameProcessor):
        pass

    class FakeEngine:
        name = "fake"
        supports_two_way = False
        uses_own_voice = False

        def __init__(self) -> None:
            self.validated: list[InterpreterSettings] = []
            self.built: list[tuple[InterpreterSettings, EngineContext]] = []

        def validate(self, settings) -> None:
            self.validated.append(settings)
            if settings.languages[1] == "tr":
                raise RuntimeError("fake engine cannot speak tr")

        def build_processors(self, settings, context):
            self.built.append((settings, context))
            return [Passthrough(name="FakeEngineProcessor")]

    engine = FakeEngine()
    register_engine(engine)
    try:
        yield engine
    finally:
        translation_engine._ENGINES.pop("fake", None)


def test_a_new_engine_plugs_in_without_touching_the_pipeline(light_models, fake_engine) -> None:
    from pipecat.pipeline.pipeline import Pipeline

    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    config = PipelineConfig(
        "ws://localhost:7880", "token", "room", "", "sk", "", "", "prompt",
        mode="interpreter",
        interpreter={"languages": ["en", "ru"], "direction": "one_way", "engine": "fake"},
    )  # fmt: skip
    _, worker = build_livekit_pipeline(config, session)
    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    names = [
        p.name for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]
    assert names[1:-1] == ["FakeEngineProcessor"]  # between the transport's input and output
    [(settings, context)] = fake_engine.built
    assert settings.engine == "fake" and context.config is config and context.session is session
    assert context.directory is None
    # The pipeline gives the engine a timeline recorder named after it (RM-003).
    assert context.timeline is not None and context.timeline.engine == "fake"


def test_config_checks_follow_the_engine_capabilities(monkeypatch, fake_engine) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    for name in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID"):
        monkeypatch.delenv(name, raising=False)
    session = {
        "room_name": "r",
        "livekit_token": "t",
        "mode": "interpreter",
        "interpreter": {"languages": ["en", "ru"], "direction": "one_way", "engine": "fake"},
    }
    # No own voice: no STT/TTS credentials needed.
    assert PipelineConfig.from_environment(session).interpreter["engine"] == "fake"
    # Not two-way: participants are refused.
    with pytest.raises(RuntimeError, match="cascaded"):
        PipelineConfig.from_environment({**session, "participants": {"a": {"language": "en"}}})
    # The engine's own validation runs when the config is read.
    bad = {**session, "interpreter": {**session["interpreter"], "languages": ["en", "tr"]}}
    with pytest.raises(RuntimeError, match="fake engine cannot speak tr"):
        PipelineConfig.from_environment(bad)


def test_builtin_pipelines_are_unchanged_by_the_registry(light_models, monkeypatch) -> None:
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.pipeline.pipeline import Pipeline

    class Light:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", Light)
    session = RuntimeSession(uuid4(), uuid4(), uuid4())

    def names_for(interpreter):
        config = PipelineConfig(
            "ws://localhost:7880", "token", "room", "dg", "sk", "cart", "voice", "prompt",
            mode="interpreter", interpreter=interpreter,
        )  # fmt: skip
        _, worker = build_livekit_pipeline(config, session)
        pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
        inner = [
            type(p).__name__
            for p in pipeline._processors
            if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
        ]
        assert inner[0] == "LiveKitInputTransport" and inner[-1] == "LiveKitOutputTransport"
        return inner[1:-1]

    assert names_for({"languages": ["en", "ru"], "direction": "auto"}) == [
        "VADProcessor",
        "DeepgramSTTService",
        "UserTurnProcessor",
        "TranslationProcessor",
        "CartesiaTTSService",
        "RuntimeEventProcessor",
    ]
    assert names_for({"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"}) == [
        "VADProcessor",
        "RealtimeTranslateProcessor",
        "RuntimeEventProcessor",
    ]

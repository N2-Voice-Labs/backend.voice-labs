"""RM-011: Instant / Balanced / Accurate interpreter modes."""

from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from fastapi import HTTPException

from src.runtime import api
from src.runtime.interpreter import (
    InterpreterSettings,
    SentenceChunker,
    build_translation_messages,
)
from src.runtime.latency_modes import (
    LATENCY_MODES,
    PROFILES,
    default_engine_for,
    profile_for,
)
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession

KO_EN = {"languages": ["ko", "en"], "direction": "one_way"}
EN_RU_AUTO = {"languages": ["en", "ru"], "direction": "auto"}


# -- mapping ----------------------------------------------------------------------


def test_profiles_match_the_roadmap() -> None:
    assert LATENCY_MODES == ("instant", "balanced", "accurate")
    instant, balanced, accurate = (PROFILES[mode] for mode in LATENCY_MODES)
    assert (instant.turn_pause_secs, instant.clause_min_chars, instant.translation_model) == (
        0.2, 20, "gpt-4.1-nano",
    )  # fmt: skip
    # Balanced is exactly what the cascaded engine did before modes existed.
    assert (balanced.turn_pause_secs, balanced.clause_min_chars, balanced.translation_model) == (
        0.3, 40, "gpt-4.1-mini",
    )  # fmt: skip
    assert (accurate.turn_pause_secs, accurate.clause_min_chars, accurate.translation_model) == (
        0.7, None, "gpt-4.1",
    )  # fmt: skip
    assert {p.context_pairs for p in PROFILES.values()} == {4}
    assert profile_for(None) is balanced


@pytest.mark.parametrize(
    "mode, direction, target, engine",
    [
        ("instant", "one_way", "en", "realtime"),
        ("instant", "one_way", "ko", "realtime"),
        ("instant", "auto", "ru", "cascaded"),  # two-way: the realtime model is one-way only
        ("instant", "one_way", "nl", "cascaded"),  # target the realtime model cannot speak
        ("instant", "one_way", "tr", "cascaded"),
        ("balanced", "one_way", "en", "cascaded"),
        ("accurate", "one_way", "en", "cascaded"),
        (None, "one_way", "en", "cascaded"),
    ],
)
def test_mode_picks_the_engine(mode, direction, target, engine) -> None:
    assert default_engine_for(mode, direction, target) == engine


def test_settings_default_to_balanced_cascaded() -> None:
    settings = InterpreterSettings.from_mapping(EN_RU_AUTO)
    assert (settings.latency_mode, settings.engine) == ("balanced", "cascaded")
    assert settings.profile is PROFILES["balanced"]


def test_instant_chooses_realtime_only_when_allowed() -> None:
    assert InterpreterSettings.from_mapping({**KO_EN, "latency_mode": "instant"}).engine == "realtime"
    assert InterpreterSettings.from_mapping({**EN_RU_AUTO, "latency_mode": "instant"}).engine == "cascaded"
    nl_target = {"languages": ["en", "nl"], "direction": "one_way", "latency_mode": "instant"}
    assert InterpreterSettings.from_mapping(nl_target).engine == "cascaded"
    assert InterpreterSettings.from_mapping({**KO_EN, "latency_mode": "accurate"}).engine == "cascaded"


def test_an_explicit_engine_wins_over_the_mode() -> None:
    forced = InterpreterSettings.from_mapping({**KO_EN, "latency_mode": "instant", "engine": "cascaded"})
    assert (forced.engine, forced.latency_mode) == ("cascaded", "instant")
    assert forced.profile is PROFILES["instant"]  # the cascaded engine still gets the instant tuning
    realtime = InterpreterSettings.from_mapping({**KO_EN, "latency_mode": "accurate", "engine": "realtime"})
    assert realtime.engine == "realtime"
    # An explicit engine is still validated: realtime cannot do two-way.
    with pytest.raises(RuntimeError, match="one_way"):
        InterpreterSettings.from_mapping({**EN_RU_AUTO, "latency_mode": "instant", "engine": "realtime"})


@pytest.mark.parametrize("mode", ["fast", "INSTANT", 3, True])
def test_unknown_latency_mode_is_rejected(mode) -> None:
    with pytest.raises(RuntimeError, match="latency_mode"):
        InterpreterSettings.from_mapping({**EN_RU_AUTO, "latency_mode": mode})


# -- what the cascaded engine does with the profile ----------------------------------------


def test_sentence_chunker_clause_threshold_follows_the_mode() -> None:
    text = "First clause is long enough, then more words follow here. Done."
    default = SentenceChunker()
    assert default.feed(text) == ["First clause is long enough, then more words follow here."]
    instant = SentenceChunker(PROFILES["instant"].clause_min_chars)
    assert instant.feed("A clause of twenty chars, rest of it") == ["A clause of twenty chars,"]
    assert SentenceChunker(20).feed("Too short, rest") == []  # under 20 characters
    accurate = SentenceChunker(PROFILES["accurate"].clause_min_chars)
    assert accurate.feed("A clause of twenty chars, and a very long continuation here, ") == []
    assert accurate.feed("then the end. Next") == [
        "A clause of twenty chars, and a very long continuation here, then the end."
    ]
    assert accurate.flush() == ["Next"]


def test_accurate_chunker_speaks_unpunctuated_ru_and_en_text_on_flush() -> None:
    # PR-10: accurate (sentence-only) must still deliver text that never ends in punctuation,
    # streamed token by token, for Cyrillic and Latin alike (nothing is lost to the buffer).
    for text in ("Мы можем предложить вам более раннее время", "Please bring your passport"):
        chunker = SentenceChunker(PROFILES["accurate"].clause_min_chars)
        assert [c for token in text.split(" ") for c in chunker.feed(token + " ")] == []
        assert chunker.flush() == [text]


def test_translation_context_length_is_configurable() -> None:
    history = tuple(("en", f"s{i}", "ru", f"t{i}") for i in range(6))
    default = build_translation_messages("x", "en", "ru", history)[0]["content"]
    assert "s2" in default and "s1" not in default  # the last 4 pairs
    two = build_translation_messages("x", "en", "ru", history, 2)[0]["content"]
    assert "s4" in two and "s3" not in two
    assert "Earlier utterances" not in build_translation_messages("x", "en", "ru", history, 0)[0]["content"]


@pytest.fixture
def light_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer)
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)


def _processors(worker):
    from pipecat.pipeline.pipeline import Pipeline

    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    return [
        p for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]


def _cascaded(mode, provider_settings=None, interpreter=None):
    config = PipelineConfig(
        "ws://localhost:7880", "token", "room", "deepgram", "openai", "cartesia", "voice",
        "prompt", mode="interpreter",
        interpreter={**(interpreter or EN_RU_AUTO), "latency_mode": mode},
        provider_settings=provider_settings or {},
    )  # fmt: skip
    _, worker = build_livekit_pipeline(config, RuntimeSession(uuid4(), uuid4(), uuid4()))
    processors = _processors(worker)
    by_name = {type(p).__name__: p for p in processors}
    return processors, by_name


@pytest.mark.parametrize(
    "mode, pause, model, clause",
    [
        ("instant", 0.2, "gpt-4.1-nano", 20),
        ("balanced", 0.3, "gpt-4.1-mini", 40),
        ("accurate", 0.7, "gpt-4.1", None),
    ],
)
def test_cascaded_pipeline_is_tuned_by_the_mode(light_models, mode, pause, model, clause) -> None:
    _, by_name = _cascaded(mode)
    stops = by_name["UserTurnProcessor"]._user_turn_controller._user_turn_strategies.stop
    assert [s._user_speech_timeout for s in stops] == [pause]
    translation = by_name["TranslationProcessor"]
    assert translation._model == model
    assert translation._clause_min_chars == clause
    assert translation._context_pairs == 4


def test_an_explicit_llm_model_overrides_the_mode_model(light_models) -> None:
    _, by_name = _cascaded("accurate", {"llm": {"model": "gpt-test"}})
    assert by_name["TranslationProcessor"]._model == "gpt-test"
    # ... but the rest of the accurate profile still applies.
    stops = by_name["UserTurnProcessor"]._user_turn_controller._user_turn_strategies.stop
    assert stops[0]._user_speech_timeout == 0.7


def test_instant_with_a_realtime_capable_pair_builds_the_realtime_pipeline(light_models) -> None:
    processors, _ = _cascaded("instant", interpreter=KO_EN)
    names = [type(p).__name__ for p in processors]
    assert "RealtimeTranslateProcessor" in names and "DeepgramSTTService" not in names


def test_instant_two_way_falls_back_to_the_cascaded_instant_profile(light_models) -> None:
    processors, by_name = _cascaded("instant")
    assert "DeepgramSTTService" in {type(p).__name__ for p in processors}
    assert by_name["TranslationProcessor"]._model == "gpt-4.1-nano"


# -- contract ----------------------------------------------------------------------------


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "LIVEKIT_URL": "ws://localhost:7880",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }.items():
        monkeypatch.setenv(key, value)


def test_latency_mode_travels_in_the_interpreter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    session = {
        "room_name": "r",
        "livekit_token": "t",
        "mode": "interpreter",
        "interpreter": {**EN_RU_AUTO, "latency_mode": "accurate"},
    }
    config = PipelineConfig.from_environment(session)
    assert config.interpreter["latency_mode"] == "accurate"
    session["interpreter"]["latency_mode"] = "warp"
    with pytest.raises(RuntimeError, match="latency_mode"):
        PipelineConfig.from_environment(session)


def test_instant_mode_without_an_engine_needs_no_speech_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(monkeypatch)
    for name in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID"):
        monkeypatch.delenv(name)
    session = {
        "room_name": "r",
        "livekit_token": "t",
        "mode": "interpreter",
        "interpreter": {**KO_EN, "latency_mode": "instant"},
    }
    PipelineConfig.from_environment(session)  # resolves to realtime
    session["interpreter"]["latency_mode"] = "balanced"
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        PipelineConfig.from_environment(session)


def test_participants_need_a_two_way_engine_even_when_the_mode_picked_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(monkeypatch)
    session = {
        "room_name": "r",
        "livekit_token": "t",
        "mode": "interpreter",
        "interpreter": {**KO_EN, "latency_mode": "instant"},  # => realtime
        "participants": {"alice": {"language": "ko"}},
    }
    with pytest.raises(RuntimeError, match="cascaded"):
        PipelineConfig.from_environment(session)
    session["interpreter"]["engine"] = "cascaded"
    assert PipelineConfig.from_environment(session).participants


def test_runtime_api_rejects_participants_for_a_mode_resolved_realtime_engine() -> None:
    payload = api.SessionStart(
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        room_name="room",
        mode="interpreter",
        interpreter={**KO_EN, "latency_mode": "instant"},
        participants={"alice": {"language": "ko"}},
    )
    with pytest.raises(HTTPException) as error:
        api._build_participant_directory(payload)
    assert error.value.status_code == 422
    assert error.value.detail == "participants_require_cascaded_interpreter"
    payload.interpreter = {**KO_EN, "latency_mode": "instant", "engine": "cascaded"}
    assert api._build_participant_directory(payload).lane_of("alice") == 0

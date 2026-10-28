"""Language capability table (ST-20) and glossary engine routing (ST-41)."""

import logging
from uuid import uuid4

import pytest

from src.runtime.language_capabilities import (
    CAPABILITIES,
    as_json,
    auto_detect_languages,
    can_auto_detect,
    interpreter_languages,
    realtime_output_languages,
    stt_language_for,
)
from src.runtime.latency_modes import default_engine_for


def test_table_invariants() -> None:
    for code in auto_detect_languages():
        assert CAPABILITIES[code]["stt_fixed"], code
    assert "ko" not in auto_detect_languages()
    assert auto_detect_languages() <= interpreter_languages()
    assert auto_detect_languages() == {"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"}
    assert len(realtime_output_languages()) == 13
    assert {"id", "vi"} <= realtime_output_languages()
    assert not {"id", "vi"} & interpreter_languages()
    assert {"name", "stt_fixed", "stt_multi", "tts", "realtime_output"} == set(CAPABILITIES["en"])
    assert as_json().endswith("\n")


def test_other_runtime_sets_come_from_the_table() -> None:
    from src.runtime import latency_modes

    assert latency_modes._REALTIME_OUTPUT_LANGUAGES == realtime_output_languages()
    pytest.importorskip("pipecat")
    from src.runtime.agent_translation import AUTO_DETECTION_LANGUAGES
    from src.runtime.interpreter import SUPPORTED_LANGUAGES
    from src.runtime.realtime_translate import REALTIME_TRANSLATE_OUTPUT_LANGUAGES

    assert AUTO_DETECTION_LANGUAGES == auto_detect_languages()
    assert set(SUPPORTED_LANGUAGES) == interpreter_languages()
    assert REALTIME_TRANSLATE_OUTPUT_LANGUAGES == realtime_output_languages()


@pytest.mark.parametrize(
    ("langs", "direction", "expected"),
    [
        (("en", "ru"), "auto", "multi"),
        (("ru", "en"), "auto", "multi"),
        (("ko", "en"), "auto", "ko"),
        (("en", "tr"), "auto", "en"),
        (("zh", "pl"), "auto", "zh"),
        (("en", "ru"), "one_way", "en"),
        (("ko", "en"), "one_way", "ko"),
        (("hi", "nl"), "auto", "multi"),
    ],
)
def test_stt_language_for(langs, direction, expected) -> None:
    assert stt_language_for(langs, direction) == expected


def test_can_auto_detect() -> None:
    assert can_auto_detect(["en", "ja", "nl"])
    assert not can_auto_detect(["en", "ko"])
    assert not can_auto_detect([])
    with pytest.raises(ValueError):
        stt_language_for([], "auto")


def test_glossary_keeps_instant_on_the_cascaded_engine() -> None:
    assert default_engine_for("instant", "one_way", "ru") == "realtime"
    assert default_engine_for("instant", "one_way", "ru", has_glossary=True) == "cascaded"
    assert default_engine_for("balanced", "one_way", "ru", has_glossary=True) == "cascaded"
    assert default_engine_for("instant", "one_way", "ru", has_glossary=False) == "realtime"


def test_cascaded_engine_requests_multi_only_when_every_language_supports_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("pipecat")
    from pipecat.audio.vad import silero
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3

    from src.runtime import translation_engine
    from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
    from src.runtime.session import RuntimeSession

    class Light:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(silero, "SileroVADAnalyzer", Light)
    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", Light)
    requested: list[str] = []
    real_build_stt = translation_engine.build_stt

    def spy(config, language, keyterms=None):
        requested.append(language)
        return real_build_stt(config, language, keyterms)

    monkeypatch.setattr(translation_engine, "build_stt", spy)

    def build(languages, direction):
        requested.clear()
        config = PipelineConfig(
            "ws://localhost:7880", "token", "room", "dg", "sk", "cart", "voice", "prompt",
            mode="interpreter", interpreter={"languages": languages, "direction": direction},
        )  # fmt: skip
        build_livekit_pipeline(config, RuntimeSession(uuid4(), uuid4(), uuid4()))
        return list(requested)

    assert build(["en", "ru"], "auto") == ["multi"]
    with caplog.at_level(logging.WARNING, logger="src.runtime.translation_engine"):
        assert build(["ko", "en"], "auto") == ["ko"]
    assert "not all in Deepgram's multi set" in caplog.text
    assert build(["ko", "en"], "one_way") == ["ko"]
    assert build(["en", "ru"], "one_way") == ["en"]

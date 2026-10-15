import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from src.runtime.api import SessionStart
from src.runtime.interpreter import (
    InterpreterSettings,
    OpenAITranslator,
    TranslationProcessor,
    build_translation_messages,
    guess_language_from_script,
    normalize_language,
)
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession


class FakeSession:
    """Collects (event_type, payload) tuples the way RuntimeSession.enqueue_event would."""

    def __init__(self) -> None:
        self.event_client = object()
        self.session_id = uuid4()
        self.events: list[tuple[str, dict]] = []

    def event(self, event_type, payload=None):
        return (event_type, payload or {})

    def enqueue_event(self, event) -> None:
        self.events.append(event)

    def of(self, event_type: str) -> list[dict]:
        return [payload for kind, payload in self.events if kind == event_type]


def make_processor(
    translate=None,
    *,
    languages=("en", "ru"),
    direction="auto",
    session=None,
    initial_tts_language="ru",
) -> TranslationProcessor:
    async def default_translate(messages):
        return "T:" + messages[-1]["content"]

    return TranslationProcessor(
        settings=InterpreterSettings(languages=languages, direction=direction),
        translate=translate or default_translate,
        model="test-model",
        session=session,
        initial_tts_language=initial_tts_language,
    )


def transcript(text, language=None, finalized=True) -> TranscriptionFrame:
    return TranscriptionFrame(text, "speaker", "2026-01-01T00:00:00Z", language, finalized=finalized)


def utterance(text, language=None):
    """One speaker turn as the user-turn processor would deliver it."""

    return [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        transcript(text, language),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.05),
    ]


def run(processor, frames):
    return asyncio.run(run_test(processor, frames_to_send=frames, expected_down_frames=None))


def spoken(down_frames):
    return [f.text for f in down_frames if isinstance(f, TTSSpeakFrame)]


# -- language resolution -------------------------------------------------------


def test_normalize_language_strips_region_and_enum() -> None:
    assert normalize_language(Language.EN_US) == "en"
    assert normalize_language("RU-ru") == "ru"
    assert normalize_language(None) is None


def test_settings_validation() -> None:
    settings = InterpreterSettings.from_mapping({"languages": ["en", "ru"], "direction": "one_way"})
    assert settings.languages == ("en", "ru") and settings.direction == "one_way"
    for bad in (
        None,
        {"languages": ["en"]},
        {"languages": ["en", "en"]},
        {"languages": ["en", "ru"], "direction": "both"},
    ):
        with pytest.raises(RuntimeError):
            InterpreterSettings.from_mapping(bad)


def test_auto_resolves_both_directions_from_detected_language() -> None:
    processor = make_processor()
    assert processor.resolve_languages("en", "hello") == ("en", "ru")
    assert processor.resolve_languages(Language.RU, "привет") == ("ru", "en")
    assert processor.resolve_languages("en-US", "hello") == ("en", "ru")


def test_auto_unknown_language_falls_back_to_script_then_previous_then_first() -> None:
    processor = make_processor()
    # Detector says Ukrainian but the pair is en/ru: Cyrillic text means ru.
    assert processor.resolve_languages("uk", "привіт світ") == ("ru", "en")
    assert processor.resolve_languages(None, "hello there") == ("en", "ru")  # languages[0]
    processor._last_source = "ru"
    # Latin text with a pair that shares the script: previous source wins.
    spanish = make_processor(languages=("en", "es"))
    spanish._last_source = "es"
    assert spanish.resolve_languages("pt", "obrigado") == ("es", "en")
    assert spanish.resolve_languages(None, "ok") == ("es", "en")


def test_script_guess_handles_cjk_and_hangul() -> None:
    assert guess_language_from_script("안녕하세요", ("ko", "en")) == "ko"
    assert guess_language_from_script("你好", ("en", "zh")) == "zh"
    assert guess_language_from_script("こんにちは", ("ja", "en")) == "ja"
    assert guess_language_from_script("hello", ("en", "ru")) is None


def test_one_way_ignores_detected_language() -> None:
    processor = make_processor(languages=("ko", "en"), direction="one_way")
    assert processor.resolve_languages("en", "hello") == ("ko", "en")
    assert processor.resolve_languages(None, "안녕") == ("ko", "en")


# -- prompt --------------------------------------------------------------------


def test_prompt_names_languages_and_forbids_answering() -> None:
    messages = build_translation_messages("Where is the station?", "en", "ru")
    system = messages[0]["content"]
    assert messages[0]["role"] == "system" and messages[1] == {
        "role": "user",
        "content": "Where is the station?",
    }
    assert "English (en)" in system and "Russian (ru)" in system
    assert "ONLY the translation" in system
    assert "Never answer questions" in system
    assert "Earlier utterances" not in system


def test_prompt_context_window_keeps_last_four_pairs() -> None:
    history = [("en", f"src{i}", "ru", f"tgt{i}") for i in range(6)]
    system = build_translation_messages("next", "en", "ru", history)[0]["content"]
    assert "src0" not in system and "src1" not in system
    for index in range(2, 6):
        assert f"src{index} => tgt{index}" in system


def test_processor_history_never_exceeds_four_pairs() -> None:
    seen: list[str] = []

    async def translate(messages):
        seen.append(messages[0]["content"])
        return f"t{len(seen)}"

    processor = make_processor(translate)
    processor.push_frame = _discard

    async def go() -> None:
        for index in range(7):
            await processor.process_utterance(_utt(f"s{index}", "en"))

    asyncio.run(go())
    assert len(processor._history) == 4
    assert seen[-1].count("=>") == 4
    assert "s2 =>" in seen[-1] and "s1 =>" not in seen[-1]


async def _discard(frame, direction=None) -> None:
    return None


def _utt(text, language):
    from src.runtime.interpreter import _Utterance

    return _Utterance(text=text, language=language, user_id="u", timestamp="now")


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        class Message:
            content = " Привет "

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


def test_openai_translator_uses_temperature_zero_and_configured_model() -> None:
    completions = FakeCompletions()

    class Client:
        chat = type("Chat", (), {"completions": completions})()

    translator = OpenAITranslator(api_key="k", model="gpt-x", client=Client())
    messages = build_translation_messages("hi", "en", "ru")
    assert asyncio.run(translator(messages)) == " Привет "
    assert completions.calls == [{"model": "gpt-x", "messages": messages, "temperature": 0}]


# -- pipeline behaviour --------------------------------------------------------


def test_translates_in_order_even_when_earlier_translation_is_slower() -> None:
    delays = {"first": 0.15, "second": 0.0, "third": 0.0}

    async def translate(messages):
        text = messages[-1]["content"]
        await asyncio.sleep(delays[text])
        return f"tr-{text}"

    processor = make_processor(translate)
    frames = []
    for text in ("first", "second", "third"):
        frames += utterance(text, "en")
    down, _ = run(processor, frames)

    assert spoken(down) == ["tr-first", "tr-second", "tr-third"]


def test_segments_of_one_turn_are_joined_and_interims_are_forwarded() -> None:
    received: list[str] = []

    async def translate(messages):
        received.append(messages[-1]["content"])
        return "ok"

    processor = make_processor(translate)
    frames = [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        InterimTranscriptionFrame("hel", "speaker", "now", Language.EN),
        transcript("hello there,", "en", finalized=False),
        transcript("how are you", "en"),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.05),
    ]
    down, _ = run(processor, frames)

    assert received == ["hello there, how are you"]
    assert not any(isinstance(f, TranscriptionFrame) for f in down)
    assert any(isinstance(f, InterimTranscriptionFrame) for f in down)


def test_segment_arriving_after_stop_signal_is_still_translated() -> None:
    received: list[str] = []

    async def translate(messages):
        received.append(messages[-1]["content"])
        return "ok"

    processor = make_processor(translate)
    frames = [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        transcript("late segment", "en"),
        SleepFrame(sleep=0.05),
    ]
    run(processor, frames)
    assert received == ["late segment"]


def test_tts_language_update_only_when_target_changes() -> None:
    processor = make_processor(initial_tts_language="ru")
    frames = []
    for text, language in (
        ("one", "en"),  # -> ru, already ru
        ("two", "en"),  # -> ru
        ("три", "ru"),  # -> en : change
        ("четыре", "ru"),  # -> en
        ("five", "en"),  # -> ru : change
    ):
        frames += utterance(text, language)
    down, _ = run(processor, frames)

    sequence = []
    for frame in down:
        if isinstance(frame, TTSUpdateSettingsFrame):
            sequence.append(("lang", frame.delta.language))
        elif isinstance(frame, TTSSpeakFrame):
            sequence.append(("say", frame.text))
    assert sequence == [
        ("say", "T:one"),
        ("say", "T:two"),
        ("lang", "en"),
        ("say", "T:три"),
        ("say", "T:четыре"),
        ("lang", "ru"),
        ("say", "T:five"),
    ]
    assert all(
        f.append_to_context is False for f in down if isinstance(f, TTSSpeakFrame)
    )


def test_translation_errors_are_skipped_without_crashing() -> None:
    calls = 0

    async def translate(messages):
        nonlocal calls
        calls += 1
        if messages[-1]["content"] == "boom":
            raise RuntimeError("provider down")
        if messages[-1]["content"] == "blank":
            return "   "
        return "fine"

    session = FakeSession()
    processor = make_processor(translate, session=session)
    frames = []
    for text in ("boom", "blank", "good"):
        frames += utterance(text, "en")
    down, _ = run(processor, frames)

    assert calls == 3
    assert spoken(down) == ["fine"]
    assistant = [p for p in session.of("TranscriptFinal") if p["role"] == "assistant"]
    assert [p["content"] for p in assistant] == ["fine"]


def test_events_use_roles_languages_and_translation_latency() -> None:
    session = FakeSession()
    processor = make_processor(session=session)
    frames = utterance("Hello", Language.EN) + utterance("Привет", "ru")
    run(processor, frames)

    transcripts = session.of("TranscriptFinal")
    assert [(p["role"], p["content"], p["primary_language"], p["detected_languages"]) for p in transcripts] == [
        ("user", "Hello", "en", ["en"]),
        ("assistant", "T:Hello", "ru", ["ru"]),
        ("user", "Привет", "ru", ["ru"]),
        ("assistant", "T:Привет", "en", ["en"]),
    ]
    assert transcripts[0]["user_id"] == "speaker"
    latency = session.of("LatencyMeasured")
    assert len(latency) == 2
    assert latency[0]["metric"] == "translation_ms"
    assert latency[0]["processor"] == "interpreter"
    assert latency[0]["model"] == "test-model"
    assert isinstance(latency[0]["value_ms"], float) and latency[0]["value_ms"] >= 0


def test_one_way_translates_to_fixed_target() -> None:
    session = FakeSession()
    processor = make_processor(
        languages=("ko", "en"), direction="one_way", session=session, initial_tts_language="en"
    )
    down, _ = run(processor, utterance("안녕하세요", None))

    assert spoken(down) == ["T:안녕하세요"]
    assert not any(isinstance(f, TTSUpdateSettingsFrame) for f in down)
    user, assistant = session.of("TranscriptFinal")
    assert (user["primary_language"], assistant["primary_language"]) == ("ko", "en")


def test_event_loss_never_breaks_interpretation() -> None:
    class ClosedSession(FakeSession):
        def enqueue_event(self, event) -> None:
            raise RuntimeError("closed")

    down, _ = run(make_processor(session=ClosedSession()), utterance("Hello", "en"))
    assert spoken(down) == ["T:Hello"]


# -- pipeline builder ----------------------------------------------------------


@pytest.fixture
def light_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer)
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)


def _config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        "ws://localhost:7880", "token", "room", "deepgram", "openai", "cartesia", "voice",
        "prompt", **kwargs,
    )  # fmt: skip


def _processors(worker):
    from pipecat.pipeline.pipeline import Pipeline

    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    return [
        p for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]


def test_builder_chooses_interpreter_pipeline_without_llm_or_aggregators(light_models) -> None:
    config = _config(
        mode="interpreter",
        interpreter={"languages": ["en", "ru"], "direction": "auto"},
        provider_settings={"llm": {"model": "gpt-test"}, "stt": {"model": "nova-3"}},
    )
    _, worker = build_livekit_pipeline(config, RuntimeSession(uuid4(), uuid4(), uuid4()))
    processors = _processors(worker)
    names = [type(p).__name__ for p in processors]

    assert names == [
        "LiveKitInputTransport",
        "VADProcessor",
        "DeepgramSTTService",
        "UserTurnProcessor",
        "TranslationProcessor",
        "CartesiaTTSService",
        "RuntimeEventProcessor",
        "LiveKitOutputTransport",
    ]
    # Exactly one event observer, and it never sees transcription frames (the
    # translator consumes them), so transcripts cannot be emitted twice.
    assert names.count("RuntimeEventProcessor") == 1
    assert not any("LLM" in name or "Aggregator" in name for name in names)
    stt = processors[2]
    assert stt._settings.language == "multi" and stt._settings.model == "nova-3"
    translator = processors[4]
    assert translator._model == "gpt-test"
    tts = processors[5]
    assert tts._settings.voice == "voice"


def test_interpreter_pipeline_disables_interruptions_and_fixes_one_way_source(light_models) -> None:
    config = _config(
        mode="interpreter",
        interpreter={"languages": ["ko", "en"], "direction": "one_way"},
    )
    _, worker = build_livekit_pipeline(config)
    processors = _processors(worker)
    assert not any(type(p).__name__ == "RuntimeEventProcessor" for p in processors)
    assert processors[2]._settings.language == "ko"
    assert processors[5]._settings.language == "en"
    user_turns = processors[3]
    starts = user_turns._user_turn_controller._user_turn_strategies.start
    assert starts and all(strategy._enable_interruptions is False for strategy in starts)


def test_builder_keeps_agent_pipeline_for_agent_mode(light_models) -> None:
    _, worker = build_livekit_pipeline(_config(), RuntimeSession(uuid4(), uuid4(), uuid4()))
    names = [type(p).__name__ for p in _processors(worker)]
    assert "OpenAILLMService" in names and "TranslationProcessor" not in names


# -- config / contract ---------------------------------------------------------


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "LIVEKIT_URL": "ws://localhost:7880",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }.items():
        monkeypatch.setenv(key, value)


def test_config_carries_mode_and_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    base = {"room_name": "room", "livekit_token": "t"}
    interpreter = {"languages": ["en", "ru"], "direction": "auto"}
    config = PipelineConfig.from_environment({**base, "mode": "interpreter", "interpreter": interpreter})
    assert config.mode == "interpreter" and config.interpreter == interpreter
    legacy = PipelineConfig.from_environment(base)
    assert legacy.mode == "agent" and legacy.interpreter is None
    agent = PipelineConfig.from_environment({**base, "mode": "agent", "interpreter": None})
    assert agent.mode == "agent" and agent.interpreter is None


def test_config_rejects_bad_mode_and_missing_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    base = {"room_name": "room", "livekit_token": "t"}
    with pytest.raises(RuntimeError, match="unsupported session mode"):
        PipelineConfig.from_environment({**base, "mode": "translator"})
    with pytest.raises(RuntimeError, match="interpreter"):
        PipelineConfig.from_environment({**base, "mode": "interpreter", "interpreter": None})
    with pytest.raises(RuntimeError, match="distinct"):
        PipelineConfig.from_environment(
            {**base, "mode": "interpreter", "interpreter": {"languages": ["en", "en"]}}
        )


def test_session_start_accepts_old_and_new_contracts() -> None:
    old = {
        "session_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "room",
    }
    legacy = SessionStart(**old)
    assert legacy.mode == "agent" and legacy.interpreter is None
    new = SessionStart(
        **old,
        mode="interpreter",
        interpreter={"languages": ["en", "ru"], "direction": "auto"},
    )
    assert new.mode == "interpreter"
    assert new.interpreter == {"languages": ["en", "ru"], "direction": "auto"}

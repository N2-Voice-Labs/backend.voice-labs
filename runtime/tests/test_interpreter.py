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
from src.runtime import interpreter as interpreter_module
from src.runtime.interpreter import (
    InterpreterSettings,
    OpenAITranslator,
    SentenceChunker,
    TranslationProcessor,
    as_streaming_translator,
    build_translation_messages,
    estimate_max_tokens,
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
    """Mimics ``chat.completions.create(stream=True)``: an async iterator of chunks."""

    def __init__(self, deltas=(" При", "вет ", None)) -> None:
        self.calls: list[dict] = []
        self.deltas = deltas
        self.closed = False

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        completions = self

        class Stream:
            def __aiter__(self):
                return self._iterate()

            async def _iterate(self):
                for delta in completions.deltas:
                    delta_obj = type("Delta", (), {"content": delta})()
                    yield type("Chunk", (), {"choices": [type("Choice", (), {"delta": delta_obj})()]})()
                yield type("Chunk", (), {"choices": []})()  # trailing usage-style chunk

            async def close(self):
                completions.closed = True

        return Stream()


async def collect(iterator) -> list[str]:
    return [delta async for delta in iterator]


def test_openai_translator_streams_with_temperature_zero_model_and_token_cap() -> None:
    completions = FakeCompletions()

    class Client:
        chat = type("Chat", (), {"completions": completions})()

    translator = OpenAITranslator(api_key="k", model="gpt-x", client=Client())
    messages = build_translation_messages("hi", "en", "ru")
    assert asyncio.run(collect(translator(messages))) == [" При", "вет "]
    assert completions.calls == [
        {
            "model": "gpt-x",
            "messages": messages,
            "temperature": 0,
            "stream": True,
            "max_tokens": 64,
        }
    ]
    assert completions.closed


def test_default_model_is_gpt_4_1_mini() -> None:
    assert interpreter_module.DEFAULT_TRANSLATION_MODEL == "gpt-4.1-mini"
    assert OpenAITranslator(api_key="k", client=object()).model == "gpt-4.1-mini"


def test_max_tokens_is_twice_estimated_input_clamped() -> None:
    assert estimate_max_tokens("hi") == 64  # min
    assert estimate_max_tokens("x" * 100) == 100
    assert estimate_max_tokens("x" * 101) == 102
    assert estimate_max_tokens("x" * 5000) == 512  # max


def test_adapter_wraps_plain_translator_and_passes_streams_through() -> None:
    async def plain(messages):
        return "whole"

    async def empty(messages):
        return ""

    async def streaming(messages):
        yield "a"
        yield "b"

    assert asyncio.run(collect(as_streaming_translator(plain)([]))) == ["whole"]
    assert asyncio.run(collect(as_streaming_translator(empty)([]))) == []
    assert asyncio.run(collect(as_streaming_translator(streaming)([]))) == ["a", "b"]


# -- chunking -------------------------------------------------------------------


def chunks_of(text: str, step: int | None = None) -> list[str]:
    chunker = SentenceChunker()
    out: list[str] = []
    pieces = [text] if step is None else [text[i : i + step] for i in range(0, len(text), step)]
    for piece in pieces:
        out += chunker.feed(piece)
    return out + chunker.flush()


def test_chunker_splits_sentences_on_terminators_followed_by_space_or_end() -> None:
    text = "Hello there. How are you? Fine! Wait... Really\u2026 Yes"
    expected = ["Hello there.", "How are you?", "Fine!", "Wait...", "Really\u2026", "Yes"]
    assert chunks_of(text) == expected
    # Same result however the stream is sliced, including one character at a time.
    for step in (1, 2, 3, 7):
        assert chunks_of(text, step) == expected


def test_chunker_does_not_split_decimals_or_short_clauses() -> None:
    assert chunks_of("It costs 3.5 dollars, maybe 4,000 later") == [
        "It costs 3.5 dollars, maybe 4,000 later"
    ]
    # Fewer than 40 chars pending: a comma, semicolon or colon is not a boundary.
    assert chunks_of("Well, yes; sure: ok") == ["Well, yes; sure: ok"]


def test_chunker_splits_clauses_once_forty_chars_are_pending() -> None:
    text = "When the meeting finally starts tomorrow morning, please bring the report; thanks"
    first = text.index(",") + 1
    assert first >= 40
    assert chunks_of(text) == [
        "When the meeting finally starts tomorrow morning,",
        "please bring the report; thanks",
    ]
    long_semicolon = "x" * 38 + "ab; and then more words follow"
    assert chunks_of(long_semicolon) == ["x" * 38 + "ab;", "and then more words follow"]
    assert chunks_of("x" * 36 + "ab: rest") == ["x" * 36 + "ab: rest"]  # 39 chars incl. colon
    assert chunks_of("x" * 37 + "ab: rest") == ["x" * 37 + "ab:", "rest"]  # 40 chars


def test_chunker_handles_cjk_punctuation_without_spaces() -> None:
    assert chunks_of("\u4f60\u597d\u3002\u4eca\u5929\u600e\u4e48\u6837\uff1f\u5f88\u597d\uff01\u8c22\u8c22") == [
        "\u4f60\u597d\u3002",
        "\u4eca\u5929\u600e\u4e48\u6837\uff1f",
        "\u5f88\u597d\uff01",
        "\u8c22\u8c22",
    ]
    # Trailing closing quotes stay with their sentence.
    assert chunks_of("\u300c\u3053\u3093\u306b\u3061\u306f\u3002\u300d\u3068\u8a00\u3063\u305f\u3002") == [
        "\u300c\u3053\u3093\u306b\u3061\u306f\u3002\u300d",
        "\u3068\u8a00\u3063\u305f\u3002",
    ]
    assert chunks_of("\u3053\u3093\u306b\u3061\u306f\u3002\u3053\u3093\u306b\u3061\u306f\u3002", 1) == [
        "\u3053\u3093\u306b\u3061\u306f\u3002",
        "\u3053\u3093\u306b\u3061\u306f\u3002",
    ]


def test_chunker_waits_for_text_after_a_trailing_terminator() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("Version 3.") == []  # could be "3.5"
    assert chunker.feed("5 is out.") == []
    assert chunker.feed(" Next") == ["Version 3.5 is out."]
    assert chunker.flush() == ["Next"]
    assert chunker.flush() == []


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
    assert [p["metric"] for p in latency] == [
        "translation_first_chunk_ms",
        "translation_ms",
        "translation_first_chunk_ms",
        "translation_ms",
    ]
    for payload in latency:
        assert payload["processor"] == "interpreter"
        assert payload["model"] == "test-model"
        assert isinstance(payload["value_ms"], float) and payload["value_ms"] >= 0
    assert latency[0]["value_ms"] <= latency[1]["value_ms"]


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


def test_streamed_chunks_are_spoken_in_order_and_full_text_emitted_once() -> None:
    async def translate(messages):
        for delta in ("Privet. ", "Kak ", "dela? ", "Horosho"):
            yield delta

    session = FakeSession()
    processor = make_processor(translate, session=session)
    down, _ = run(processor, utterance("Hello", "en"))

    assert spoken(down) == ["Privet.", "Kak dela?", "Horosho"]
    assistant = [p for p in session.of("TranscriptFinal") if p["role"] == "assistant"]
    assert [p["content"] for p in assistant] == ["Privet. Kak dela? Horosho"]


def _process_directly(processor):
    frames: list = []

    async def capture(frame, direction=None) -> None:
        frames.append(frame)

    processor.push_frame = capture
    return frames


def test_first_chunk_is_spoken_before_the_translation_completes() -> None:
    async def scenario():
        release = asyncio.Event()

        async def translate(messages):
            yield "First sentence. Sec"
            await release.wait()  # the rest of the translation is "slow"
            yield "ond sentence."

        session = FakeSession()
        processor = make_processor(translate, session=session)
        frames = _process_directly(processor)
        task = asyncio.create_task(processor.process_utterance(_utt("hello", "en")))
        for _ in range(100):
            await asyncio.sleep(0.005)
            if any(isinstance(f, TTSSpeakFrame) for f in frames):
                break
        assert [f.text for f in frames if isinstance(f, TTSSpeakFrame)] == ["First sentence."]
        assert not task.done()
        # Only the first-chunk latency exists so far; the assistant transcript is still
        # pending and no complete-translation metric has been emitted.
        assert [p["metric"] for p in session.of("LatencyMeasured")] == ["translation_first_chunk_ms"]
        assert not [p for p in session.of("TranscriptFinal") if p["role"] == "assistant"]
        release.set()
        await task
        return frames, session

    frames, session = asyncio.run(scenario())
    assert [f.text for f in frames if isinstance(f, TTSSpeakFrame)] == [
        "First sentence.",
        "Second sentence.",
    ]
    metrics = {p["metric"]: p["value_ms"] for p in session.of("LatencyMeasured")}
    assert set(metrics) == {"translation_first_chunk_ms", "translation_ms"}
    assert metrics["translation_first_chunk_ms"] < metrics["translation_ms"]


def test_language_update_precedes_first_chunk_only() -> None:
    async def translate(messages):
        yield "One. Two. "
        yield "Three."

    processor = make_processor(translate, initial_tts_language="en")
    down, _ = run(processor, utterance("hello", "en"))  # en -> ru, TTS currently en

    kinds = [
        ("lang", f.delta.language) if isinstance(f, TTSUpdateSettingsFrame) else ("say", f.text)
        for f in down
        if isinstance(f, (TTSUpdateSettingsFrame, TTSSpeakFrame))
    ]
    assert kinds == [("lang", "ru"), ("say", "One."), ("say", "Two."), ("say", "Three.")]


def test_streaming_utterances_stay_in_order_when_the_first_is_slow() -> None:
    async def translate(messages):
        text = messages[-1]["content"]
        if text == "first":
            await asyncio.sleep(0.12)
        yield f"{text}-a. "
        await asyncio.sleep(0.01)
        yield f"{text}-b."

    down, _ = run(make_processor(translate), sum((utterance(t, "en") for t in ("first", "second")), []))
    assert spoken(down) == ["first-a.", "first-b.", "second-a.", "second-b."]


def test_mid_stream_error_speaks_nothing_further_and_continues() -> None:
    async def translate(messages):
        text = messages[-1]["content"]
        if text == "boom":
            yield "Spoken. Half a sen"
            raise RuntimeError("stream dropped")
        yield f"ok-{text}."

    session = FakeSession()
    down, _ = run(
        make_processor(translate, session=session),
        sum((utterance(t, "en") for t in ("boom", "next")), []),
    )

    assert spoken(down) == ["Spoken.", "ok-next."]  # the unfinished fragment is never spoken
    assistant = [p["content"] for p in session.of("TranscriptFinal") if p["role"] == "assistant"]
    assert assistant == ["Spoken.", "ok-next."]
    metrics = [p["metric"] for p in session.of("LatencyMeasured")]
    assert metrics.count("translation_ms") == 1  # only the successful utterance


def test_failed_stream_does_not_enter_translation_history() -> None:
    async def translate(messages):
        yield "Partial. "
        raise RuntimeError("boom")

    processor = make_processor(translate)
    processor.push_frame = _discard
    asyncio.run(processor.process_utterance(_utt("hello", "en")))
    assert len(processor._history) == 0


# -- turn end ------------------------------------------------------------------


def test_finalized_segment_after_stop_is_translated_immediately(monkeypatch) -> None:
    monkeypatch.setattr(interpreter_module, "LATE_SEGMENT_GRACE_SECONDS", 5.0)
    received: list[str] = []

    async def translate(messages):
        received.append(messages[-1]["content"])
        return "ok"

    frames = [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        transcript("late but finalized", "en", finalized=True),
        SleepFrame(sleep=0.05),  # far shorter than the grace period
    ]
    run(make_processor(translate), frames)
    assert received == ["late but finalized"]


def test_late_non_finalized_segment_waits_for_its_finalized_continuation(monkeypatch) -> None:
    monkeypatch.setattr(interpreter_module, "LATE_SEGMENT_GRACE_SECONDS", 5.0)
    received: list[str] = []

    async def translate(messages):
        received.append(messages[-1]["content"])
        return "ok"

    frames = [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        transcript("where is", "en", finalized=False),
        SleepFrame(sleep=0.05),
        transcript("the station", "en", finalized=True),
        SleepFrame(sleep=0.05),
    ]
    run(make_processor(translate), frames)
    assert received == ["where is the station"]  # one sentence, one translation


def test_late_non_finalized_segment_is_translated_after_the_grace_period(monkeypatch) -> None:
    monkeypatch.setattr(interpreter_module, "LATE_SEGMENT_GRACE_SECONDS", 0.05)
    received: list[str] = []

    async def translate(messages):
        received.append(messages[-1]["content"])
        return "ok"

    frames = [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        transcript("never finalized", "en", finalized=False),
        SleepFrame(sleep=0.25),
    ]
    run(make_processor(translate), frames)
    assert received == ["never finalized"]


def test_pending_late_segment_is_not_merged_into_the_next_turn(monkeypatch) -> None:
    monkeypatch.setattr(interpreter_module, "LATE_SEGMENT_GRACE_SECONDS", 5.0)
    received: list[str] = []

    async def translate(messages):
        received.append(messages[-1]["content"])
        return "ok"

    frames = [
        UserStartedSpeakingFrame(),
        UserStoppedSpeakingFrame(),
        transcript("old turn", "en", finalized=False),
        SleepFrame(sleep=0.02),
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        transcript("new turn", "en"),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.05),
    ]
    run(make_processor(translate), frames)
    assert received == ["old turn", "new turn"]


def test_one_sentence_is_never_translated_twice() -> None:
    """A finalized segment mid-turn, the stop signal and a duplicate stop yield one call."""

    received: list[str] = []

    async def translate(messages):
        received.append(messages[-1]["content"])
        yield "Where is the station?"

    frames = [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        transcript("where is", "en", finalized=False),
        transcript("the station", "en", finalized=True),  # VAD finalize while turn is open
        SleepFrame(sleep=0.05),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.4),  # longer than the late-segment grace period
    ]
    down, _ = run(make_processor(translate), frames)
    assert received == ["where is the station"]
    assert spoken(down) == ["Where is the station?"]


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
    stops = user_turns._user_turn_controller._user_turn_strategies.stop
    assert [type(strategy).__name__ for strategy in stops] == ["SpeechTimeoutUserTurnStopStrategy"]
    assert stops[0]._user_speech_timeout == 0.3


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

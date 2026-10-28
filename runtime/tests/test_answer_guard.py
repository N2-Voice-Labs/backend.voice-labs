"""Answer guard (ST-12): empty or broken answers never produce silence."""

import asyncio

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from src.runtime.agent_translation import (
    AgentTranslationSettings,
    CallerLanguageState,
    OutboundTranslator,
)
from src.runtime.answer_guard import MAX_SPOKEN_CHARS, AnswerGuard
from src.runtime.fallback_lines import CHECKING, REPEAT, line
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor


def run(processor, frames):
    down, _ = asyncio.run(
        run_test(processor, frames_to_send=frames, expected_down_frames=None)
    )
    return list(down)


def speaks(down):
    return [f for f in down if isinstance(f, TTSSpeakFrame)]


def texts(down):
    return [f.text for f in down if isinstance(f, LLMTextFrame)]


def response(*deltas):
    return [
        LLMFullResponseStartFrame(),
        *[LLMTextFrame(d) for d in deltas],
        LLMFullResponseEndFrame(),
    ]


def test_empty_response_speaks_repeat_line_in_caller_language() -> None:
    empty = []
    guard = AnswerGuard(language=lambda: "ru", on_empty=empty.append)

    down = run(guard, response())

    assert [s.text for s in speaks(down)] == [line(REPEAT, "ru")]
    assert speaks(down)[0].append_to_context is False
    assert any(isinstance(f, LLMFullResponseEndFrame) for f in down)
    assert empty == ["ru"]


def test_whitespace_only_response_gets_the_fallback() -> None:
    empty = []
    guard = AnswerGuard(language="en", on_empty=empty.append)

    down = run(guard, response("  ", "\n\t", " "))

    assert [s.text for s in speaks(down)] == [line(REPEAT, "en")]
    assert empty == ["en"]


def test_function_call_only_response_gets_no_fallback() -> None:
    empty = []
    guard = AnswerGuard(language="en", on_empty=empty.append)
    frames = [
        LLMFullResponseStartFrame(),
        FunctionCallInProgressFrame(
            function_name="lookup", tool_call_id="c1", arguments={}
        ),
        LLMFullResponseEndFrame(),
    ]

    down = run(guard, frames)

    assert speaks(down) == []
    assert empty == []


def test_text_response_passes_through_unchanged() -> None:
    empty = []
    guard = AnswerGuard(language="en", on_empty=empty.append)

    down = run(guard, response("Hello there.", " How can I help?"))

    assert texts(down) == ["Hello there.", " How can I help?"]
    assert speaks(down) == []
    assert empty == []


def test_long_answer_is_cut_after_the_next_sentence_boundary() -> None:
    truncated = []
    guard = AnswerGuard(
        language="en", max_spoken_chars=10, on_truncated=truncated.append
    )

    down = run(guard, response("Alpha beta gamma. Delta", " epsilon.", " Zeta."))

    assert texts(down) == ["Alpha beta gamma."]
    assert truncated == [17]
    assert speaks(down) == []


def test_cap_is_reached_mid_sentence_and_cut_at_the_next_boundary() -> None:
    truncated = []
    guard = AnswerGuard(
        language="en", max_spoken_chars=10, on_truncated=truncated.append
    )

    down = run(guard, response("Alpha beta", "gamma delta. Epsilon", " more."))

    assert texts(down) == ["Alpha beta", "gamma delta."]
    assert truncated == [22]


def test_answer_under_the_cap_is_not_truncated() -> None:
    truncated = []
    guard = AnswerGuard(language="en", on_truncated=truncated.append)
    assert MAX_SPOKEN_CHARS == 600

    down = run(guard, response("A short answer. ", "Another sentence."))

    assert texts(down) == ["A short answer. ", "Another sentence."]
    assert truncated == []


def test_interruption_mid_response_discards_it_without_fallback() -> None:
    empty = []
    guard = AnswerGuard(language="en", on_empty=empty.append)
    frames = [
        LLMFullResponseStartFrame(),
        SleepFrame(sleep=0.02),
        InterruptionFrame(),
        LLMFullResponseEndFrame(),
    ]

    down = run(guard, frames)

    assert speaks(down) == []
    assert empty == []


def test_interruption_resets_the_length_cap_for_the_next_response() -> None:
    guard = AnswerGuard(language="en", max_spoken_chars=5)
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("abcdefgh"),
        SleepFrame(sleep=0.02),
        InterruptionFrame(),
        *response("ab cd. Ef"),
    ]

    down = run(guard, frames)

    # The first response is past the cap; the reset lets the next one cut at its own "."
    assert texts(down) == ["abcdefgh", "ab cd."]


def test_interruption_then_empty_response_speaks_exactly_one_fallback() -> None:
    empty = []
    guard = AnswerGuard(language="ja", on_empty=empty.append)
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("partial"),
        SleepFrame(sleep=0.02),
        InterruptionFrame(),
        *response(),
    ]

    down = run(guard, frames)

    assert [s.text for s in speaks(down)] == [line(REPEAT, "ja")]
    assert empty == ["ja"]


def test_frames_outside_a_response_pass_through() -> None:
    guard = AnswerGuard(language="en")

    down = run(guard, [LLMTextFrame("stray"), LLMFullResponseEndFrame()])

    assert texts(down) == ["stray"]
    assert speaks(down) == []


def test_context_only_text_does_not_count_as_spoken() -> None:
    guard = AnswerGuard(language="en")
    context_only = LLMTextFrame("hidden")
    context_only.skip_tts = True

    down = run(
        guard, [LLMFullResponseStartFrame(), context_only, LLMFullResponseEndFrame()]
    )

    assert [s.text for s in speaks(down)] == [line(REPEAT, "en")]


def test_fallback_lines_cover_every_language_and_default_to_english() -> None:
    languages = [
        "en",
        "ru",
        "ko",
        "es",
        "fr",
        "de",
        "ja",
        "zh",
        "pt",
        "it",
        "nl",
        "hi",
        "tr",
        "pl",
    ]
    for language in languages:
        for kind in (REPEAT, CHECKING):
            assert line(kind, language).strip(), (kind, language)
    assert line(REPEAT, "en") == "Sorry, could you say that again?"
    assert line(CHECKING, "en") == "Let me check that again…"
    assert line(REPEAT, "ru") != line(REPEAT, "en")
    assert line(REPEAT, "pt-BR") == line(REPEAT, "pt")
    assert line(CHECKING, "ru_RU") == line(CHECKING, "ru")
    for unsupported in ("xx", "", None):
        assert line(REPEAT, unsupported) == line(REPEAT, "en")
    with pytest.raises(ValueError):
        line("unknown", "en")


def test_interpreter_reports_empty_translation_through_the_hook() -> None:
    async def translate(messages):
        return "   " if messages[-1]["content"] == "blank" else "fine"

    def utterance(text):
        return [
            UserStartedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            TranscriptionFrame(
                text, "speaker", "2026-01-01T00:00:00Z", "en", finalized=True
            ),
            SleepFrame(sleep=0.02),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),
        ]

    empty = []
    processor = TranslationProcessor(
        settings=InterpreterSettings(languages=("en", "ru"), direction="auto"),
        translate=translate,
        model="test-model",
        on_empty_translation=empty.append,
    )

    down = run(processor, utterance("blank") + utterance("good"))

    assert [s.text for s in speaks(down)] == ["fine"]
    assert len(empty) == 1
    assert empty[0]["text"] == "blank"
    assert empty[0]["source_language"] == "en"


def test_outbound_reports_an_empty_sentence_translation_through_the_hook() -> None:
    async def translate(messages):
        return ""

    settings = AgentTranslationSettings(
        working_language="en", caller_languages=("ru", "es")
    )
    state = CallerLanguageState(settings)
    state.language = "ru"
    empty = []
    outbound = OutboundTranslator(
        state=state, translate=translate, model="m", on_empty_translation=empty.append
    )

    down = run(outbound, response("Hello there."))

    assert speaks(down) == []
    assert empty == [
        {"text": "Hello there.", "source_language": "en", "target_language": "ru"}
    ]

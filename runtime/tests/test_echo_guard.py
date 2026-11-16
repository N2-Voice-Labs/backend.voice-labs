"""Echo guard: the call's own translated speech coming back through a microphone must not loop."""

import asyncio

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import TTSSpeakFrame

from src.runtime.echo_guard import EchoGuard, normalize
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor, _Utterance
from test_interpreter_participants import ALICE, BOB, FakeSession, directory_with, spec


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_normalize_ignores_case_punctuation_and_spacing_in_any_script() -> None:
    assert normalize("Clipping is NOT possible!") == "clippingisnotpossible"
    assert normalize("클리핑이 불가능합니다.") == "클리핑이불가능합니다"


def test_recent_spoken_text_is_an_echo_until_the_window_passes() -> None:
    clock = Clock()
    guard = EchoGuard(window_s=10, clock=clock)
    guard.record("도와드릴 수 없습니다.")
    assert guard.is_echo("도와드릴 수 없습니다")
    assert guard.is_echo("도와드릴 수 없습니다요")  # recognizer noise still matches
    assert not guard.is_echo("서류 마감일은 십일월 십이일입니다")
    clock.now += 11
    assert not guard.is_echo("도와드릴 수 없습니다")


def test_short_phrases_are_never_dropped() -> None:
    guard = EchoGuard()
    guard.record("Hello.")
    assert not guard.is_echo("Hello")  # answering a greeting with the same word is normal


def test_off_by_env(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_ECHO_GUARD", "0")
    assert EchoGuard.from_env() is None
    monkeypatch.setenv("RUNTIME_ECHO_GUARD", "1")
    monkeypatch.setenv("RUNTIME_ECHO_GUARD_SECS", "4")
    assert EchoGuard.from_env() is not None


def test_the_2026_11_13_loop_stops_at_the_first_echo() -> None:
    """Shared laptop mic: Bob's Korean, translated to English for Alice, comes back in through the mic."""
    table = {
        "도와드릴 수 없습니다.": "I cannot help you with that.",
        "I cannot help you with that.": "그 일은 도와드릴 수 없습니다.",
        "When is the deadline for documents?": "서류 마감일은 언제인가요?",
    }

    async def translate(messages):
        return table[messages[-1]["content"]]

    session = FakeSession()
    directory = directory_with((ALICE, spec("en")), (BOB, spec("ko")))
    processor = TranslationProcessor(
        settings=InterpreterSettings(languages=directory.languages, direction="auto"),
        translate=translate, model="test-model", session=session,
        initial_tts_language="ko", participants=directory, echo_guard=EchoGuard(),
    )
    said: list[str] = []

    async def push_frame(frame, direction=None):
        if isinstance(frame, TTSSpeakFrame):
            said.append(frame.text)

    processor.push_frame = push_frame  # type: ignore[method-assign]

    async def scenario():
        await processor.process_utterance(_Utterance("도와드릴 수 없습니다.", "ko", BOB, "t1"))
        # The English translation played on Alice's speakers and her mic transcribed it.
        await processor.process_utterance(_Utterance("I cannot help you with that.", "en", ALICE, "t2"))
        # Alice's real next question still goes through.
        await processor.process_utterance(_Utterance("When is the deadline for documents?", "en", ALICE, "t3"))

    asyncio.run(scenario())
    assert said == ["I cannot help you with that.", "서류 마감일은 언제인가요?"]
    users = [p["content"] for p in session.of("TranscriptFinal") if p["role"] == "user"]
    assert users == ["도와드릴 수 없습니다.", "When is the deadline for documents?"]

"""ST-12 / ST-02a wiring: the answer guard in the agent pipelines, the TurnDebug observer
(agent mode only), the guard/empty-translation events, and ``turn_id`` on transcripts."""

import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import SleepFrame, run_test

from src.runtime import pipeline as pipeline_module
from src.runtime.agent_translation import (
    AgentTranslationSettings,
    CallerLanguageState,
    OutboundTranslator,
)
from src.runtime.answer_guard import MAX_SPOKEN_CHARS, AnswerGuard
from src.runtime.events import EVENT_TYPES
from src.runtime.guard_events import translation_empty_hook
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession
from src.runtime.turn_ids import turn_state


class FakeSession:
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


@pytest.fixture
def light_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.audio.turn.smart_turn import local_smart_turn_v3
    from pipecat.audio.vad import silero

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(local_smart_turn_v3, "LocalSmartTurnAnalyzerV3", LightweightAnalyzer)
    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)


@pytest.fixture
def turn_debug_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        pipeline_module, "watch_turn_debug", lambda *args: calls.append(args)
    )
    return calls


def _config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        "ws://localhost:7880", "token", "room", "deepgram", "openai", "cartesia", "voice",
        "prompt", **kwargs,
    )  # fmt: skip


def _processors(worker):
    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    return [
        p for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]


def _names(processors) -> list[str]:
    return [type(p).__name__ for p in processors]


def run(processor, frames):
    down, _ = asyncio.run(run_test(processor, frames_to_send=frames, expected_down_frames=None))
    return list(down)


def response(*deltas):
    return [LLMFullResponseStartFrame(), *[LLMTextFrame(d) for d in deltas], LLMFullResponseEndFrame()]


# -- pipeline construction ------------------------------------------------------------


def test_agent_pipeline_guards_the_answer_before_tts_and_watches_turn_debug(
    light_models, turn_debug_calls
) -> None:
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    config = _config(provider_settings={"tts": {"language": "ru"}})
    _, worker = build_livekit_pipeline(config, session)
    processors = _processors(worker)
    names = _names(processors)

    guard_index = names.index("AnswerGuard")
    assert names[guard_index + 1] == "CartesiaTTSService"
    assert names[guard_index - 1] == "RuntimeEventProcessor"
    guard = processors[guard_index]
    assert isinstance(guard, AnswerGuard)
    assert guard._language == "ru"
    assert guard._max_spoken_chars == MAX_SPOKEN_CHARS
    assert guard._on_empty is not None and guard._on_truncated is not None
    assert turn_debug_calls == [(worker, session, config.providers)]


def test_agent_pipeline_guard_defaults_to_no_language_without_a_configured_one(
    light_models, turn_debug_calls
) -> None:
    _, worker = build_livekit_pipeline(_config(), RuntimeSession(uuid4(), uuid4(), uuid4()))
    guard = next(p for p in _processors(worker) if isinstance(p, AnswerGuard))
    assert guard._language is None  # the guard speaks English then


def test_translation_pipeline_guards_after_the_outbound_translator(
    light_models, turn_debug_calls
) -> None:
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    config = _config(translation={"working_language": "en", "caller_languages": ["ru", "es"]})
    _, worker = build_livekit_pipeline(config, session)
    processors = _processors(worker)
    names = _names(processors)

    i = names.index("OutboundTranslator")
    assert names[i + 1] == "AnswerGuard" and names[i + 2] == "CartesiaTTSService"
    inbound = processors[names.index("InboundTranslator")]
    guard = processors[i + 1]
    assert inbound._on_empty_translation is not None
    assert processors[i]._on_empty_translation is not None
    # The guard follows the caller's current language.
    assert guard._language() == "ru"
    assert turn_debug_calls == [(worker, session, config.providers)]


def test_interpreter_pipeline_has_no_guard_no_turn_debug_but_reports_empty_translation(
    light_models, turn_debug_calls
) -> None:
    config = _config(mode="interpreter", interpreter={"languages": ["en", "ru"], "direction": "auto"})
    _, worker = build_livekit_pipeline(config, RuntimeSession(uuid4(), uuid4(), uuid4()))
    processors = _processors(worker)
    assert "AnswerGuard" not in _names(processors)
    assert turn_debug_calls == []
    translator = next(p for p in processors if isinstance(p, TranslationProcessor))
    assert translator._on_empty_translation is not None


# -- events ---------------------------------------------------------------------------


def test_new_event_types_are_declared() -> None:
    assert {"AnswerEmpty", "AnswerTruncated", "TranslationEmpty"} <= set(EVENT_TYPES)


def test_empty_answer_emits_answer_empty_with_the_turn_id() -> None:
    session = FakeSession()
    turn_state(session).begin()
    guard = pipeline_module.answer_guard(session, "ru")

    run(guard, response())

    assert session.of("AnswerEmpty") == [{"turn_id": 1, "language": "ru", "fallback": "repeat"}]


def test_empty_answer_without_a_language_reports_english_and_no_turn_id() -> None:
    session = FakeSession()
    run(pipeline_module.answer_guard(session, None), response())
    assert session.of("AnswerEmpty") == [{"language": "en", "fallback": "repeat"}]


def test_overlong_answer_emits_answer_truncated_without_text() -> None:
    session = FakeSession()
    turn_state(session).begin()
    guard = pipeline_module.answer_guard(session, "en")

    run(guard, response("a" * (MAX_SPOKEN_CHARS + 10), ". More words."))

    (event,) = session.of("AnswerTruncated")
    assert event["turn_id"] == 1 and event["max_spoken_chars"] == MAX_SPOKEN_CHARS
    assert event["spoken_chars"] >= MAX_SPOKEN_CHARS
    assert set(event) == {"turn_id", "spoken_chars", "max_spoken_chars"}


def test_outbound_empty_translation_emits_translation_empty_without_the_text() -> None:
    async def translate(messages):
        return ""

    session = FakeSession()
    state = CallerLanguageState(AgentTranslationSettings(working_language="en", caller_languages=("ru",)))
    state.language = "ru"
    outbound = OutboundTranslator(
        state=state, translate=translate, model="m", session=session,
        on_empty_translation=translation_empty_hook(session, "outbound"),
    )  # fmt: skip

    run(outbound, response("Secret sentence."))

    assert session.of("TranslationEmpty") == [
        {"stage": "outbound", "source_language": "en", "target_language": "ru"}
    ]


def test_interpreter_empty_translation_emits_event_and_turn_id_on_transcripts() -> None:
    async def translate(messages):
        return "" if messages[-1]["content"] == "blank" else "fine"

    def utterance(text):
        return [
            UserStartedSpeakingFrame(),
            SleepFrame(sleep=0.02),
            TranscriptionFrame(text, "speaker", "2026-01-01T00:00:00Z", "en", finalized=True),
            SleepFrame(sleep=0.02),
            UserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),
        ]

    session = FakeSession()
    turn_state(session).begin()
    processor = TranslationProcessor(
        settings=InterpreterSettings(languages=("en", "ru"), direction="auto"),
        translate=translate,
        model="test-model",
        session=session,
        on_empty_translation=translation_empty_hook(session, "interpreter"),
    )

    run(processor, utterance("blank") + utterance("good"))

    (empty,) = session.of("TranslationEmpty")
    assert empty == {"turn_id": 1, "stage": "interpreter", "source_language": "en", "target_language": "ru", "user_id": "speaker"}
    assert "blank" not in str(empty)
    transcripts = session.of("TranscriptFinal")
    assert transcripts and all(t["turn_id"] == 1 for t in transcripts)


def test_agent_translation_transcripts_carry_the_turn_id() -> None:
    async def translate(messages):
        return "pong"

    session = FakeSession()
    turn_state(session).begin()
    state = CallerLanguageState(AgentTranslationSettings(working_language="en", caller_languages=("ru",)))
    state.language = "ru"
    outbound = OutboundTranslator(state=state, translate=translate, model="m", session=session)

    run(outbound, response("Hello there."))

    (event,) = session.of("TranscriptFinal")
    assert event["role"] == "assistant" and event["turn_id"] == 1

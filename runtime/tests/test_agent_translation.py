"""Agent + translator (TR-040): inbound/outbound translation around an unchanged LLM."""

import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from src.runtime.agent_translation import (
    AgentTranslationSettings,
    CallerLanguageState,
    CallerLanguageTap,
    InboundTranslator,
    OutboundTranslator,
)
from src.runtime.api import SessionStart
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.session import RuntimeSession

SETTINGS = AgentTranslationSettings(working_language="en", caller_languages=("ru", "es"))


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


async def echo_translate(messages):
    """Fake translator: ``<target>:<text>``, the target read from the system prompt."""

    system = messages[0]["content"]
    target = system.split(" to ", 1)[1].split("(", 1)[1].split(")", 1)[0]
    return f"{target}:{messages[-1]['content']}"


def run(processor, frames):
    return asyncio.run(run_test(processor, frames_to_send=frames, expected_down_frames=None))


def segment(text, language=None) -> TranscriptionFrame:
    return TranscriptionFrame(text, "caller", "2026-01-01T00:00:00Z", language, finalized=True)


# -- settings ---------------------------------------------------------------------


def test_settings_validation_and_derived_values() -> None:
    auto = AgentTranslationSettings.from_mapping(
        {"working_language": "EN", "caller_languages": ["ru", "es"]}
    )
    assert auto.working_language == "en" and auto.caller_languages == ("ru", "es")
    assert auto.auto and auto.stt_language == "multi" and auto.initial_tts_language == "ru"
    # One caller language outside Deepgram's code-switching set: detection is fixed.
    fixed = AgentTranslationSettings.from_mapping(
        {"working_language": "en", "caller_languages": ["ko"]}
    )
    assert not fixed.auto and fixed.stt_language == "ko"
    assert "translated into English (en)" in auto.instruction
    assert "Always answer in English (en)" in auto.instruction
    for bad in (
        None,
        {"working_language": "en", "caller_languages": []},
        {"working_language": "en", "caller_languages": ["en"]},
        {"working_language": "en", "caller_languages": ["ru", "ru"]},
        {"working_language": "en", "caller_languages": ["ru", "es", "fr", "de", "it"]},
        {"working_language": "en", "caller_languages": ["ko", "ru"]},
        {"working_language": "xx", "caller_languages": ["ru"]},
    ):
        with pytest.raises(RuntimeError):
            AgentTranslationSettings.from_mapping(bad)


def test_language_detection_is_constrained_to_the_caller_languages() -> None:
    state = CallerLanguageState(SETTINGS)
    from src.runtime.agent_translation import _Segment

    def seg(text, language):
        return _Segment(text, language, "u", "t")

    assert state.detect([seg("hola", Language.ES), seg("x", "ru")], "hola x") == "es"
    # Working language is a candidate: the caller may simply speak English.
    assert state.detect([seg("hello", "en-US")], "hello") == "en"
    # The text beats the STT label: Deepgram "multi" calls a short Russian greeting "en".
    assert state.detect([seg("Здравствуйте.", "en")], "Здравствуйте.") == "ru"
    # ... and an implausible label for Latin text is ignored.
    assert state.detect([seg("Hello there", "ru")], "Hello there") == "en"
    assert state.detect([seg("привіт", "uk")], "привіт") == "ru"
    assert state.detect([], "...") == "ru"  # no letters: keep the previous language
    state.language = "es"
    assert state.detect([seg("ok", "pt")], "ok") == "es"
    assert state.detect([], "ok") == "es"
    state.language = "en"
    assert state.detect([], "ok") == "en"
    fixed = CallerLanguageState(
        AgentTranslationSettings(working_language="en", caller_languages=("ko",))
    )
    assert fixed.detect([seg("hello", "en")], "hello") == "ko"


# -- inbound ------------------------------------------------------------------------


def context_with_user(text: str) -> LLMContext:
    return LLMContext(messages=[{"role": "user", "content": text}])


def inbound(translate=echo_translate, session=None, state=None):
    state = state or CallerLanguageState(SETTINGS)
    tap = CallerLanguageTap(state)
    translator = InboundTranslator(
        state=state, translate=translate, model="test-model", session=session
    )
    return Pipeline([tap, translator]), state


def run_turn(pipeline, context, *segments, extra=()):
    frames = [*segments, SleepFrame(sleep=0.02), LLMContextFrame(context=context), *extra]
    return run(pipeline, [*frames, SleepFrame(sleep=0.05)])


def test_inbound_translates_the_user_turn_into_the_working_language_in_place() -> None:
    session = FakeSession()
    pipeline, state = inbound(session=session)
    context = context_with_user("Здравствуйте мне нужна помощь")
    down, _ = run_turn(
        pipeline,
        context,
        InterimTranscriptionFrame("Здрав", "caller", "t", Language.RU),
        segment("Здравствуйте", Language.RU),
        segment("мне нужна помощь", Language.RU),
    )

    # The LLM (and the stored history) only ever see the working language.
    assert context.messages == [
        {"role": "user", "content": "en:Здравствуйте мне нужна помощь"}
    ]
    assert state.language == "ru"
    # Transcripts and the context frame still reach the next processor.
    assert any(isinstance(f, LLMContextFrame) for f in down)
    assert sum(isinstance(f, TranscriptionFrame) for f in down) == 2
    user = [p for p in session.of("TranscriptFinal")]
    assert len(user) == 1
    assert user[0] == {
        "content": "Здравствуйте мне нужна помощь",
        "role": "user",
        "primary_language": "ru",
        "detected_languages": ["ru"],
        "translated_text": "en:Здравствуйте мне нужна помощь",
        "working_language": "en",
        "user_id": "caller",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    (latency,) = session.of("LatencyMeasured")
    assert latency["metric"] == "inbound_translation_ms"
    assert latency["processor"] == "agent_translation" and latency["model"] == "test-model"
    assert isinstance(latency["value_ms"], float) and latency["value_ms"] >= 0


def test_inbound_passes_working_language_speech_through_untranslated() -> None:
    calls = []

    async def translate(messages):
        calls.append(messages)
        return "never"

    session = FakeSession()
    pipeline, state = inbound(translate, session)
    context = context_with_user("I need help")
    run_turn(pipeline, context, segment("I need help", Language.EN_US))

    assert calls == [] and context.messages[0]["content"] == "I need help"
    assert state.language == "en"
    (event,) = session.of("TranscriptFinal")
    assert event["content"] == event["translated_text"] == "I need help"
    assert event["primary_language"] == "en" and event["working_language"] == "en"
    assert session.of("LatencyMeasured") == []


def test_inbound_translates_each_message_once_and_keeps_history_translations() -> None:
    calls = []

    async def translate(messages):
        calls.append(messages[-1]["content"])
        return await echo_translate(messages)

    pipeline, _ = inbound(translate)
    context = LLMContext(
        messages=[
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "спасибо"},
        ]
    )
    # Two frames for the same context (e.g. a re-run): the second must not re-translate.
    run_turn(
        pipeline,
        context,
        segment("спасибо", "ru"),
        extra=[LLMContextFrame(context=context)],
    )
    assert calls == ["спасибо"]
    # Only the trailing user message is pending; earlier history is left alone.
    assert [m["content"] for m in context.messages] == ["привет", "Hello!", "en:спасибо"]


def test_inbound_joins_a_turn_split_into_several_user_messages_into_one_call() -> None:
    # A turn cut off by a barge-in leaves an untranslated user message in the context.
    prompts = []

    async def translate(messages):
        prompts.append(messages)
        return await echo_translate(messages)

    session = FakeSession()
    pipeline, _ = inbound(translate, session)
    context = LLMContext(
        messages=[
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Здравствуйте."},
            {"role": "user", "content": "Я хотел бы записаться."},
        ]
    )
    run_turn(pipeline, context, segment("Здравствуйте.", "ru"), segment("Я хотел бы записаться.", "ru"))

    assert len(prompts) == 1  # one short call, no history in the prompt
    assert "Earlier utterances" not in prompts[0][0]["content"]
    assert context.messages == [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "en:Здравствуйте. Я хотел бы записаться."},
    ]
    (event,) = session.of("TranscriptFinal")
    assert event["content"] == "Здравствуйте. Я хотел бы записаться."
    assert len(session.of("LatencyMeasured")) == 1


def test_regression_stt_mislabelled_greeting_is_still_translated() -> None:
    """Live bug: one spoken sentence reached the context as two user messages and the
    first ("Здравствуйте.", labelled ``en`` by Deepgram multi) was never translated."""

    class AppendUserMessage(FrameProcessor):
        """Plays the aggregator: when the caller carries on, a second user message lands."""

        def __init__(self, context) -> None:
            super().__init__()
            self._context = context

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TextFrame) and frame.text == "CARRY-ON":
                self._context.add_message(
                    {"role": "user", "content": "Я хотел бы записаться на приём."}
                )
                return
            await self.push_frame(frame, direction)

    session = FakeSession()
    pipeline, _ = inbound(session=session)
    context = LLMContext(messages=[{"role": "user", "content": "Здравствуйте."}])
    pause = SleepFrame(sleep=0.1)
    down, _ = run(
        Pipeline([AppendUserMessage(context), pipeline]),
        [
            # The aggregator ended the turn at the pause after the greeting ...
            segment("Здравствуйте.", Language.EN),
            SleepFrame(sleep=0.02),
            LLMContextFrame(context=context),
            pause,
            # ... then the rest of the sentence arrived as a second user message.
            TextFrame("CARRY-ON"),
            pause,
            segment("Я хотел бы записаться на приём.", Language.RU),
            SleepFrame(sleep=0.02),
            LLMContextFrame(context=context),
            pause,
        ],
    )

    assert sum(isinstance(f, LLMContextFrame) for f in down) == 2
    assert [m["content"] for m in context.messages] == [
        "en:Здравствуйте.",
        "en:Я хотел бы записаться на приём.",
    ]
    greeting, rest = session.of("TranscriptFinal")
    assert (greeting["primary_language"], rest["primary_language"]) == ("ru", "ru")
    assert greeting["content"] == "Здравствуйте."
    assert greeting["translated_text"] == "en:Здравствуйте."


def test_inbound_failure_forwards_the_original_words() -> None:
    async def broken(messages):
        raise RuntimeError("provider down")

    session = FakeSession()
    pipeline, _ = inbound(broken, session)
    context = context_with_user("привет")
    down, _ = run_turn(pipeline, context, segment("привет", "ru"))

    assert any(isinstance(f, LLMContextFrame) for f in down)
    assert context.messages[0]["content"] == "привет"
    assert session.of("LatencyMeasured") == []
    assert session.of("TranscriptFinal")[0]["translated_text"] == "привет"


def test_inbound_ignores_non_user_tails_and_upstream_context_frames() -> None:
    calls = []

    async def translate(messages):
        calls.append(1)
        return "x"

    pipeline, _ = inbound(translate)
    context = LLMContext(
        messages=[
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "Hi"},
            {"role": "tool", "content": "{}", "tool_call_id": "1"},
        ]
    )
    run_turn(pipeline, context)
    assert calls == []


# -- outbound -------------------------------------------------------------------------


def outbound(translate=echo_translate, session=None, state=None, language="ru"):
    state = state or CallerLanguageState(SETTINGS)
    state.language = language
    return OutboundTranslator(
        state=state, translate=translate, model="test-model", session=session
    ), state


def response(*deltas: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *[LLMTextFrame(delta) for delta in deltas],
        LLMFullResponseEndFrame(),
    ]


def speak_list(down):
    return [f.text for f in down if isinstance(f, TTSSpeakFrame)]


def test_outbound_translates_sentence_by_sentence_and_hides_the_original_from_tts() -> None:
    session = FakeSession()
    processor, _ = outbound(session=session)
    down, _ = run(processor, response("Hello ", "there. How are", " you today? Fine"))

    assert speak_list(down) == [
        "ru:Hello there.",
        "ru:How are you today?",
        "ru:Fine",
    ]
    assert all(f.append_to_context is False for f in down if isinstance(f, TTSSpeakFrame))
    # Originals travel on for the assistant context but are never spoken.
    originals = [f for f in down if isinstance(f, LLMTextFrame)]
    assert [f.text for f in originals] == ["Hello there.", "How are you today?", "Fine"]
    assert all(f.skip_tts and not f.includes_inter_frame_spaces for f in originals)
    start = next(f for f in down if isinstance(f, LLMFullResponseStartFrame))
    end = next(f for f in down if isinstance(f, LLMFullResponseEndFrame))
    assert start.skip_tts and end.skip_tts
    # Each original is released right before its first spoken chunk.
    order = [
        ("say", f.text) if isinstance(f, TTSSpeakFrame) else ("ctx", f.text)
        for f in down
        if isinstance(f, (TTSSpeakFrame, LLMTextFrame))
    ]
    assert order[:2] == [("ctx", "Hello there."), ("say", "ru:Hello there.")]
    # Language is set from the first response (default already ru: no update).
    assert not any(isinstance(f, TTSUpdateSettingsFrame) for f in down)

    (event,) = session.of("TranscriptFinal")
    assert event == {
        "content": "ru:Hello there. ru:How are you today? ru:Fine",
        "role": "assistant",
        "primary_language": "ru",
        "detected_languages": ["ru"],
        "original_text": "Hello there. How are you today? Fine",
        "working_language": "en",
    }
    (latency,) = session.of("LatencyMeasured")
    assert latency["metric"] == "outbound_translation_first_chunk_ms"
    assert latency["processor"] == "agent_translation"


def test_outbound_streams_sub_chunks_and_translates_with_history() -> None:
    prompts = []

    async def streaming(messages):
        prompts.append(messages[0]["content"])
        for piece in ("Первая часть, ", "потом вторая часть, которая длиннее сорока знаков. ", "Конец."):
            yield piece

    processor, state = outbound(streaming)
    down, _ = run(processor, response("One sentence only."))

    # The streamed translation is re-chunked at sentence boundaries for early speech.
    assert speak_list(down) == [
        "Первая часть, потом вторая часть, которая длиннее сорока знаков.",
        "Конец.",
    ]
    assert len([f for f in down if isinstance(f, LLMTextFrame)]) == 1
    assert "English (en) to Russian (ru)" in prompts[0]
    assert list(state.history)[0][:3] == ("en", "One sentence only.", "ru")


def test_outbound_switches_tts_language_when_the_caller_language_changes() -> None:
    state = CallerLanguageState(SETTINGS)

    async def translate(messages):
        text = await echo_translate(messages)
        if messages[-1]["content"] == "First.":
            state.language = "es"  # the caller's next turn is detected as Spanish
        return text

    processor, _ = outbound(translate, state=state, language="ru")
    down, _ = run(processor, [*response("First."), *response("Second."), *response("Third.")])
    events = [
        ("lang", f.delta.language) if isinstance(f, TTSUpdateSettingsFrame) else ("say", f.text)
        for f in down
        if isinstance(f, (TTSUpdateSettingsFrame, TTSSpeakFrame))
    ]
    assert events == [("say", "ru:First."), ("lang", "es"), ("say", "es:Second."), ("say", "es:Third.")]


def test_outbound_passes_through_when_the_caller_speaks_the_working_language() -> None:
    session = FakeSession()
    calls = []

    async def translate(messages):
        calls.append(messages)
        return "never"

    state = CallerLanguageState(SETTINGS)
    processor, _ = outbound(translate, session, state=state, language="en")
    down, _ = run(processor, response("Hi ", "there."))

    assert calls == []
    texts = [f for f in down if isinstance(f, LLMTextFrame)]
    # Untouched LLM frames: TTS speaks them itself, in the working language.
    assert [(f.text, f.skip_tts) for f in texts] == [("Hi ", None), ("there.", None)]
    assert not any(isinstance(f, TTSSpeakFrame) for f in down)
    assert [f.delta.language for f in down if isinstance(f, TTSUpdateSettingsFrame)] == ["en"]
    (event,) = session.of("TranscriptFinal")
    assert (event["content"], event["original_text"], event["primary_language"]) == (
        "Hi there.",
        "Hi there.",
        "en",
    )
    assert session.of("LatencyMeasured") == []


def test_outbound_failure_speaks_the_sentence_untranslated() -> None:
    async def broken(messages):
        if "Bad" in messages[-1]["content"]:
            raise RuntimeError("down")
        return await echo_translate(messages)

    processor, _ = outbound(broken)
    down, _ = run(processor, response("Bad one. Good one."))
    assert speak_list(down) == ["Bad one.", "ru:Good one."]


def test_interruption_drops_pending_translation_and_reports_what_was_spoken() -> None:
    session = FakeSession()
    closed = []

    async def slow(messages):
        text = messages[-1]["content"]
        if text == "Second sentence.":
            try:
                await asyncio.Event().wait()  # never finishes: the provider is slow
            finally:
                closed.append(text)
        yield f"ru:{text}"

    processor, state = outbound(slow, session)
    frames = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("First sentence. Second sentence. Third sentence."),
        LLMFullResponseEndFrame(),
        SleepFrame(sleep=0.1),
        InterruptionFrame(),
        SleepFrame(sleep=0.05),
        *response("Fresh start."),
    ]
    down, _ = run(processor, frames)

    assert speak_list(down) == ["ru:First sentence.", "ru:Fresh start."]
    assert closed == ["Second sentence."]  # the in-flight stream was cancelled and closed
    # The interrupted response never reached its end frame; the new one is complete.
    assert sum(isinstance(f, LLMFullResponseEndFrame) for f in down) == 1
    assert [f.text for f in down if isinstance(f, LLMTextFrame)] == [
        "First sentence.",
        "Fresh start.",
    ]
    assert any(isinstance(f, InterruptionFrame) for f in down)
    assistant = [p for p in session.of("TranscriptFinal") if p["role"] == "assistant"]
    assert [(p["content"], p["original_text"]) for p in assistant] == [
        ("ru:First sentence.", "First sentence."),
        ("ru:Fresh start.", "Fresh start."),
    ]


# -- assistant context ---------------------------------------------------------------


class FakeTTS(FrameProcessor):
    """Models TTSService: skip_tts frames pass through, other text is "spoken"."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            await self.push_frame(
                TTSTextFrame(frame.text, "sentence", append_to_context=frame.append_to_context)
            )
        elif isinstance(frame, TextFrame) and not frame.skip_tts:
            spoken = TTSTextFrame(frame.text, "sentence")
            spoken.includes_inter_frame_spaces = frame.includes_inter_frame_spaces
            await self.push_frame(spoken)
        else:
            await self.push_frame(frame, direction)


def assistant_context_run(frames, language="ru"):
    context = LLMContext()
    _, assistant = LLMContextAggregatorPair(context)
    processor, _ = outbound(language=language)
    run(Pipeline([processor, FakeTTS(), assistant]), frames)
    return context


def test_assistant_context_keeps_the_working_language_reply() -> None:
    context = assistant_context_run(response("Your order ships ", "today. Anything else?"))
    assert context.messages == [
        {"role": "assistant", "content": "Your order ships today. Anything else?"}
    ]
    assert "ru:" not in str(context.messages)


def test_assistant_context_after_an_interruption_holds_only_started_sentences() -> None:
    async def slow(messages):
        text = messages[-1]["content"]
        if text == "Second sentence.":
            await asyncio.Event().wait()
        return f"ru:{text}"

    context = LLMContext()
    _, assistant = LLMContextAggregatorPair(context)
    processor, _ = outbound(slow)
    run(
        Pipeline([processor, FakeTTS(), assistant]),
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("First sentence. Second sentence. Third sentence."),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
            SleepFrame(sleep=0.05),
        ],
    )
    assert context.messages == [{"role": "assistant", "content": "First sentence."}]


def test_passthrough_reply_reaches_the_context_through_the_normal_path() -> None:
    context = assistant_context_run(response("Hi ", "there."), language="en")
    assert context.messages == [{"role": "assistant", "content": "Hi there."}]


# -- contract, config and builder -----------------------------------------------------


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "LIVEKIT_URL": "ws://localhost:7880",
        "DEEPGRAM_API_KEY": "deepgram",
        "OPENAI_API_KEY": "openai",
        "CARTESIA_API_KEY": "cartesia",
        "CARTESIA_VOICE_ID": "voice",
    }.items():
        monkeypatch.setenv(key, value)


BASE_SESSION = {
    "session_id": str(uuid4()),
    "conversation_id": str(uuid4()),
    "agent_id": str(uuid4()),
    "room_name": "room",
}


def test_session_start_omits_translation_unless_set() -> None:
    assert "translation" not in SessionStart(**BASE_SESSION).model_dump()
    translation = {"working_language": "en", "caller_languages": ["ru"]}
    started = SessionStart(**BASE_SESSION, translation=translation)
    assert started.model_dump(mode="json")["translation"] == translation


def test_config_carries_and_validates_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    base = {"room_name": "room", "livekit_token": "t"}
    translation = {"working_language": "en", "caller_languages": ["ru", "es"]}
    config = PipelineConfig.from_environment({**base, "translation": translation})
    assert config.translation == translation and config.mode == "agent"
    assert PipelineConfig.from_environment(base).translation is None
    with pytest.raises(RuntimeError, match="only supported for agent"):
        PipelineConfig.from_environment(
            {
                **base,
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
                "translation": translation,
            }
        )
    with pytest.raises(RuntimeError, match="caller_languages"):
        PipelineConfig.from_environment(
            {**base, "translation": {"working_language": "en", "caller_languages": ["en"]}}
        )


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
        "You are a clinic assistant.", **kwargs,
    )  # fmt: skip


def _processors(worker):
    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    return [
        p for p in pipeline._processors if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]


def test_builder_selects_the_translator_pipeline_only_when_translation_is_set(light_models) -> None:
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    config = _config(
        translation={"working_language": "en", "caller_languages": ["ru", "es"]},
        provider_settings={"stt": {"model": "nova-3", "language": "en"}, "tts": {"language": "en"}},
    )
    _, worker = build_livekit_pipeline(config, session)
    processors = _processors(worker)
    names = [type(p).__name__ for p in processors]

    assert names == [
        "LiveKitInputTransport",
        "DeepgramSTTService",
        "CallerLanguageTap",
        "LLMUserAggregator",
        "InboundTranslator",
        "OpenAILLMService",
        "OutboundTranslator",
        "CartesiaTTSService",
        "RuntimeEventProcessor",
        "LiveKitOutputTransport",
        "LLMAssistantAggregator",
    ]
    stt, llm, tts = processors[1], processors[5], processors[7]
    assert stt._settings.language == "multi" and stt._settings.model == "nova-3"
    assert tts._settings.language == "ru"
    instruction = llm._settings.system_instruction
    assert instruction.startswith("You are a clinic assistant.")
    assert "their words reach you translated into English (en)" in instruction
    # The translators own the transcripts: the remaining event processor mutes them,
    # so no transcript event can be emitted twice.
    assert names.count("RuntimeEventProcessor") == 1
    assert processors[8]._transcript_events is False
    assert processors[4]._session is session and processors[6]._session is session
    # The translation legs use a small fast model, not the LLM's; warm-up is wired.
    assert processors[4]._model == processors[6]._model == "gpt-4.1-nano"
    assert processors[4]._warm_up is not None


def test_translation_model_is_configurable_and_never_reaches_the_llm_settings(light_models) -> None:
    config = _config(
        translation={"working_language": "en", "caller_languages": ["ru"]},
        provider_settings={"llm": {"model": "gpt-big", "translation_model": "gpt-4.1-mini"}},
    )
    _, worker = build_livekit_pipeline(config)
    processors = _processors(worker)
    assert processors[4]._model == processors[6]._model == "gpt-4.1-mini"
    assert processors[5]._settings.model == "gpt-big"


def test_builder_fixes_the_stt_language_for_a_single_non_detectable_caller(light_models) -> None:
    config = _config(translation={"working_language": "en", "caller_languages": ["ko"]})
    _, worker = build_livekit_pipeline(config)
    processors = _processors(worker)
    assert processors[1]._settings.language == "ko"
    assert processors[7]._settings.language == "ko"
    assert "RuntimeEventProcessor" not in [type(p).__name__ for p in processors]


def test_builder_leaves_the_normal_agent_pipeline_untouched(light_models) -> None:
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    _, worker = build_livekit_pipeline(_config(), session)
    processors = _processors(worker)
    assert [type(p).__name__ for p in processors] == [
        "LiveKitInputTransport",
        "DeepgramSTTService",
        "RuntimeEventProcessor",
        "LLMUserAggregator",
        "OpenAILLMService",
        "RuntimeEventProcessor",
        "CartesiaTTSService",
        "RuntimeEventProcessor",
        "LiveKitOutputTransport",
        "LLMAssistantAggregator",
    ]
    assert all(p._transcript_events for p in processors if type(p).__name__ == "RuntimeEventProcessor")
    assert "translated into" not in processors[4]._settings.system_instruction


# -- RM-003: utterance timeline ---------------------------------------------------------


def _timeline_pipeline(session, *, language="ru"):
    from src.runtime.timeline import TimelineTracker

    tracker = TimelineTracker(session, "agent_translation")
    state = CallerLanguageState(SETTINGS, timeline=tracker)
    state.language = language
    tap = CallerLanguageTap(state)
    inbound_t = InboundTranslator(state=state, translate=echo_translate, model="m", session=session)
    outbound_t = OutboundTranslator(state=state, translate=echo_translate, model="m", session=session)
    return Pipeline([tap, inbound_t, outbound_t]), tracker


def _voiced(ms: int, rate: int = 16000) -> bytes:
    return b"\x01\x10" * (rate * ms // 1000)


def test_agent_translation_timeline_runs_from_caller_speech_to_first_translated_reply() -> None:
    from pipecat.frames.frames import VADUserStartedSpeakingFrame

    session = FakeSession()
    pipeline, tracker = _timeline_pipeline(session)
    context = context_with_user("Здравствуйте")
    run(
        pipeline,
        [
            VADUserStartedSpeakingFrame(start_secs=0.2),
            SleepFrame(sleep=0.03),
            InterimTranscriptionFrame("Здрав", "caller", "t", Language.RU),
            SleepFrame(sleep=0.03),
            segment("Здравствуйте", Language.RU),
            SleepFrame(sleep=0.03),
            LLMContextFrame(context=context),
            SleepFrame(sleep=0.05),
            *response("Hello there. ", "How can I help?"),
            SleepFrame(sleep=0.1),
        ],
    )
    [queued] = list(tracker._queue)
    assert list(queued.marks) == ["vad", "first_transcript", "translation_start"]
    assert queued.chunks_pushed == 2 and queued.speech_complete  # one TTS context per sentence

    tracker.on_audio(_voiced(300), 16000)
    tracker.on_tts_stopped()
    assert session.of("LatencyMeasured")[-1]["metric"] != "utterance_timeline"  # still playing
    tracker.on_audio(_voiced(300), 16000)
    tracker.on_tts_stopped()
    [event] = [p for p in session.of("LatencyMeasured") if p["metric"] == "utterance_timeline"]
    assert event["engine"] == "agent_translation"
    marks = event["marks_ms"]
    assert 200 <= marks["vad"] <= 200.5
    assert marks["vad"] <= marks["first_transcript"] <= marks["translation_start"] <= marks["first_audio"]
    assert event["value_ms"] == marks["first_audio"]


def test_agent_timeline_when_the_caller_speaks_the_working_language_has_no_translation_mark() -> None:
    from pipecat.frames.frames import VADUserStartedSpeakingFrame

    session = FakeSession()
    pipeline, tracker = _timeline_pipeline(session, language="en")
    run(
        pipeline,
        [
            VADUserStartedSpeakingFrame(start_secs=0.1),
            SleepFrame(sleep=0.03),
            segment("Hello", Language.EN),
            SleepFrame(sleep=0.03),
            LLMContextFrame(context=context_with_user("Hello")),
            SleepFrame(sleep=0.05),
            *response("Hi there."),
            SleepFrame(sleep=0.1),
        ],
    )
    [queued] = list(tracker._queue)
    assert list(queued.marks) == ["vad", "first_transcript"]
    assert queued.chunks_pushed is None  # TTS speaks the LLM text: the count is unknown
    tracker.on_audio(_voiced(200), 16000)
    tracker.on_idle()
    [event] = [p for p in session.of("LatencyMeasured") if p["metric"] == "utterance_timeline"]
    assert "translation_start" not in event["marks_ms"] and event["engine"] == "agent_translation"


def test_only_the_first_response_of_a_turn_is_timed() -> None:
    from pipecat.frames.frames import VADUserStartedSpeakingFrame

    session = FakeSession()
    pipeline, tracker = _timeline_pipeline(session)
    run(
        pipeline,
        [
            VADUserStartedSpeakingFrame(start_secs=0.1),
            SleepFrame(sleep=0.03),
            segment("Здравствуйте", Language.RU),
            SleepFrame(sleep=0.03),
            LLMContextFrame(context=context_with_user("Здравствуйте")),
            SleepFrame(sleep=0.05),
            *response("One."),
            SleepFrame(sleep=0.05),
            *response("Two."),  # e.g. the reply after a tool call
            SleepFrame(sleep=0.1),
        ],
    )
    assert len(tracker._queue) == 1


def test_a_turn_continued_after_a_pause_keeps_the_original_speech_start() -> None:
    from pipecat.frames.frames import VADUserStartedSpeakingFrame

    session = FakeSession()
    pipeline, tracker = _timeline_pipeline(session)
    context = context_with_user("Здравствуйте")
    run(
        pipeline,
        [
            VADUserStartedSpeakingFrame(start_secs=0.2),
            SleepFrame(sleep=0.03),
            segment("Здравствуйте", Language.RU),
            SleepFrame(sleep=0.03),
            LLMContextFrame(context=context),  # first part of the turn reaches the LLM
            SleepFrame(sleep=0.03),
            segment("мне нужна помощь", Language.RU),  # the caller goes on, no new VAD start
            SleepFrame(sleep=0.03),
            LLMContextFrame(context=context_with_user("мне нужна помощь")),
            SleepFrame(sleep=0.05),
            *response("Sure."),
            SleepFrame(sleep=0.1),
        ],
    )
    [queued] = list(tracker._queue)
    assert "vad" in queued.marks and queued.chunks_pushed == 1


def test_a_reply_cut_off_before_any_audio_hands_its_timing_to_the_next_reply() -> None:
    from pipecat.frames.frames import VADUserStartedSpeakingFrame

    session = FakeSession()
    pipeline, tracker = _timeline_pipeline(session)
    run(
        pipeline,
        [
            VADUserStartedSpeakingFrame(start_secs=0.2),
            SleepFrame(sleep=0.03),
            segment("Здравствуйте", Language.RU),
            SleepFrame(sleep=0.03),
            LLMContextFrame(context=context_with_user("Здравствуйте")),
            SleepFrame(sleep=0.05),
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hel"),
            SleepFrame(sleep=0.05),
            InterruptionFrame(),  # the caller talks over the reply before it is heard
            SleepFrame(sleep=0.05),
            *response("Hello there."),
            SleepFrame(sleep=0.1),
        ],
    )
    [queued] = list(tracker._queue)
    assert "vad" in queued.marks and queued.chunks_pushed == 1 and queued.speech_complete

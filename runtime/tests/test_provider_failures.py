"""PR-30 / ST-10a: provider errors emit ProviderError (never log-only) and end the call cleanly."""

import asyncio
import json
from uuid import uuid4

import httpx
import openai
import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

import src.runtime.agent_translation
import src.runtime.pipeline
from src.runtime.agent_translation import (
    AgentTranslationSettings,
    CallerLanguageState,
    InboundTranslator,
    OutboundTranslator,
)
from src.runtime.interpreter import InterpreterSettings, TranslationProcessor
from src.runtime.pipeline import PipelineConfig
from src.runtime.provider_failures import (
    cascaded_fallback_config,
    classify_error,
    provider_kind,
    report_provider_failure,
    watch_provider_failures,
)
from src.runtime.realtime_translate import RealtimeTranslateProcessor
from src.runtime.session import RuntimeSession, SessionState

SECRET = "sk-live-SECRET-key-123"
PAYLOAD_KEYS = {"provider", "kind", "category", "stage", "retryable", "attempt", "message"}
TIMEOUT = 10


# -- helpers --------------------------------------------------------------------


def openai_error(cls, status: int, body: dict | None = None):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls(f"Incorrect API key provided: {SECRET}", response=response, body=body)


def auth_error():
    return openai_error(openai.AuthenticationError, 401)


def quota_error():
    """Cartesia / generic billing wall: HTTP 402."""

    request = httpx.Request("POST", "https://api.cartesia.ai/tts")
    response = httpx.Response(402, request=request, text="quota_exceeded")
    return httpx.HTTPStatusError(f"402 quota_exceeded for key {SECRET}", request=request, response=response)


class RecordingClient:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.event_type for e in self.events]

    def of(self, event_type: str) -> list[dict]:
        return [e.payload for e in self.events if e.event_type == event_type]


class FakeTransport:
    def __init__(self) -> None:
        self.handlers = {}
        self.cleaned = False

    def add_event_handler(self, name, handler) -> None:
        self.handlers[name] = handler

    async def cleanup(self) -> None:
        self.cleaned = True


class Stub(FrameProcessor):
    """A provider stand-in: speaks every TTSSpeakFrame until ``errors`` say otherwise.

    ``errors`` maps the n-th frame (1-based) to ``(message, exception, unusable)``; that
    frame pushes an upstream ErrorFrame exactly like a Pipecat service does, and nothing
    is spoken for it (nor for any later frame once the stub is unusable).
    """

    def __init__(self, name: str, errors: dict | None = None) -> None:
        super().__init__(name=name)
        self.errors = errors or {}
        self.seen = 0
        self.spoken: list[str] = []

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame):
            self.seen += 1
            if not self.is_usable:
                return  # "service is no longer usable, not speaking"
            error = self.errors.get(self.seen)
            if error is not None:
                message, exception, unusable = error
                await self.push_error(message, exception=exception, force_treat_as_permanent=unusable)
                return
            self.spoken.append(frame.text)
        await self.push_frame(frame, direction)


class FakeCartesiaTTSService(Stub):
    pass


class FakeOpenAILLMService(Stub):
    pass


class FakeDeepgramSTTService(Stub):
    pass


def pipeline_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        "ws://livekit", "token", "room-42", "deepgram", "openai", "cartesia", "voice", "prompt", **kwargs
    )


async def run_session(monkeypatch, stub: Stub, frames, *, config=None, linger: float = 0.3):
    """Run a real PipelineWorker through RuntimeSession.run_pipeline; return (client, session).

    ``frames`` are queued one by one after the pipeline started (``linger`` apart). The call
    must finish on its own: a hang fails the test through the timeout.
    """

    import pipecat.workers.runner  # noqa: F401

    client = RecordingClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    worker = PipelineWorker(Pipeline([stub]), params=PipelineParams(enable_metrics=False))
    watch_provider_failures(worker, session, {"stt": "deepgram", "llm": "openai", "tts": "cartesia"})
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(*_):
        started.set()

    transport = FakeTransport()
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, worker))
    run = asyncio.create_task(session.run_pipeline(config or pipeline_config()))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    for frame in frames:
        if worker.has_finished():
            break
        await worker.queue_frame(frame)
        await asyncio.sleep(linger)
    return run, client, session, transport, worker


def speak(text: str) -> TTSSpeakFrame:
    return TTSSpeakFrame(text, append_to_context=False)


def assert_clean_failure(client: RecordingClient, session: RuntimeSession, transport, **expected) -> None:
    failed = client.of("ProviderError")[-1]  # the terminal one
    expected = {("category" if k == "reason" else k): v for k, v in expected.items()}
    assert {k: failed[k] for k in expected} == expected
    assert set(failed) == PAYLOAD_KEYS
    assert failed["retryable"] is False
    assert SECRET not in json.dumps(failed)
    assert client.types()[-1] == "SessionFailed"
    assert client.of("SessionFailed") == [{"outcome": "failed", "reason": "provider_failed"}]
    assert session.state == SessionState.ENDED
    assert transport.cleaned  # the room connection is released
    assert session._pipeline_worker is None  # the worker is freed


# -- classification ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"exception": auth_error()}, "auth"),
        ({"exception": quota_error()}, "quota"),
        ({"exception": openai_error(openai.RateLimitError, 429, {"code": "insufficient_quota"})}, "quota"),
        ({"exception": openai_error(openai.APIStatusError, 503)}, "unavailable"),
        ({"exception": TimeoutError("read timed out")}, "timeout"),
        ({"exception": httpx.ReadTimeout("x")}, "timeout"),
        ({"exception": openai.APITimeoutError(request=httpx.Request("POST", "https://x"))}, "timeout"),
        ({"exception": openai_error(openai.APIStatusError, 408)}, "timeout"),
        ({"text": "request timed out"}, "timeout"),
        ({"exception": ConnectionClosedError(Close(3000, "invalid_api_key"), None)}, "auth"),
        ({"exception": ConnectionClosedError(Close(1006, ""), None)}, "unavailable"),
        ({"exception": ValueError("boom")}, "other"),
        ({"text": "CartesiaTTSService error: quota_exceeded"}, "quota"),
        ({"text": "received 402 Payment Required"}, "quota"),
        ({"text": "HTTP 401 unauthorized"}, "auth"),
        ({"text": "invalid_api_key: bad key"}, "auth"),
        ({"text": "something odd"}, "other"),
        # a tenant's inaccessible model is a configuration error, not a rejected key
        (
            {"exception": openai_error(openai.PermissionDeniedError, 403, {"code": "model_not_found"})},
            "other",
        ),
        ({"text": "Error code: 403 - Project does not have access to model gpt-5"}, "other"),
        ({"text": "The model `gpt-9` does not exist or you do not have access to it (404)"}, "other"),
        ({"text": "403 model_not_found"}, "other"),
        ({"text": "403 Forbidden"}, "auth"),
        ({"exception": openai_error(openai.PermissionDeniedError, 403)}, "auth"),
        # a bare "quota" word is not a quota failure
        ({"text": "per-minute quota header missing"}, "other"),
        ({"text": "exceeded your current quota, please check your plan"}, "quota"),
        ({"text": "insufficient_quota"}, "quota"),
        ({"category": ErrorCategory.AUTHENTICATION}, "auth"),
        ({"category": ErrorCategory.QUOTA}, "quota"),
        ({"category": ErrorCategory.CONNECTIVITY}, "unavailable"),
        ({"category": ErrorCategory.UNKNOWN, "text": "nothing"}, "other"),
    ],
)
def test_classify_error(kwargs, expected) -> None:
    assert classify_error(**kwargs) == expected


def test_provider_kind_from_class_name() -> None:
    assert provider_kind(FakeCartesiaTTSService("x")) == "tts"
    assert provider_kind(FakeOpenAILLMService("x")) == "llm"
    assert provider_kind(FakeDeepgramSTTService("x")) == "stt"
    assert provider_kind(object()) is None
    assert provider_kind(type("LLMUserAggregator", (), {})()) is None  # not a provider


# -- mid-call errors through a real Pipecat worker ----------------------------------


@pytest.mark.parametrize(
    "stub_cls, kind, provider, exception, reason",
    [
        (FakeCartesiaTTSService, "tts", "cartesia", quota_error(), "quota"),  # 402 mid-call
        (FakeCartesiaTTSService, "tts", "cartesia", auth_error(), "auth"),  # 401 mid-call
        (FakeOpenAILLMService, "llm", "openai", auth_error(), "auth"),
        (FakeOpenAILLMService, "llm", "openai", quota_error(), "quota"),
        (FakeDeepgramSTTService, "stt", "deepgram", auth_error(), "auth"),
    ],
)
def test_fatal_provider_error_mid_call_ends_the_session_cleanly(
    monkeypatch, stub_cls, kind, provider, exception, reason
) -> None:
    async def scenario():
        stub = stub_cls("provider", {2: ("provider error", exception, False)})
        run, client, session, transport, _ = await run_session(
            monkeypatch, stub, [speak("one"), speak("two"), speak("three")]
        )
        await asyncio.wait_for(run, TIMEOUT)  # no hang: the pipeline ended by itself
        return stub, client, session, transport

    stub, client, session, transport = asyncio.run(scenario())
    assert stub.spoken == ["one"]  # the call did not go on silently afterwards
    assert_clean_failure(client, session, transport, kind=kind, provider=provider, reason=reason)


def test_service_marked_unusable_without_status_code_still_ends_the_call(monkeypatch) -> None:
    """Cartesia's quota_exceeded: text only, then the service is "no longer usable"."""

    async def scenario():
        stub = FakeCartesiaTTSService(
            "tts", {1: ("CartesiaTTSService error: quota_exceeded", None, True)}
        )
        run, client, session, transport, _ = await run_session(monkeypatch, stub, [speak("a"), speak("b")])
        await asyncio.wait_for(run, TIMEOUT)
        return stub, client, session, transport

    stub, client, session, transport = asyncio.run(scenario())
    assert stub.spoken == []
    assert_clean_failure(client, session, transport, kind="tts", provider="cartesia", reason="quota")


def test_unusable_service_with_unrecognised_error_is_reason_other(monkeypatch) -> None:
    async def scenario():
        stub = FakeCartesiaTTSService("tts", {1: ("weird", ValueError("x"), True)})
        run, client, session, transport, _ = await run_session(monkeypatch, stub, [speak("a")])
        await asyncio.wait_for(run, TIMEOUT)
        return client, session, transport

    assert_clean_failure(*asyncio.run(scenario()), kind="tts", reason="other")


def test_one_transient_error_is_retried_and_the_second_ends_the_call(monkeypatch) -> None:
    timeout = httpx.ReadTimeout("read timed out")

    async def scenario():
        stub = FakeCartesiaTTSService(
            "tts", {2: ("timeout", timeout, False), 3: ("timeout", timeout, False)}
        )
        run, client, session, transport, worker = await run_session(
            monkeypatch, stub, [speak("1"), speak("2")], linger=0.2
        )
        assert not run.done()  # one blip: the call goes on ...
        await asyncio.sleep(0.1)
        [first] = client.of("ProviderError")  # ... but it is announced, never log-only
        assert (first["kind"], first["category"], first["attempt"], first["retryable"]) == (
            "tts",
            "timeout",
            1,
            True,
        )
        assert client.of("SessionFailed") == []
        await worker.queue_frame(speak("3"))
        await asyncio.wait_for(run, TIMEOUT)
        return stub, client, session, transport

    stub, client, session, transport = asyncio.run(scenario())
    assert stub.spoken == ["1"]
    assert [p["attempt"] for p in client.of("ProviderError")] == [1, 2]
    assert_clean_failure(client, session, transport, kind="tts", reason="timeout", attempt=2)


def test_unavailable_503_is_retried_once_then_fails_like_a_timeout() -> None:
    async def scenario():
        client = RecordingClient()
        session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
        now = [0.0]
        clock = lambda: now[0]  # noqa: E731
        first = report_provider_failure(session, "llm", "openai", "unavailable", clock=clock)
        now[0] = 31.0  # outside the window: this is a new first error
        second = report_provider_failure(session, "llm", "openai", "unavailable", clock=clock)
        now[0] = 40.0
        third = report_provider_failure(session, "llm", "openai", "unavailable", clock=clock)
        await asyncio.sleep(0)
        await session.close_event_dispatcher()
        return (first, second, third), client

    outcomes, client = asyncio.run(scenario())
    assert outcomes == ("retry", "retry", "failed")
    events = client.of("ProviderError")
    assert [(e["attempt"], e["retryable"]) for e in events] == [(1, True), (1, True), (2, False)]
    assert all(e["category"] == "unavailable" for e in events)


def test_error_after_the_session_started_ending_is_ignored(monkeypatch) -> None:
    client = RecordingClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    session._pipeline_cancel_requested = True
    assert report_provider_failure(session, "tts", "cartesia", "auth") == "ignored"
    assert client.events == []


def test_a_failure_is_reported_once() -> None:
    async def scenario():
        client = RecordingClient()
        session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
        outcomes = [
            report_provider_failure(session, "tts", "cartesia", "quota"),
            report_provider_failure(session, "llm", "openai", "auth"),
        ]
        await asyncio.sleep(0)
        await session.close_event_dispatcher()
        return outcomes, client

    outcomes, client = asyncio.run(scenario())
    assert outcomes == ["failed", "duplicate"]
    assert [e.payload["kind"] for e in client.events] == ["tts"]


# -- translators (LLM errors are caught inside the processors) ------------------------


class FakeSession:
    def __init__(self) -> None:
        self.event_client = object()
        self.session_id = uuid4()
        self.events: list[tuple[str, dict]] = []
        self.failures: list[str] = []
        self.end_requests: list[str] = []

    def event(self, event_type, payload=None):
        return (event_type, payload or {})

    def enqueue_event(self, event) -> None:
        self.events.append(event)

    def fail(self, reason) -> None:
        self.failures.append(reason)

    async def request_end(self, reason="") -> bool:
        self.end_requests.append(reason)
        return True

    def of(self, event_type: str) -> list[dict]:
        return [payload for kind, payload in self.events if kind == event_type]


def failing_translate(exception):
    async def translate(messages):
        raise exception
        yield  # pragma: no cover - makes this an async generator

    return translate


def say(text: str):
    from pipecat.frames.frames import (
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )

    return [
        UserStartedSpeakingFrame(),
        SleepFrame(sleep=0.02),
        TranscriptionFrame(text, "speaker", "2026-01-01T00:00:00Z", None, finalized=True),
        SleepFrame(sleep=0.02),
        UserStoppedSpeakingFrame(),
        SleepFrame(sleep=0.1),
    ]


@pytest.mark.parametrize(
    "exception, reason", [(auth_error(), "auth"), (quota_error(), "quota")]
)
def test_interpreter_translation_401_or_402_ends_the_session(exception, reason) -> None:
    async def scenario():
        session = FakeSession()
        processor = TranslationProcessor(
            settings=InterpreterSettings(languages=("en", "ru"), direction="one_way"),
            translate=failing_translate(exception),
            model="m",
            session=session,
            initial_tts_language="ru",
        )
        await run_test(processor, frames_to_send=say("hello there"), expected_down_frames=None)
        await asyncio.sleep(0.05)
        return session

    session = asyncio.run(scenario())
    [failed] = session.of("ProviderError")
    assert (failed["kind"], failed["provider"], failed["category"]) == ("translation", "openai", reason)
    assert SECRET not in json.dumps(failed)
    assert session.failures == ["provider_failed"]
    assert session.end_requests == ["provider_failed"]


def test_interpreter_tolerates_one_transient_translation_error() -> None:
    async def scenario():
        session = FakeSession()
        processor = TranslationProcessor(
            settings=InterpreterSettings(languages=("en", "ru"), direction="one_way"),
            translate=failing_translate(openai_error(openai.APIStatusError, 503)),
            model="m",
            session=session,
            initial_tts_language="ru",
        )
        await run_test(processor, frames_to_send=say("hello"), expected_down_frames=None)
        await asyncio.sleep(0.05)
        return session

    session = asyncio.run(scenario())
    [event] = session.of("ProviderError")
    assert (event["kind"], event["category"], event["attempt"], event["retryable"]) == (
        "translation",
        "unavailable",
        1,
        True,
    )
    assert session.end_requests == []


def non_provider_error_session(utterances: int) -> FakeSession:
    async def scenario():
        session = FakeSession()
        processor = TranslationProcessor(
            settings=InterpreterSettings(languages=("en", "ru"), direction="one_way"),
            translate=failing_translate(ValueError("bug")),
            model="m",
            session=session,
            initial_tts_language="ru",
        )
        frames = [frame for _ in range(utterances) for frame in say("one")]
        await run_test(processor, frames_to_send=frames, expected_down_frames=None)
        await asyncio.sleep(0.05)
        return session

    return asyncio.run(scenario())


def test_unclassified_translator_error_is_emitted_but_does_not_end_the_call() -> None:
    session = non_provider_error_session(2)
    events = session.of("ProviderError")
    assert [(e["category"], e["attempt"], e["retryable"]) for e in events] == [
        ("other", 1, False),
        ("other", 2, False),
    ]
    assert SECRET not in json.dumps(events) and "bug" not in json.dumps(events)
    assert session.end_requests == []


def test_repeated_unclassified_translator_errors_end_the_call() -> None:
    session = non_provider_error_session(3)
    assert [e["attempt"] for e in session.of("ProviderError")] == [1, 2, 3]
    assert session.failures == ["provider_failed"] and session.end_requests == ["provider_failed"]


def test_agent_translator_401_ends_the_session_on_both_legs() -> None:
    from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame

    settings = AgentTranslationSettings(working_language="en", caller_languages=("ru", "es"))

    async def inbound():
        session = FakeSession()
        state = CallerLanguageState(settings)
        processor = InboundTranslator(
            state=state, translate=failing_translate(auth_error()), model="m", session=session
        )
        text = await processor._translate_message("привет", "ru")
        await asyncio.sleep(0.05)
        return text, session

    text, session = asyncio.run(inbound())
    assert text == "привет"  # the caller's words still pass through
    assert session.of("ProviderError")[0]["category"] == "auth"
    assert session.end_requests == ["provider_failed"]

    async def outbound():
        session = FakeSession()
        state = CallerLanguageState(settings)
        state.language = "ru"
        processor = OutboundTranslator(
            state=state, translate=failing_translate(quota_error()), model="m", session=session
        )
        frames = [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Hello there. How are you?"),
            LLMFullResponseEndFrame(),
            SleepFrame(sleep=0.3),
        ]
        await run_test(processor, frames_to_send=frames, expected_down_frames=None)
        await asyncio.sleep(0.05)
        return session

    session = asyncio.run(outbound())
    assert session.of("ProviderError")[0]["category"] == "quota"
    assert session.end_requests == ["provider_failed"]


def flaky_translate(first_exception, text="hello"):
    """Fails once with ``first_exception``, then translates."""

    calls = []

    async def translate(messages):
        calls.append(1)
        if len(calls) == 1:
            raise first_exception
        yield text

    translate.calls = calls
    return translate


def inbound_translator(translate, session):
    settings = AgentTranslationSettings(working_language="en", caller_languages=("ru", "es"))
    return InboundTranslator(
        state=CallerLanguageState(settings), translate=translate, model="m", session=session
    )


def test_inbound_translator_retries_a_transient_failure_once_and_recovers() -> None:
    async def scenario():
        session = FakeSession()
        translate = flaky_translate(openai_error(openai.APIStatusError, 503), "hello")
        text = await inbound_translator(translate, session)._translate_message("привет", "ru")
        await asyncio.sleep(0.05)
        return text, session, translate

    text, session, translate = asyncio.run(scenario())
    assert text == "hello" and len(translate.calls) == 2  # a real request-level retry
    [event] = session.of("ProviderError")
    assert (event["kind"], event["category"], event["attempt"], event["retryable"]) == (
        "translation",
        "unavailable",
        1,
        True,
    )
    assert event["stage"] == "inbound_translation"
    assert session.failures == [] and session.end_requests == []  # no SessionFailed


def test_inbound_translator_double_failure_fails_the_call_but_passes_text_through() -> None:
    async def scenario():
        session = FakeSession()
        translate = failing_translate(httpx.ReadTimeout("slow"))
        text = await inbound_translator(translate, session)._translate_message("привет", "ru")
        await asyncio.sleep(0.05)
        return text, session

    text, session = asyncio.run(scenario())
    assert text == "привет"
    assert [(e["category"], e["attempt"], e["retryable"]) for e in session.of("ProviderError")] == [
        ("timeout", 1, True),
        ("timeout", 2, False),
    ]
    assert session.failures == ["provider_failed"] and session.end_requests == ["provider_failed"]


def test_inbound_translator_unclassified_failure_emits_and_passes_text_through() -> None:
    async def scenario():
        session = FakeSession()
        translate = failing_translate(ValueError("secret " + SECRET))
        text = await inbound_translator(translate, session)._translate_message("привет", "ru")
        return text, session

    text, session = asyncio.run(scenario())
    assert text == "привет"
    [event] = session.of("ProviderError")
    assert (event["kind"], event["category"], event["retryable"]) == ("translation", "other", False)
    assert SECRET not in json.dumps(event)
    assert session.end_requests == []


def test_hung_translator_times_out_through_the_first_token_guard(monkeypatch) -> None:
    import src.runtime.stall_watchdog as watchdog

    async def hang(messages):
        await asyncio.sleep(3600)
        yield "never"

    async def scenario():
        session = FakeSession()
        monkeypatch.setattr(
            src.runtime.agent_translation,
            "guard_first_token",
            lambda stream: watchdog.guard_first_token(stream, 0.05),
        )
        monkeypatch.setattr(src.runtime.agent_translation, "INBOUND_RETRY_DELAY_SECONDS", 0)
        text = await inbound_translator(hang, session)._translate_message("привет", "ru")
        return text, session

    text, session = asyncio.run(scenario())
    assert text == "привет"
    assert [e["category"] for e in session.of("ProviderError")] == ["timeout", "timeout"]
    assert session.end_requests == ["provider_failed"]


# -- realtime engine ---------------------------------------------------------------------


class FakeWS:
    def __init__(self, script=()) -> None:
        self.sent = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        for item in script:
            self._inbox.put_nowait(item)

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def close(self) -> None:
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._inbox.get()
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)


class Connector:
    def __init__(self, *items) -> None:
        self.items = list(items)
        self.calls = 0

    async def __call__(self, url, headers):
        self.calls += 1
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item


def run_realtime(connector, session, *, sleep: float = 0.3):
    processor = RealtimeTranslateProcessor(
        settings=InterpreterSettings(languages=("ru", "en"), direction="one_way", engine="realtime"),
        api_key=SECRET,
        session=session,
        connect=connector,
        reconnect_delay=0.01,
    )
    asyncio.run(
        run_test(processor, frames_to_send=[SleepFrame(sleep=sleep)], expected_down_frames=None)
    )


def test_realtime_close_3000_invalid_api_key_fails_without_retry() -> None:
    session = FakeSession()
    closed = ConnectionClosedError(Close(3000, "invalid_api_key"), None)
    connector = Connector(FakeWS([{"type": "session.created"}, closed]))
    run_realtime(connector, session)

    assert connector.calls == 1  # a rejected key is never retried
    [failed] = session.of("ProviderError")
    assert (failed["kind"], failed["provider"], failed["category"]) == ("realtime", "openai", "auth")
    assert SECRET not in json.dumps(failed)
    assert session.of("EngineFallback") == []  # the cascaded engine needs the same key
    assert session.failures == ["provider_failed"]
    assert session.end_requests == ["provider_failed"]


def test_realtime_401_on_connect_is_auth() -> None:
    session = FakeSession()
    request = httpx.Request("GET", "https://api.openai.com")
    rejected = httpx.HTTPStatusError("HTTP 401", request=request, response=httpx.Response(401, request=request))
    connector = Connector(rejected)
    run_realtime(connector, session)

    assert connector.calls == 1
    assert session.of("ProviderError")[0]["category"] == "auth"
    assert session.end_requests == ["provider_failed"]


def test_realtime_insufficient_quota_error_event_mid_call_ends_the_session() -> None:
    session = FakeSession()
    error = {"type": "error", "error": {"type": "insufficient_quota", "message": "You exceeded your quota"}}
    connector = Connector(FakeWS([{"type": "session.created"}, error]))
    run_realtime(connector, session)

    assert connector.calls == 1
    assert session.of("ProviderError")[0]["category"] == "quota"
    assert session.end_requests == ["provider_failed"]


def test_realtime_reconnect_that_works_emits_a_retryable_error_but_no_failure() -> None:
    session = FakeSession()
    connector = Connector(OSError("network down"), FakeWS([{"type": "session.created"}]))
    run_realtime(connector, session)

    assert connector.calls == 2
    [event] = session.of("ProviderError")
    assert (event["kind"], event["attempt"], event["retryable"]) == ("realtime", 1, True)
    assert session.failures == [] and session.end_requests == []


def test_realtime_outage_falls_back_to_cascaded_when_the_session_can() -> None:
    class FallbackSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.fallbacks = 0

        def request_engine_fallback(self) -> bool:
            self.fallbacks += 1
            return True

    session = FallbackSession()
    connector = Connector(OSError("network down"))
    run_realtime(connector, session)

    assert connector.calls == 2  # first attempt + exactly one retry
    assert session.fallbacks == 1
    retry, failed = session.of("ProviderError")
    assert (retry["attempt"], retry["retryable"], retry["stage"]) == (1, True, "connect")
    assert (failed["attempt"], failed["retryable"], failed["category"]) == (2, False, "unavailable")
    assert session.of("EngineFallback") == [{"from": "realtime", "to": "cascaded"}]
    assert session.failures == [] and session.end_requests == []


def test_realtime_outage_without_fallback_ends_the_session() -> None:
    session = FakeSession()
    connector = Connector(OSError("network down"))
    run_realtime(connector, session)

    assert [e["attempt"] for e in session.of("ProviderError")] == [1, 2]
    assert session.of("ProviderError")[-1]["category"] == "unavailable"
    assert session.of("EngineFallback") == []
    assert session.end_requests == ["provider_failed"]


# -- engine fallback in the session --------------------------------------------------------


def realtime_config(**overrides) -> PipelineConfig:
    values = dict(
        interpreter={"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"},
        mode="interpreter",
    )
    values.update(overrides)
    return PipelineConfig(
        "ws://livekit", "token", "room-42", "dg-key", "openai-key", "cartesia-key", "voice", "prompt",
        **values,
    )


def test_cascaded_fallback_config_only_when_the_session_can_run_cascaded() -> None:
    config = realtime_config()
    fallback = cascaded_fallback_config(config)
    assert fallback is not None and fallback.interpreter["engine"] == "cascaded"
    assert config.interpreter["engine"] == "realtime"  # the original is untouched

    no_stt = PipelineConfig(
        "ws://l", "t", "room", "", "openai-key", "", "", "prompt",
        mode="interpreter",
        interpreter={"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"},
    )
    assert cascaded_fallback_config(no_stt) is None  # no STT/TTS credentials configured
    assert cascaded_fallback_config(pipeline_config()) is None  # not an interpreter
    cascaded = realtime_config(
        interpreter={"languages": ["ru", "en"], "direction": "one_way", "engine": "cascaded"}
    )
    assert cascaded_fallback_config(cascaded) is None
    two_person = realtime_config(participants={"a": {"language": "ru"}, "b": {"language": "en"}})
    assert cascaded_fallback_config(two_person) is None


def test_session_restarts_the_pipeline_on_the_cascaded_engine(monkeypatch) -> None:
    import pipecat.workers.runner

    client = RecordingClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    built: list[str] = []

    class Worker:
        def __init__(self) -> None:
            self.cancelled: list[str | None] = []

        def cancel(self, *, reason=None) -> None:
            self.cancelled.append(reason)

    workers: list[Worker] = []

    def build(config, _session):
        built.append(config.interpreter["engine"])
        workers.append(Worker())
        transport = FakeTransport()
        return transport, workers[-1]

    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", build)
    runs: list[int] = []

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            self.worker = worker

        async def run(self) -> None:
            runs.append(1)
            if len(runs) == 1:
                # The realtime engine dies of an outage mid-call.
                outcome = report_provider_failure(
                    session, "realtime", "openai", "unavailable", fatal=True, allow_fallback=True
                )
                assert outcome == "fallback"
                for _ in range(100):  # wait for the end request to reach the worker
                    if self.worker.cancelled:
                        break
                    await asyncio.sleep(0.01)
                assert self.worker.cancelled == ["engine_fallback"]

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(asyncio.wait_for(session.run_pipeline(realtime_config()), TIMEOUT))

    assert built == ["realtime", "cascaded"]
    assert client.types() == ["ProviderError", "EngineFallback", "SessionEnded"]
    assert client.of("EngineFallback") == [{"from": "realtime", "to": "cascaded"}]
    # the call itself carried on; nobody asked it to stop, so the transport closing it is
    assert client.of("SessionEnded") == [
        {"outcome": "ended", "reason": "participant_disconnected"}
    ]


def test_a_provider_failing_after_the_fallback_ends_the_session(monkeypatch) -> None:
    import pipecat.workers.runner

    client = RecordingClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    cancelled: list[str | None] = []

    class Worker:
        def cancel(self, *, reason=None) -> None:
            cancelled.append(reason)

    monkeypatch.setattr(
        src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (FakeTransport(), Worker())
    )
    runs: list[int] = []

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            pass

        async def run(self) -> None:
            runs.append(1)
            if len(runs) == 1:
                report_provider_failure(
                    session, "realtime", "openai", "unavailable", fatal=True, allow_fallback=True
                )
            else:  # the cascaded TTS runs out of credit
                outcome = report_provider_failure(session, "tts", "cartesia", "quota")
                assert outcome == "failed"
            for _ in range(100):
                if len(cancelled) == len(runs):
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("pipeline was not asked to stop")

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(asyncio.wait_for(session.run_pipeline(realtime_config()), TIMEOUT))

    assert len(runs) == 2
    assert cancelled == ["engine_fallback", "provider_failed"]
    assert client.types() == [
        "ProviderError",
        "EngineFallback",
        "ProviderError",
        "SessionFailed",
    ]
    assert client.of("SessionFailed") == [{"outcome": "failed", "reason": "provider_failed"}]


# -- end / hang-up while an engine fallback is pending (H2) ---------------------------------


def _run_with_end_during_fallback(monkeypatch, end, *, wait_for_cancel: bool = True):
    """Run a realtime session whose engine fails; ``end(session, transport)`` fires mid-fallback."""

    import pipecat.workers.runner

    client = RecordingClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    built: list[str] = []
    cancelled: list[str | None] = []
    results: list[object] = []

    class Worker:
        def cancel(self, *, reason=None) -> None:
            cancelled.append(reason)

    transports: list[FakeTransport] = []

    def build(config, _session):
        built.append(config.interpreter["engine"])
        transports.append(FakeTransport())
        return transports[-1], Worker()

    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", build)

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            pass

        async def run(self) -> None:
            if len(built) > 1:
                # A (wrongly) restarted pipeline keeps running until somebody stops it.
                for _ in range(30):
                    if len(cancelled) > 1:
                        return
                    await asyncio.sleep(0.01)
                raise AssertionError("pipeline restarted on the cascaded engine")
            outcome = report_provider_failure(
                session, "realtime", "openai", "unavailable", fatal=True, allow_fallback=True
            )
            assert outcome == "fallback"
            if wait_for_cancel:
                for _ in range(100):
                    if cancelled:
                        break
                    await asyncio.sleep(0.01)
                assert cancelled == ["engine_fallback"]
            results.append(await end(session, transports[0]))

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(asyncio.wait_for(session.run_pipeline(realtime_config()), TIMEOUT))
    return client, session, built, results


def test_end_during_a_pending_fallback_ends_the_session(monkeypatch) -> None:
    async def end(session, _transport):
        return await session.request_end("Session ended by request")

    client, session, built, results = _run_with_end_during_fallback(monkeypatch, end)

    assert results == [True]
    assert built == ["realtime"]  # no cascaded rebuild
    assert session.state == SessionState.ENDED
    assert client.types()[-1] == "SessionEnded"
    assert client.of("SessionEnded") == [{"outcome": "ended", "reason": "platform_end"}]


def test_end_before_the_fallback_cancel_runs_still_ends_the_session(monkeypatch) -> None:
    async def end(session, _transport):
        return await session.request_end("Session ended by request")

    client, session, built, _ = _run_with_end_during_fallback(
        monkeypatch, end, wait_for_cancel=False
    )

    assert built == ["realtime"]
    assert session.state == SessionState.ENDED
    assert client.of("SessionEnded") == [{"outcome": "ended", "reason": "platform_end"}]


def test_participant_disconnect_during_a_pending_fallback_ends_the_session(monkeypatch) -> None:
    async def end(_session, transport):
        await transport.handlers["on_participant_disconnected"](transport, "caller")

    client, session, built, _ = _run_with_end_during_fallback(monkeypatch, end)

    assert built == ["realtime"]
    assert session.state == SessionState.ENDED
    assert client.of("SessionEnded") == [
        {"outcome": "ended", "reason": "participant_disconnected"}
    ]

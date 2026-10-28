"""ST-13 (runtime half): end reasons, idle timeout, max duration, admission control."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.runtime import api, lifecycle
from src.runtime.fallback_lines import IDLE, line
from src.runtime.lifecycle import (
    END_REASONS,
    EndReason,
    LifecycleMonitor,
    classify_end_reason,
    terminal_payload,
)
from src.runtime.session import RuntimeSession


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


class FakeSession:
    """Just what the monitor touches."""

    def __init__(self, max_duration_secs=None) -> None:
        self.session_id = uuid4()
        self.max_duration_secs = max_duration_secs
        self.lifecycle_started_at = None
        self.ends: list[str] = []

    async def request_end(self, reason: str = "") -> bool:
        self.ends.append(reason)
        return True


def make(clock, *, session=None, interpreter=False, **kwargs):
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    session = session or FakeSession()
    kwargs.setdefault("prompt_secs", 60)
    kwargs.setdefault("end_secs", 90)
    kwargs.setdefault("max_secs", 1800)
    monitor = LifecycleMonitor(
        session, speak=speak, interpreter=interpreter, clock=clock, **kwargs
    )
    return monitor, session, spoken


def check(monitor):
    asyncio.run(monitor.check())


# -- end reasons -------------------------------------------------------------------------


def test_end_reason_values_are_the_stable_set() -> None:
    assert END_REASONS == {
        "user_hangup",
        "participant_disconnected",
        "idle_timeout",
        "max_duration",
        "provider_failed",
        "runtime_error",
        "capacity",
        "platform_end",
        "cancelled",
    }


@pytest.mark.parametrize(
    "raw, expected, detail",
    [
        ("LiveKit disconnected", "participant_disconnected", None),
        ("Browser participant disconnected", "participant_disconnected", None),
        ("participant_disconnected", "participant_disconnected", None),
        ("Session ended by request", "platform_end", None),
        ("provider_failed", "provider_failed", None),
        ("idle_timeout", "idle_timeout", None),
        ("max_duration", "max_duration", None),
        ("user_hangup", "user_hangup", None),
        ("event_delivery_failed", "runtime_error", "event_delivery_failed"),
        ("Platform event delivery failed", "runtime_error", "event_delivery_failed"),
        ("Runtime event queue capacity exceeded", "runtime_error", "event_queue_capacity_exceeded"),
        ("SomeWeirdError", "runtime_error", "SomeWeirdError"),
        (None, "participant_disconnected", None),
        (EndReason.CANCELLED, "cancelled", None),
    ],
)
def test_classify_end_reason(raw, expected, detail) -> None:
    reason, mapped_detail = classify_end_reason(raw)
    assert reason.value == expected and mapped_detail == detail


def test_terminal_payload_only_carries_stable_reasons() -> None:
    assert terminal_payload("ended", "Session ended by request") == {
        "outcome": "ended",
        "reason": "platform_end",
    }
    assert terminal_payload("failed", EndReason.RUNTIME_ERROR, detail="ValueError") == {
        "outcome": "failed",
        "reason": "runtime_error",
        "detail": "ValueError",
    }
    assert terminal_payload("failed", "KeyError")["reason"] in END_REASONS


class RecordingClient:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


def run_session_with(monkeypatch, run, *, setup=None):
    """Run RuntimeSession.run_pipeline with a fake worker; ``run(session)`` is the pipeline."""

    import pipecat.workers.runner

    import src.runtime.pipeline as pipeline

    client = RecordingClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    if setup:
        setup(session)

    class Transport:
        def add_event_handler(self, *_):
            pass

        async def cleanup(self):
            pass

    class Worker:
        def __init__(self):
            self.reasons = []

        def cancel(self, *, reason=None):
            self.reasons.append(reason)

    worker = Worker()
    monkeypatch.setattr(pipeline, "build_livekit_pipeline", lambda *_: (Transport(), worker))

    class Runner:
        def __init__(self, **_):
            pass

        async def add_workers(self, _worker):
            pass

        async def run(self):
            await run(session)

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    config = pipeline.PipelineConfig(
        "ws://livekit", "token", "room", "deepgram", "openai", "cartesia", "voice", "prompt"
    )
    return session, client, worker, config


def terminal(client):
    return [e for e in client.events if e.event_type in {"SessionEnded", "SessionFailed"}]


@pytest.mark.parametrize(
    "requested, event_type, expected",
    [
        ("idle_timeout", "SessionEnded", {"outcome": "ended", "reason": "idle_timeout"}),
        ("max_duration", "SessionEnded", {"outcome": "ended", "reason": "max_duration"}),
        ("LiveKit disconnected", "SessionEnded", {"outcome": "ended", "reason": "participant_disconnected"}),
        ("Session ended by request", "SessionEnded", {"outcome": "ended", "reason": "platform_end"}),
    ],
)
def test_requested_end_reason_reaches_the_terminal_event(
    monkeypatch, requested, event_type, expected
) -> None:
    async def run(session):
        assert await session.request_end(requested)

    session, client, worker, config = run_session_with(monkeypatch, run)
    asyncio.run(session.run_pipeline(config))

    assert worker.reasons == [requested]
    assert [(e.event_type, e.payload) for e in terminal(client)] == [(event_type, expected)]


def test_unknown_fail_reason_is_a_runtime_error_with_detail(monkeypatch) -> None:
    async def run(session):
        session.fail("boom")

    session, client, _, config = run_session_with(monkeypatch, run)
    asyncio.run(session.run_pipeline(config))

    assert terminal(client)[0].payload == {
        "outcome": "failed",
        "reason": "runtime_error",
        "detail": "boom",
    }


def test_pipeline_exception_is_runtime_error_with_class_name_detail(monkeypatch) -> None:
    async def run(session):
        raise ValueError("bad")

    session, client, _, config = run_session_with(monkeypatch, run)
    with pytest.raises(ValueError):
        asyncio.run(session.run_pipeline(config))

    assert terminal(client)[0].payload == {
        "outcome": "failed",
        "reason": "runtime_error",
        "detail": "ValueError",
    }


def test_cancellation_is_reason_cancelled(monkeypatch) -> None:
    async def run(session):
        raise asyncio.CancelledError

    session, client, _, config = run_session_with(monkeypatch, run)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(session.run_pipeline(config))

    assert terminal(client)[0].payload == {"outcome": "cancelled", "reason": "cancelled"}


def test_run_pipeline_attaches_the_lifecycle_observer(monkeypatch) -> None:
    attached = []
    monkeypatch.setattr(
        lifecycle, "attach_lifecycle", lambda worker, session, config=None, **kw: attached.append(worker)
    )

    async def run(session):
        pass

    session, _, worker, config = run_session_with(monkeypatch, run)
    asyncio.run(session.run_pipeline(config))
    assert attached == [worker]


# -- idle timeout --------------------------------------------------------------------------


def test_idle_prompt_then_end() -> None:
    clock = Clock()
    monitor, session, spoken = make(clock)

    clock.advance(59)
    check(monitor)
    assert spoken == [] and session.ends == []

    clock.advance(2)  # 61s of silence
    check(monitor)
    assert spoken == [line(IDLE, "en")] and session.ends == []
    check(monitor)  # prompted once only
    assert spoken == [line(IDLE, "en")]

    clock.advance(30)  # 91s
    check(monitor)
    assert session.ends == ["idle_timeout"]
    check(monitor)
    assert session.ends == ["idle_timeout"]  # ended once


def test_user_speech_resets_the_idle_clock_and_the_prompt() -> None:
    clock = Clock()
    monitor, session, spoken = make(clock)

    clock.advance(70)
    check(monitor)
    assert len(spoken) == 1

    monitor.user_started()
    monitor.user_stopped()
    clock.advance(80)  # would be past the end deadline without the reset
    check(monitor)
    assert session.ends == []
    assert len(spoken) == 2  # a fresh silence prompts again
    clock.advance(30)
    check(monitor)
    assert session.ends == ["idle_timeout"]


def test_bot_speech_before_the_prompt_counts_but_the_prompt_itself_does_not() -> None:
    clock = Clock()
    monitor, session, spoken = make(clock)

    clock.advance(50)
    monitor.bot_started()
    monitor.bot_stopped()
    clock.advance(50)  # 50s since the bot spoke
    check(monitor)
    assert spoken == []

    clock.advance(15)
    check(monitor)
    assert len(spoken) == 1
    monitor.bot_started()  # the prompt is being spoken
    clock.advance(5)
    monitor.bot_stopped()
    clock.advance(25)  # 95s after the last real activity (the bot's earlier speech)
    check(monitor)
    assert session.ends == ["idle_timeout"]


def test_nothing_ends_while_someone_is_speaking() -> None:
    clock = Clock()
    monitor, session, spoken = make(clock)

    monitor.user_started()
    clock.advance(500)
    check(monitor)
    assert session.ends == [] and spoken == []
    monitor.user_stopped()
    clock.advance(10)
    check(monitor)
    assert session.ends == []


def test_interpreter_never_prompts_and_ends_at_the_idle_end() -> None:
    clock = Clock()
    monitor, session, spoken = make(clock, interpreter=True)

    clock.advance(75)
    check(monitor)
    assert spoken == [] and session.ends == []
    clock.advance(20)
    check(monitor)
    assert spoken == [] and session.ends == ["idle_timeout"]


def test_idle_prompt_language_follows_the_caller() -> None:
    clock = Clock()
    spoken = []

    async def speak(text):
        spoken.append(text)

    monitor = LifecycleMonitor(
        FakeSession(),
        speak=speak,
        language=lambda: "ru",
        prompt_secs=60,
        end_secs=90,
        max_secs=1800,
        clock=clock,
    )
    clock.advance(61)
    check(monitor)
    assert spoken == [line(IDLE, "ru")]
    assert spoken[0] != line(IDLE, "en")


def test_loop_ends_an_idle_session_with_a_fake_sleep() -> None:
    clock = Clock()
    session = FakeSession()

    async def fake_sleep(secs: float) -> None:
        clock.advance(secs)
        await asyncio.sleep(0)

    spoken = []

    async def speak(text):
        spoken.append(text)

    monitor = LifecycleMonitor(
        session,
        speak=speak,
        prompt_secs=5,
        end_secs=8,
        max_secs=1800,
        clock=clock,
        sleep=fake_sleep,
    )
    asyncio.run(asyncio.wait_for(monitor.run(), 5))
    assert len(spoken) == 1 and session.ends == ["idle_timeout"]


# -- max duration ----------------------------------------------------------------------------


def test_max_duration_ends_the_session_even_when_active() -> None:
    clock = Clock()
    monitor, session, spoken = make(clock, max_secs=300)

    for _ in range(40):  # constant chatter
        clock.advance(9)
        monitor.user_started()
        monitor.user_stopped()
        check(monitor)
        if session.ends:
            break
    assert session.ends == ["max_duration"]
    assert spoken == []


def test_per_session_max_duration_overrides_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_MAX_SESSION_SECS", "1000")
    clock = Clock()
    session = FakeSession(max_duration_secs=120)
    monitor = LifecycleMonitor(session, speak=None, clock=clock, prompt_secs=600, end_secs=900)
    clock.advance(119)
    check(monitor)
    assert session.ends == []
    clock.advance(2)
    check(monitor)
    assert session.ends == ["max_duration"]


def test_max_duration_spans_a_pipeline_restart(monkeypatch) -> None:
    clock = Clock()
    session = FakeSession()
    first = LifecycleMonitor(session, speak=None, clock=clock, max_secs=100, prompt_secs=600, end_secs=900)
    first.stop()
    clock.advance(90)
    second = LifecycleMonitor(session, speak=None, clock=clock, max_secs=100, prompt_secs=600, end_secs=900)
    clock.advance(11)
    check(second)
    assert session.ends == ["max_duration"]


def test_environment_knobs(monkeypatch) -> None:
    for name in (
        "RUNTIME_IDLE_PROMPT_SECS",
        "RUNTIME_IDLE_END_SECS",
        "RUNTIME_MAX_SESSION_SECS",
        "RUNTIME_MAX_SESSIONS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert lifecycle.idle_prompt_secs() == 60
    assert lifecycle.idle_end_secs() == 90
    assert lifecycle.max_session_secs() == 1800
    assert lifecycle.max_sessions() == 20
    monkeypatch.setenv("RUNTIME_IDLE_PROMPT_SECS", "5")
    monkeypatch.setenv("RUNTIME_MAX_SESSIONS", "3")
    monkeypatch.setenv("RUNTIME_MAX_SESSION_SECS", "garbage")
    assert lifecycle.idle_prompt_secs() == 5
    assert lifecycle.max_sessions() == 3
    assert lifecycle.max_session_secs() == 1800


# -- observer ------------------------------------------------------------------------------------


def push(observer, frame):
    asyncio.run(observer.on_push_frame(SimpleNamespace(frame=frame)))


def test_observer_feeds_speech_activity_to_the_monitor() -> None:
    pytest.importorskip("pipecat")
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        StartFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
    )

    clock = Clock()
    monitor, session, spoken = make(clock)
    observer = lifecycle.observer_class()(monitor)

    async def scenario():
        await observer.on_push_frame(SimpleNamespace(frame=StartFrame()))  # starts the loop
        assert monitor._task is not None
        for frame, user, bot in [
            (UserStartedSpeakingFrame(), True, False),
            (UserStoppedSpeakingFrame(), False, False),
            (VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=0.0), True, False),
            (BotStartedSpeakingFrame(), False, True),
            (BotStoppedSpeakingFrame(), False, False),
        ]:
            clock.advance(40)
            before = monitor._last_activity
            await observer.on_push_frame(SimpleNamespace(frame=frame))
            assert monitor._last_activity > before
            if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
                monitor.user_stopped()  # keep the next frame's checks simple
        await observer.cleanup()
        assert monitor._task is None

    asyncio.run(scenario())
    assert session.ends == []


def test_attach_lifecycle_adds_one_observer_and_queues_a_tts_speak_frame() -> None:
    pytest.importorskip("pipecat")
    from pipecat.frames.frames import TTSSpeakFrame

    queued, observers = [], []

    class Worker:
        def add_observer(self, observer):
            observers.append(observer)

        async def queue_frame(self, frame):
            queued.append(frame)

    clock = Clock()
    session = FakeSession()
    config = SimpleNamespace(mode="agent", provider_settings={"tts": {"language": "es"}})
    monitor = lifecycle.attach_lifecycle(
        Worker(), session, config, clock=clock, prompt_secs=60, end_secs=90, max_secs=1800
    )
    assert len(observers) == 1 and monitor is not None
    clock.advance(61)
    check(monitor)
    assert len(queued) == 1 and isinstance(queued[0], TTSSpeakFrame)
    assert queued[0].text == line(IDLE, "es")
    assert queued[0].append_to_context is False


def test_attach_lifecycle_is_a_noop_without_session_or_observers() -> None:
    assert lifecycle.attach_lifecycle(object(), FakeSession()) is None
    assert lifecycle.attach_lifecycle(SimpleNamespace(add_observer=lambda o: None), None) is None


def test_attach_lifecycle_marks_interpreter_mode() -> None:
    pytest.importorskip("pipecat")
    worker = SimpleNamespace(add_observer=lambda o: None)
    monitor = lifecycle.attach_lifecycle(worker, FakeSession(), SimpleNamespace(mode="interpreter"))
    assert monitor._interpreter is True


# -- admission control --------------------------------------------------------------------------------


@pytest.fixture
def held_sessions(monkeypatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "test-runtime-token")
    release: dict[str, asyncio.Event] = {}
    keys: list[str] = []

    async def hold_session(request: api.SessionStart) -> None:
        event = release.setdefault(str(request.session_id), asyncio.Event())
        try:
            await event.wait()
        finally:
            api._active_sessions.pop(str(request.session_id), None)

    monkeypatch.setattr(api, "run_session", hold_session)
    yield keys
    for key in keys:
        for store in (api._active_sessions, api._session_states, api._session_contracts):
            store.pop(key, None)


def payload():
    return {
        "session_id": str(uuid4()),
        "conversation_id": str(uuid4()),
        "agent_id": str(uuid4()),
        "room_name": "voice-test-room",
    }


HEADERS = {"X-Runtime-Token": "test-runtime-token"}


def test_start_session_is_refused_at_capacity_and_recovers(monkeypatch, held_sessions) -> None:
    monkeypatch.setenv("RUNTIME_MAX_SESSIONS", "2")
    first, second, third = payload(), payload(), payload()
    held_sessions.extend(p["session_id"] for p in (first, second, third))

    with TestClient(api.app) as client:
        assert client.post("/sessions", json=first, headers=HEADERS).status_code == 202
        assert client.post("/sessions", json=second, headers=HEADERS).status_code == 202
        assert client.get("/health").json() == {
            "status": "ok",
            "active_sessions": 2,
            "max_sessions": 2,
        }

        refused = client.post("/sessions", json=third, headers=HEADERS)
        assert refused.status_code == 503
        body = refused.json()
        assert body["code"] == "runtime_at_capacity" and body["message"]
        # nothing was started or remembered for the refused session
        assert third["session_id"] not in api._active_sessions
        assert third["session_id"] not in api._session_states
        assert third["session_id"] not in api._session_contracts

        # a retry of a session that is already running is not a new admission
        again = client.post("/sessions", json=first, headers=HEADERS)
        assert again.status_code == 202 and again.json() == {"status": "already_running"}

        # an ended session frees its slot
        ended = client.post(f"/sessions/{first['session_id']}/end", headers=HEADERS)
        assert ended.status_code == 200
        assert client.get("/health").json()["active_sessions"] == 1
        accepted = client.post("/sessions", json=third, headers=HEADERS)
        assert accepted.status_code == 202 and accepted.json() == {"status": "accepted"}


def test_health_reports_the_default_cap(monkeypatch) -> None:
    monkeypatch.delenv("RUNTIME_MAX_SESSIONS", raising=False)
    with TestClient(api.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["max_sessions"] == 20 and body["active_sessions"] >= 0


def test_max_duration_secs_is_part_of_the_start_contract() -> None:
    base = dict(session_id=uuid4(), conversation_id=uuid4(), agent_id=uuid4(), room_name="r")
    assert "max_duration_secs" not in api.SessionStart(**base).model_dump(mode="json")
    assert api.SessionStart(**base, max_duration_secs=600).model_dump(mode="json")["max_duration_secs"] == 600
    with pytest.raises(ValueError):
        api.SessionStart(**base, max_duration_secs=0)

"""Opt-in opening greeting for plain agent calls (RUNTIME_AGENT_GREETING)."""

import asyncio
from uuid import uuid4

import pytest

import src.runtime.pipeline
from src.runtime.session import RuntimeSession
from test_apology import Client, Transport, config


class GreetingWorker:
    def __init__(self) -> None:
        self.queued: list = []

    async def queue_frame(self, frame, *args, **kwargs) -> None:
        self.queued.append(frame)

    def cancel(self, *, reason=None) -> None:
        pass


def run_call(monkeypatch, cfg, track_events: int = 2):
    """Run a session whose caller's microphone track is subscribed ``track_events`` times."""

    import pipecat.workers.runner

    transport = Transport([])
    worker = GreetingWorker()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), Client([]))
    monkeypatch.setattr(src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (transport, worker))

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, _worker) -> None:
            pass

        async def run(self) -> None:
            handler = transport.handlers.get("on_audio_track_subscribed")
            for _ in range(track_events):
                if handler:
                    await handler(transport, "browser-1")

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    asyncio.run(asyncio.wait_for(session.run_pipeline(cfg), 10))
    return worker


def test_no_greeting_unless_enabled(monkeypatch) -> None:
    monkeypatch.delenv("RUNTIME_AGENT_GREETING", raising=False)
    assert run_call(monkeypatch, config()).queued == []


def test_greeting_is_spoken_once_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_AGENT_GREETING", "Hello! Ask me anything about applying to Inha.")
    (frame,) = run_call(monkeypatch, config()).queued
    assert frame.text == "Hello! Ask me anything about applying to Inha."


def test_no_greeting_on_a_translated_agent_call(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_AGENT_GREETING", "Hello!")
    cfg = config(translation={"working_language": "en", "caller_languages": ["ko"]})
    assert run_call(monkeypatch, cfg).queued == []

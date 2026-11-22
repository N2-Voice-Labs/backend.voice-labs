"""Fire-and-forget tasks keep a reference until done and log what went wrong."""

import asyncio
import logging

import pytest

pytest.importorskip("pipecat")

from src.runtime import translation_engine
from src.runtime.participants import ParticipantDirectory


def test_dropped_utterance_apology_task_is_referenced_and_failures_are_logged(caplog) -> None:
    release = asyncio.Event()

    class FakeSession:
        async def play_apology(self, *, speaker_identity):
            await release.wait()
            raise RuntimeError("apology exploded")

    async def scenario() -> None:
        translation_engine.dropped_utterance_hook(FakeSession())("alice")
        await asyncio.sleep(0)
        assert len(translation_engine._background_tasks) == 1  # strongly referenced
        release.set()
        for _ in range(5):
            await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR, logger=translation_engine.__name__):
        asyncio.run(scenario())

    assert translation_engine._background_tasks == set()
    assert any(r.exc_info and "apology exploded" in str(r.exc_info[1]) for r in caplog.records)


def test_phone_stream_close_task_is_referenced_until_done() -> None:
    pytest.importorskip("livekit.rtc")
    from src.runtime import audio_routing

    closed = asyncio.Event()

    class Stream:
        async def aclose(self) -> None:
            await asyncio.sleep(0)
            closed.set()

    async def scenario() -> None:
        directory = ParticipantDirectory.from_mapping(
            ["ru", "en"], {"a": {"language": "ru"}, "b": {"language": "en"}}
        )
        router = audio_routing.ListenerAudioRouter(directory)
        forward = asyncio.create_task(asyncio.sleep(10))
        router._phone_streams["a"] = (Stream(), forward)
        router._stop_phone_stream("a")
        assert len(audio_routing._background_tasks) == 1
        await asyncio.wait_for(closed.wait(), 1)
        await asyncio.gather(forward, return_exceptions=True)
        for _ in range(3):
            await asyncio.sleep(0)

    asyncio.run(scenario())

    assert audio_routing._background_tasks == set()


def test_background_done_logs_and_discards_a_failed_task(caplog) -> None:
    from src.runtime.translation_engine import _background_done, _background_tasks

    async def scenario() -> None:
        async def boom() -> None:
            raise ValueError("boom")

        task = asyncio.ensure_future(boom())
        _background_tasks.add(task)
        task.add_done_callback(_background_done)
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

    with caplog.at_level(logging.ERROR, logger=translation_engine.__name__):
        asyncio.run(scenario())

    assert not _background_tasks
    assert any("Background task failed" in r.getMessage() for r in caplog.records)

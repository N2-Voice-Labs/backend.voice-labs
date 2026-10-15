import asyncio
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    MetricsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection

from src.runtime.frame_events import RuntimeEventProcessor, latency_stage
from src.runtime.session import RuntimeSession


class RecordingEventClient:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


def ttfb(processor: str, value: float, model: str | None = None) -> TTFBMetricsData:
    return TTFBMetricsData.model_construct(processor=processor, model=model, value=value)


async def run(processor_factory, frames_by_instance):
    """Send each frame through each processor instance; return (events, forwarded)."""

    client = RecordingEventClient()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    processors = processor_factory(session)
    forwarded = []
    for processor in processors:
        processor.forwarded = []

        async def record(frame, direction, processor=processor):
            processor.forwarded.append(frame)

        processor.push_frame = record
    for frame in frames_by_instance:
        for processor in processors:
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    await session.close_event_dispatcher()
    for processor in processors:
        forwarded.append(processor.forwarded)
    return client.events, forwarded


def latency_payloads(events):
    return [e.payload for e in events if e.event_type == "LatencyMeasured"]


def test_stage_classification_by_class_name() -> None:
    assert latency_stage("DeepgramSTTService#0") == "stt"
    assert latency_stage("OpenAILLMService#1") == "llm"
    assert latency_stage("CartesiaTTSService#0") == "tts"
    assert latency_stage("SomethingElse#0") == "other"


def test_ttfb_metrics_emit_latency_events_with_exact_payload() -> None:
    frame = MetricsFrame(
        data=[
            ttfb("DeepgramSTTService#0", 0.1234),
            ttfb("OpenAILLMService#0", 0.5, "gpt-x"),
            ttfb("CartesiaTTSService#0", 0.25, "sonic"),
            ttfb("Weird#0", 1),
        ]
    )
    events, _ = asyncio.run(run(lambda s: [RuntimeEventProcessor(s)], [frame]))
    assert latency_payloads(events) == [
        {"metric": "stt_ttfb", "value_ms": 123.4, "processor": "DeepgramSTTService#0", "model": None},
        {"metric": "llm_ttfb", "value_ms": 500.0, "processor": "OpenAILLMService#0", "model": "gpt-x"},
        {"metric": "tts_ttfb", "value_ms": 250.0, "processor": "CartesiaTTSService#0", "model": "sonic"},
        {"metric": "other_ttfb", "value_ms": 1000.0, "processor": "Weird#0", "model": None},
    ]


def test_only_final_pipeline_instance_emits_and_frames_pass_through() -> None:
    frame = MetricsFrame(data=[ttfb("OpenAILLMService#0", 0.2)])
    events, forwarded = asyncio.run(
        run(
            lambda s: [
                RuntimeEventProcessor(s, control_events=False),
                RuntimeEventProcessor(s, assistant_output=True, control_events=False),
                RuntimeEventProcessor(s),
            ],
            [frame],
        )
    )
    assert len(latency_payloads(events)) == 1
    assert all(f == [frame] for f in forwarded)


def test_invalid_values_and_non_ttfb_metrics_are_skipped() -> None:
    frame = MetricsFrame(
        data=[
            ttfb("A-STT#0", float("nan")),
            ttfb("A-STT#0", float("inf")),
            ttfb("A-STT#0", -0.1),
            ProcessingMetricsData.model_construct(processor="A-LLM#0", model=None, value=0.3),
            ttfb("A-TTS#0", 0.0),
        ]
    )
    events, forwarded = asyncio.run(run(lambda s: [RuntimeEventProcessor(s)], [frame]))
    assert [p["metric"] for p in latency_payloads(events)] == ["tts_ttfb"]
    assert forwarded[0] == [frame]


def test_turn_to_first_audio_measured_once_per_turn(monkeypatch) -> None:
    clock = iter([10.0, 10.75])
    fake_time = type("FakeTime", (), {"monotonic": staticmethod(lambda: next(clock))})
    monkeypatch.setattr("src.runtime.frame_events.time", fake_time)
    frames = [
        UserStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        BotStartedSpeakingFrame(),  # no new user turn: no second measurement
    ]
    events, _ = asyncio.run(run(lambda s: [RuntimeEventProcessor(s)], frames))
    assert latency_payloads(events) == [
        {"metric": "turn_to_first_audio", "value_ms": 750.0, "processor": "runtime", "model": None}
    ]


def test_user_resuming_speech_discards_pending_turn_start() -> None:
    frames = [UserStoppedSpeakingFrame(), UserStartedSpeakingFrame(), BotStartedSpeakingFrame()]
    events, _ = asyncio.run(run(lambda s: [RuntimeEventProcessor(s)], frames))
    assert latency_payloads(events) == []


def test_non_final_instances_do_not_measure_turn_latency() -> None:
    frames = [UserStoppedSpeakingFrame(), BotStartedSpeakingFrame()]
    events, _ = asyncio.run(
        run(lambda s: [RuntimeEventProcessor(s, assistant_output=True)], frames)
    )
    assert latency_payloads(events) == []

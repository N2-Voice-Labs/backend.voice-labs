"""Realtime (gpt-realtime-translate) interpreter engine, against a fake WebSocket."""

import asyncio
import base64
import json
from uuid import uuid4

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    CancelFrame,
    InputAudioRawFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

from src.runtime.interpreter import InterpreterSettings
from src.runtime.pipeline import PipelineConfig, build_livekit_pipeline
from src.runtime.realtime_translate import (
    is_voiced,
    REALTIME_TRANSLATE_OUTPUT_LANGUAGES,
    RealtimeTranslateProcessor,
    build_session_update,
    realtime_translate_url,
)
from src.runtime.session import RuntimeSession

_END = object()


class FakeSession:
    """Collects (event_type, payload) like RuntimeSession.enqueue_event; records failure."""

    def __init__(self) -> None:
        self.event_client = object()
        self.session_id = uuid4()
        self.agent_id = uuid4()
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


class FakeWS:
    """Stand-in for a websockets connection.

    ``script`` is a list of server events (dicts), floats (sleep seconds) or an Exception
    (raised from the receive loop); it is played back as soon as the receiver starts.
    The server answers ``session.close`` with ``session.closed`` (plus ``on_close`` events).
    """

    def __init__(self, script=(), *, on_close=(), hang_close=False) -> None:
        self.hang_close = hang_close
        self.sent: list[dict] = []
        self.closed = False
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._script = list(script)
        self._on_close = list(on_close)
        self._player: asyncio.Task | None = None

    async def send(self, message: str) -> None:
        event = json.loads(message)
        self.sent.append(event)
        if event["type"] == "session.close":
            for item in self._on_close:
                self._inbox.put_nowait(json.dumps(item))
            self._inbox.put_nowait(json.dumps({"type": "session.closed"}))
            self._inbox.put_nowait(_END)

    async def close(self) -> None:
        if self.hang_close:  # the real API does not answer the close handshake promptly
            await asyncio.sleep(60)
        self.closed = True

    def __aiter__(self):
        if self._player is None:
            self._player = asyncio.ensure_future(self._play())
        return self

    async def _play(self) -> None:
        for item in self._script:
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
            elif isinstance(item, Exception):
                self._inbox.put_nowait(item)
            else:
                self._inbox.put_nowait(json.dumps(item))

    async def __anext__(self):
        item = await self._inbox.get()
        if item is _END:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    def types(self) -> list[str]:
        return [event["type"] for event in self.sent]

    def appended(self) -> list[bytes]:
        return [
            base64.b64decode(event["audio"])
            for event in self.sent
            if event["type"] == "session.input_audio_buffer.append"
        ]


class Connector:
    """connect(url, headers) -> next item of ``sockets`` (an Exception is raised)."""

    def __init__(self, *sockets) -> None:
        self._sockets = list(sockets)
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        item = self._sockets.pop(0) if len(self._sockets) > 1 else self._sockets[0]
        if isinstance(item, Exception):
            raise item
        return item


def audio_delta(payload: bytes, **extra) -> dict:
    return {
        "type": "session.output_audio.delta",
        "delta": base64.b64encode(payload).decode(),
        **extra,
    }


def make_processor(connector, *, session=None, languages=("ru", "en"), **kwargs):
    kwargs.setdefault("reconnect_delay", 0.01)
    kwargs.setdefault("transcript_idle_seconds", 0.05)
    return RealtimeTranslateProcessor(
        settings=InterpreterSettings(languages=languages, direction="one_way", engine="realtime"),
        api_key="sk-test",
        session=session,
        connect=connector,
        **kwargs,
    )


def run(processor, frames):
    return asyncio.run(run_test(processor, frames_to_send=frames, expected_down_frames=None))


def mic(nbytes: int, rate: int = 24000, fill: int = 1) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=bytes([fill]) * nbytes, sample_rate=rate, num_channels=1)


def played(down):
    return [f for f in down if isinstance(f, TTSAudioRawFrame)]


# -- session configuration -----------------------------------------------------


def test_connects_and_configures_the_translation_session() -> None:
    ws = FakeWS([{"type": "session.created", "session": {"id": "sess_1"}}])
    connector = Connector(ws)
    session = FakeSession()
    run(make_processor(connector, session=session), [SleepFrame(sleep=0.05)])

    url, headers = connector.calls[0]
    assert url == "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
    assert headers["Authorization"] == "Bearer sk-test"
    assert len(headers["OpenAI-Safety-Identifier"]) == 64  # sha256 hex, not the raw id
    assert str(session.agent_id) not in headers["OpenAI-Safety-Identifier"]
    assert ws.sent[0] == {
        "type": "session.update",
        "session": {
            "audio": {
                "input": {
                    "transcription": {"model": "gpt-realtime-whisper"},
                    "noise_reduction": {"type": "near_field"},
                },
                "output": {"language": "en"},
            }
        },
    }
    assert ws.closed


def test_session_update_builder_options_and_url() -> None:
    update = build_session_update("ja", transcription_model=None, noise_reduction=None)
    assert update == {"type": "session.update", "session": {"audio": {"output": {"language": "ja"}}}}
    assert realtime_translate_url("m") == "wss://api.openai.com/v1/realtime/translations?model=m"


def test_rejects_unsupported_settings() -> None:
    with pytest.raises(RuntimeError, match="one_way"):
        RealtimeTranslateProcessor(
            settings=InterpreterSettings(languages=("en", "ru"), direction="auto", engine="realtime"),
            api_key="k",
        )
    with pytest.raises(RuntimeError, match="nl"):
        RealtimeTranslateProcessor(
            settings=InterpreterSettings(languages=("en", "nl"), direction="one_way", engine="realtime"),
            api_key="k",
        )
    assert REALTIME_TRANSLATE_OUTPUT_LANGUAGES == {
        "es", "pt", "fr", "ja", "ru", "zh", "de", "ko", "hi", "id", "vi", "it", "en",
    }  # fmt: skip


# -- input audio ---------------------------------------------------------------


def test_input_audio_is_appended_in_200ms_base64_pcm_chunks() -> None:
    ws = FakeWS()
    chunk = 9600  # 200 ms of 24 kHz mono PCM16
    frames = [mic(4800, fill=1), mic(4800, fill=2), mic(4800, fill=3), SleepFrame(sleep=0.05)]
    run(make_processor(Connector(ws)), frames)

    appended = ws.appended()
    assert [len(a) for a in appended] == [chunk, 4800]  # the tail is flushed on EndFrame
    assert appended[0] == bytes([1]) * 4800 + bytes([2]) * 4800
    assert appended[1] == bytes([3]) * 4800
    assert ws.types() == [
        "session.update",
        "session.input_audio_buffer.append",
        "session.input_audio_buffer.append",
        "session.close",
    ]


def test_input_audio_is_resampled_to_24khz() -> None:
    ws = FakeWS()
    frames = [mic(3200, rate=16000, fill=0) for _ in range(8)] + [SleepFrame(sleep=0.05)]
    run(make_processor(Connector(ws)), frames)

    appended = ws.appended()
    assert appended, "no audio reached the socket"
    assert all(len(a) == 9600 for a in appended[:-1])
    total = sum(len(a) for a in appended)
    assert 0.8 * 38400 <= total <= 1.05 * 38400  # 0.8 s at 16 kHz -> ~0.8 s at 24 kHz


# -- translated audio ----------------------------------------------------------


def test_translated_audio_is_pushed_in_order_at_the_reported_sample_rate() -> None:
    ws = FakeWS(
        [
            audio_delta(b"\x01\x10" * 100, sample_rate=24000, channels=1, format="pcm16"),
            audio_delta(b"\x02\x10" * 50, sample_rate=24000),
            audio_delta(b"\x03\x10" * 10, sample_rate=16000),
            audio_delta(b"\x04\x10" * 5),  # no sample_rate field: 24 kHz default
            {"type": "session.output_audio.delta", "delta": ""},  # empty: ignored
        ]
    )
    down, _ = run(make_processor(Connector(ws)), [SleepFrame(sleep=0.1)])

    frames = played(down)
    assert [f.audio[0] for f in frames] == [1, 2, 3, 4]
    assert [len(f.audio) for f in frames] == [200, 100, 20, 10]
    assert [f.sample_rate for f in frames] == [24000, 24000, 16000, 24000]
    assert all(f.num_channels == 1 for f in frames)


def test_input_audio_is_not_forwarded_downstream() -> None:
    down, _ = run(make_processor(Connector(FakeWS())), [mic(4800), SleepFrame(sleep=0.02)])
    assert not [f for f in down if isinstance(f, InputAudioRawFrame)]


def test_end_frame_flushes_the_tail_of_the_translation_before_ending() -> None:
    ws = FakeWS(on_close=[audio_delta(b"\x09\x10" * 20)])
    down, _ = run(make_processor(Connector(ws)), [mic(4800), SleepFrame(sleep=0.02)])

    assert ws.types()[-1] == "session.close"
    assert [f.audio[0] for f in played(down)] == [9]


# -- transcripts ---------------------------------------------------------------


def test_transcripts_become_user_and_assistant_events() -> None:
    session = FakeSession()
    ws = FakeWS(
        [
            {"type": "session.input_transcript.delta", "delta": "Привет, "},
            {"type": "session.output_transcript.delta", "delta": "Hello, "},
            {"type": "session.input_transcript.delta", "delta": "как дела?"},
            {"type": "session.output_transcript.delta", "delta": "how are you?"},
        ]
    )
    run(make_processor(Connector(ws), session=session), [SleepFrame(sleep=0.3)])

    assert session.of("TranscriptFinal") == [
        {
            "content": "Привет, как дела?",
            "role": "user",
            "primary_language": "ru",
            "detected_languages": ["ru"],
        },
        {
            "content": "Hello, how are you?",
            "role": "assistant",
            "primary_language": "en",
            "detected_languages": ["en"],
        },
    ]


def test_transcripts_split_into_utterances_on_idle() -> None:
    session = FakeSession()
    ws = FakeWS(
        [
            {"type": "session.input_transcript.delta", "delta": "one"},
            {"type": "session.output_transcript.delta", "delta": "uno"},
            0.3,
            {"type": "session.input_transcript.delta", "delta": "two"},
            {"type": "session.output_transcript.delta", "delta": "dos"},
        ]
    )
    run(make_processor(Connector(ws), session=session, languages=("en", "es")), [SleepFrame(sleep=0.5)])

    pairs = [(p["role"], p["content"]) for p in session.of("TranscriptFinal")]
    assert pairs == [("user", "one"), ("assistant", "uno"), ("user", "two"), ("assistant", "dos")]


def test_transcript_is_held_while_the_speaker_is_still_talking() -> None:
    session = FakeSession()
    ws = FakeWS(
        [
            {"type": "session.output_transcript.delta", "delta": "A"},
            0.3,
            {"type": "session.output_transcript.delta", "delta": " B"},
        ]
    )
    frames = [
        SleepFrame(sleep=0.02),
        VADUserStartedSpeakingFrame(),
        SleepFrame(sleep=0.45),  # longer than the idle window: still no flush
        VADUserStoppedSpeakingFrame(),
    ]
    run(make_processor(Connector(ws), session=session, transcript_idle_seconds=0.1), frames)

    assistant = [p["content"] for p in session.of("TranscriptFinal") if p["role"] == "assistant"]
    assert assistant == ["A B"]


def test_idle_flush_still_fires_while_translated_audio_keeps_streaming() -> None:
    """The real API streams audio continuously (silence included); it must not defer the flush."""

    session = FakeSession()
    script = [
        {"type": "session.input_transcript.delta", "delta": "one"},
        {"type": "session.output_transcript.delta", "delta": "uno"},
    ]
    for _ in range(14):  # 0.7 s of audio deltas, far longer than the 0.1 s idle window
        script += [0.05, audio_delta(b"\x00\x00" * 10)]
    script += [
        {"type": "session.input_transcript.delta", "delta": "two"},
        {"type": "session.output_transcript.delta", "delta": "dos"},
    ]
    run(
        make_processor(
            Connector(FakeWS(script)),
            session=session,
            languages=("en", "es"),
            transcript_idle_seconds=0.1,
        ),
        [SleepFrame(sleep=0.9)],
    )

    pairs = [(p["role"], p["content"]) for p in session.of("TranscriptFinal")]
    assert pairs == [("user", "one"), ("assistant", "uno"), ("user", "two"), ("assistant", "dos")]


def test_real_api_event_shapes_are_handled() -> None:
    """Shapes captured from the live API (2026-10-18): ids, elapsed_ms, spacing in deltas."""

    session = FakeSession()
    ws = FakeWS(
        [
            {
                "type": "session.created",
                "event_id": "event_1",
                "session": {
                    "id": "sess_1",
                    "type": "translation",
                    "model": "gpt-realtime-translate",
                    "audio": {"input": {"noise_reduction": None, "transcription": None}, "output": {"language": "es"}},
                },
            },
            {"type": "session.updated", "event_id": "event_2", "session": {"id": "sess_1"}},
            {"type": "session.input_transcript.delta", "event_id": "e3", "delta": " З", "elapsed_ms": 800},
            {"type": "session.input_transcript.delta", "event_id": "e4", "delta": "дравствуйте", "elapsed_ms": 800},
            {"type": "session.input_transcript.delta", "event_id": "e5", "delta": ".", "elapsed_ms": 800},
            {"type": "session.output_transcript.delta", "event_id": "e6", "delta": "Hello", "elapsed_ms": 2800},
            {"type": "session.output_transcript.delta", "event_id": "e7", "delta": ",", "elapsed_ms": 3200},
            {"type": "session.output_transcript.delta", "event_id": "e8", "delta": " I'd", "elapsed_ms": 3400},
        ]
    )
    run(make_processor(Connector(ws), session=session), [SleepFrame(sleep=0.3)])

    assert [(p["role"], p["content"]) for p in session.of("TranscriptFinal")] == [
        ("user", "Здравствуйте."),
        ("assistant", "Hello, I'd"),
    ]


def test_cancel_flushes_transcripts_and_does_not_wait_for_the_close_handshake(monkeypatch) -> None:
    import src.runtime.realtime_translate as module

    monkeypatch.setattr(module, "WS_CLOSE_TIMEOUT_SECONDS", 0.05)
    session = FakeSession()
    ws = FakeWS(
        [
            {"type": "session.input_transcript.delta", "delta": "hola"},
            {"type": "session.output_transcript.delta", "delta": "hello"},
        ],
        hang_close=True,
    )
    processor = make_processor(Connector(ws), session=session, transcript_idle_seconds=30)

    async def go():
        return await asyncio.wait_for(
            run_test(
                processor,
                frames_to_send=[SleepFrame(sleep=0.1), CancelFrame()],
                expected_down_frames=None,
                send_end_frame=False,
            ),
            timeout=10,
        )

    asyncio.run(go())
    assert [(p["role"], p["content"]) for p in session.of("TranscriptFinal")] == [
        ("user", "hola"),
        ("assistant", "hello"),
    ]


def test_end_frame_does_not_hang_on_a_socket_that_never_finishes_closing(monkeypatch) -> None:
    import src.runtime.realtime_translate as module

    monkeypatch.setattr(module, "WS_CLOSE_TIMEOUT_SECONDS", 0.05)
    ws = FakeWS(hang_close=True)
    processor = make_processor(Connector(ws))
    asyncio.run(
        asyncio.wait_for(
            run_test(processor, frames_to_send=[SleepFrame(sleep=0.05)], expected_down_frames=None),
            timeout=10,
        )
    )
    assert ws.types()[-1] == "session.close"


# -- VAD and latency -----------------------------------------------------------


def test_vad_frames_are_forwarded_and_mirrored_as_user_turn_frames() -> None:
    frames = [
        VADUserStartedSpeakingFrame(start_secs=0.1),
        SleepFrame(sleep=0.02),
        VADUserStoppedSpeakingFrame(stop_secs=0.2),
        SleepFrame(sleep=0.02),
    ]
    down, _ = run(make_processor(Connector(FakeWS())), frames)

    kinds = [
        type(f).__name__
        for f in down
        if isinstance(
            f,
            (
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
            ),
        )
    ]
    assert kinds == [
        "VADUserStartedSpeakingFrame",
        "UserStartedSpeakingFrame",
        "VADUserStoppedSpeakingFrame",
        "UserStoppedSpeakingFrame",
    ]


def test_first_audio_latency_is_measured_once_per_utterance_from_speech_start() -> None:
    session = FakeSession()
    ws = FakeWS(
        [
            0.25,
            audio_delta(b"\x01\x10" * 10),
            audio_delta(b"\x02\x10" * 10),  # same utterance: no second measurement
            0.25,
            audio_delta(b"\x03\x10" * 10),  # second utterance, speech start at ~0.35
        ]
    )
    frames = [
        VADUserStartedSpeakingFrame(start_secs=0.1),
        SleepFrame(sleep=0.35),
        VADUserStoppedSpeakingFrame(stop_secs=0.2),
        VADUserStartedSpeakingFrame(start_secs=0.0),
        SleepFrame(sleep=0.4),
    ]
    run(make_processor(Connector(ws), session=session), frames)

    metrics = [p for p in session.of("LatencyMeasured") if p["metric"] == "realtime_first_audio_ms"]
    assert len(metrics) == 2
    # speech start was backdated by the VAD confirmation window (0.1 s) + 0.25 s of model lag
    assert 330 <= metrics[0]["value_ms"] < 900
    assert 100 <= metrics[1]["value_ms"] < 900
    assert metrics[0]["processor"] == "realtime-translate"
    assert metrics[0]["model"] == "gpt-realtime-translate"


def test_no_latency_metric_without_a_speech_start() -> None:
    session = FakeSession()
    ws = FakeWS([audio_delta(b"\x01\x10" * 10)])
    run(make_processor(Connector(ws), session=session), [SleepFrame(sleep=0.1)])
    assert session.of("LatencyMeasured") == []


# -- failures ------------------------------------------------------------------


def test_reconnects_once_after_a_socket_error_and_keeps_translating() -> None:
    session = FakeSession()
    first = FakeWS([ConnectionError("reset by peer")])
    second = FakeWS([audio_delta(b"\x07\x10" * 20)])
    connector = Connector(first, second)
    down, _ = run(
        make_processor(connector, session=session),
        [SleepFrame(sleep=0.1), mic(9600), SleepFrame(sleep=0.1)],
    )

    assert len(connector.calls) == 2
    assert first.closed
    assert second.types()[0] == "session.update"  # configured again
    assert second.appended() == [bytes([1]) * 9600]
    assert [f.audio[0] for f in played(down)] == [7]
    assert session.failures == [] and session.end_requests == []


def test_reconnect_after_a_clean_server_close_too() -> None:
    first = FakeWS([{"type": "session.closed"}])
    second = FakeWS()
    connector = Connector(first, second)
    run(make_processor(connector), [SleepFrame(sleep=0.15)])
    assert len(connector.calls) == 2


def test_audio_captured_while_reconnecting_is_queued_and_sent_on_the_new_socket() -> None:
    first = FakeWS([ConnectionError("boom")])
    second = FakeWS()
    connector = Connector(first, second)
    frames = [SleepFrame(sleep=0.02), mic(9600, fill=5), SleepFrame(sleep=0.2)]
    run(make_processor(connector, reconnect_delay=0.1), frames)
    assert second.appended() == [bytes([5]) * 9600]


def test_failed_retry_is_fatal() -> None:
    session = FakeSession()
    connector = Connector(FakeWS([ConnectionError("boom")]), ConnectionError("still down"))
    run(make_processor(connector, session=session), [mic(9600), SleepFrame(sleep=0.3)])
    assert len(connector.calls) == 2
    assert session.failures == ["realtime_translate_failed"]


def test_fatal_error_marks_the_session_failed_and_ends_it() -> None:
    session = FakeSession()
    connector = Connector(OSError("network down"))
    run(make_processor(connector, session=session), [SleepFrame(sleep=0.2)])

    assert len(connector.calls) == 2  # first attempt + exactly one retry
    assert session.failures == ["realtime_translate_failed"]
    assert session.end_requests == ["realtime_translate_failed"]


def test_session_rejected_by_the_api_is_a_failure() -> None:
    session = FakeSession()
    rejected = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad language"}}
    connector = Connector(FakeWS([rejected]), FakeWS([rejected]))
    run(make_processor(connector, session=session), [SleepFrame(sleep=0.2)])

    assert len(connector.calls) == 2
    assert session.failures == ["realtime_translate_failed"]


def test_error_event_after_the_session_is_ready_is_not_fatal() -> None:
    session = FakeSession()
    ws = FakeWS(
        [
            {"type": "session.created", "session": {"id": "s"}},
            {"type": "error", "error": {"type": "server_error", "message": "hiccup"}},
            audio_delta(b"\x05\x10" * 4),
        ]
    )
    connector = Connector(ws)
    down, _ = run(make_processor(connector, session=session), [SleepFrame(sleep=0.1)])

    assert len(connector.calls) == 1
    assert session.failures == []
    assert [f.audio[0] for f in played(down)] == [5]


def test_retry_allowance_is_restored_after_a_stable_connection() -> None:
    first = FakeWS([0.1, ConnectionError("blip 1")])
    second = FakeWS([0.1, ConnectionError("blip 2")])
    third = FakeWS()
    connector = Connector(first, second, third)
    session = FakeSession()
    run(
        make_processor(connector, session=session, stable_connection_seconds=0.05),
        [SleepFrame(sleep=0.5)],
    )
    assert len(connector.calls) == 3
    assert session.failures == []


# -- RuntimeSession terminal event ---------------------------------------------


def test_runtime_session_fail_turns_the_terminal_event_into_session_failed(monkeypatch) -> None:
    import pipecat.workers.runner
    import src.runtime.pipeline

    class Client:
        def __init__(self) -> None:
            self.events = []

        async def emit(self, event) -> None:
            self.events.append(event)

    class Transport:
        def add_event_handler(self, *_):
            pass

        async def cleanup(self):
            pass

    client = Client()
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), client)
    monkeypatch.setattr(
        src.runtime.pipeline, "build_livekit_pipeline", lambda *_: (Transport(), object())
    )

    class Runner:
        def __init__(self, **kwargs) -> None:
            pass

        async def add_workers(self, worker) -> None:
            pass

        async def run(self) -> None:
            session.fail("realtime_translate_failed")

    monkeypatch.setattr(pipecat.workers.runner, "WorkerRunner", Runner)
    config = PipelineConfig("ws://l", "t", "room", "d", "o", "c", "v", "prompt")
    asyncio.run(session.run_pipeline(config))

    last = client.events[-1]
    assert last.event_type == "SessionFailed"
    assert last.payload == {"outcome": "failed", "reason": "realtime_translate_failed"}


# -- config and pipeline builder -----------------------------------------------


@pytest.fixture
def light_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipecat.audio.vad import silero

    class LightweightAnalyzer:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(silero, "SileroVADAnalyzer", LightweightAnalyzer)


def _realtime_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(
        "ws://localhost:7880", "token", "room", "", "sk-openai", "", "",
        "prompt", mode="interpreter",
        interpreter={"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"},
        **kwargs,
    )  # fmt: skip


def test_builder_chooses_the_realtime_pipeline_without_stt_llm_or_tts(light_models) -> None:
    from pipecat.pipeline.pipeline import Pipeline

    _, worker = build_livekit_pipeline(
        _realtime_config(), RuntimeSession(uuid4(), uuid4(), uuid4())
    )
    pipeline = next(p for p in worker._pipeline._processors if isinstance(p, Pipeline))
    names = [
        type(p).__name__
        for p in pipeline._processors
        if type(p).__name__ not in {"PipelineSource", "PipelineSink"}
    ]
    assert names == [
        "LiveKitInputTransport",
        "VADProcessor",
        "RealtimeTranslateProcessor",
        "RuntimeEventProcessor",
        "LiveKitOutputTransport",
    ]


def test_realtime_engine_needs_only_livekit_and_openai_credentials(monkeypatch) -> None:
    for name in ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    session = {
        "room_name": "room",
        "livekit_token": "token",
        "mode": "interpreter",
        "interpreter": {"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"},
    }
    config = PipelineConfig.from_environment(session)
    assert config.interpreter["engine"] == "realtime"

    cascaded = {**session, "interpreter": {"languages": ["ru", "en"], "direction": "one_way"}}
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        PipelineConfig.from_environment(cascaded)


@pytest.mark.parametrize(
    "interpreter, message",
    [
        ({"languages": ["en", "ru"], "direction": "auto", "engine": "realtime"}, "one_way"),
        ({"languages": ["en", "nl"], "direction": "one_way", "engine": "realtime"}, "nl"),
        ({"languages": ["en", "ru"], "direction": "one_way", "engine": "magic"}, "engine"),
    ],
)
def test_invalid_realtime_interpreter_settings_fail_fast(monkeypatch, interpreter, message) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    with pytest.raises(RuntimeError, match=message):
        PipelineConfig.from_environment(
            {"room_name": "r", "livekit_token": "t", "mode": "interpreter", "interpreter": interpreter}
        )


def test_silent_translated_audio_plays_without_counting_as_speech():
    assert not is_voiced(b"\x00\x00" * 100)
    assert not is_voiced(b"\x10\x00" * 100)  # amplitude 16, below the threshold
    assert is_voiced(b"\x00\x00" * 99 + b"\x00\x10")  # one loud sample is enough

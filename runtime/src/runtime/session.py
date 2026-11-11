"""In-memory lifecycle state for one live conversation."""

import asyncio
import inspect
import logging
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from .apology import MAX_PLAYBACK_SECONDS, PLAYBACK_MARGIN_SECONDS, apology_plans, speaker_plans
from .events import RuntimeEvent, RuntimeEventType
from .lifecycle import EndReason, classify_end_reason, terminal_payload
from .participants import ParticipantDirectory
from .provider_failures import FALLBACK_REASON, cascaded_fallback_config

_EVENT_QUEUE_CAPACITY = 128
_EVENT_DRAIN_TIMEOUT_SECONDS = 5
_EVENT_WORKER_STOP_TIMEOUT_SECONDS = 1
_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .events_client import PlatformEventClient


class SessionState(str, Enum):
    CREATED = "created"
    CONNECTED = "connected"
    LISTENING = "listening"
    SPEAKING = "speaking"
    ENDED = "ended"


class RuntimeSession:
    def __init__(
        self,
        session_id: UUID,
        conversation_id: UUID,
        agent_id: UUID,
        event_client: "PlatformEventClient | None" = None,
    ) -> None:
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.agent_id = agent_id
        self.state = SessionState.CREATED
        self.cancel_requested = False
        self.output_queue: list[object] = []
        self.event_client = event_client
        # Set for two-person interpreter sessions (shared with the HTTP layer).
        self.participants: ParticipantDirectory | None = None
        self._pipeline_worker = None
        # ST-13: per-session length cap (contract ``max_duration_secs``; None -> env default)
        # and the monotonic start stamp the lifecycle monitor sets once per session.
        self.max_duration_secs: float | None = None
        self.lifecycle_started_at: float | None = None
        self._lifecycle_monitor = None
        # ST-10b: what the failure apology needs. ``_transport``/``_pipeline_config`` belong
        # to the running pipeline; ``caller_language`` is set by the agent translator.
        self._transport: object | None = None
        self._pipeline_config: object | None = None
        self.caller_language: Callable[[], str | None] | None = None
        self._pipeline_cancel_task: asyncio.Task[bool] | None = None
        self._pipeline_cancel_requested = False
        self._requested_end_reason: str | None = None
        self._failure_reason: str | None = None
        # PR-30: the cascaded config a failed realtime engine falls back to (once).
        self._fallback_config: object | None = None
        self._next_config: object | None = None
        # A fallback restart is scheduled (set by request_engine_fallback, cleared by the
        # restart). A real end request arriving meanwhile cancels it: see request_end.
        self._restart_pending = False
        self._restart_cancelled = False
        self._event_queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue(
            maxsize=_EVENT_QUEUE_CAPACITY
        )
        self._event_delivery_task: asyncio.Task[None] | None = None
        self._event_delivery_error: Exception | None = None
        self._accepting_events = True
        self._event_shutdown_expired = False
        self._close_event_client_when_worker_stops = False

    @property
    def event_delivery_error(self) -> Exception | None:
        return self._event_delivery_error

    def defer_event_client_close(self) -> bool:
        """Close the HTTP client from the dispatcher if it outlives shutdown."""

        task = self._event_delivery_task
        if task is None or task.done():
            return False
        self._close_event_client_when_worker_stops = True
        return True

    def connect(self) -> None:
        self.state = SessionState.CONNECTED

    def begin_listening(self) -> None:
        self.state = SessionState.LISTENING

    def interrupt(self) -> None:
        self.cancel_requested = True
        self.output_queue.clear()
        if self.state == SessionState.SPEAKING:
            self.state = SessionState.LISTENING

    def end(self) -> None:
        self.state = SessionState.ENDED

    def fail(self, reason: str) -> None:
        """Mark the session failed: the terminal event becomes ``SessionFailed``."""

        self._failure_reason = self._failure_reason or reason

    @property
    def ending(self) -> bool:
        """True once the pipeline was asked to stop (late provider errors are then noise)."""

        return self._pipeline_cancel_requested or self.state == SessionState.ENDED

    async def play_apology(
        self, timeout: float = MAX_PLAYBACK_SECONDS, *, speaker_identity: str | None = None
    ) -> bool:
        """Play the pre-rendered apology to the caller(s) and wait for it (ST-10b).

        The clip is written straight to the transport's output processor
        (``transport.output().queue_frame``): no TTS, no pipeline, so it works when the TTS
        or any processor is the thing that broke. Frames carry the listener's
        ``transport_destination`` in two-person sessions, so each listener hears their own
        language on their own track. Returns True when the clip was queued and its duration
        elapsed; False when nothing could be played. Never raises and never takes longer
        than ``timeout`` (queueing and playback together). With ``speaker_identity`` only that
        person hears it (a lost utterance, the call goes on).
        """

        transport, config = self._transport, self._pipeline_config
        output_of = getattr(transport, "output", None)
        if output_of is None or config is None or self._pipeline_worker is None or self.ending:
            return False
        try:
            plans = (
                apology_plans(config, self.participants, self.caller_language)
                if speaker_identity is None
                else speaker_plans(config, self.participants, speaker_identity)
            )
            if not plans:
                return False
            from .apology import to_frames

            output = output_of()
            frames = to_frames(plans)
        except Exception:
            _logger.warning("Could not prepare the failure apology", exc_info=True)
            return False
        wait = max(plan.clip.duration for plan in plans) + PLAYBACK_MARGIN_SECONDS

        async def play() -> None:
            for frame in frames:
                await output.queue_frame(frame)
            deadline = asyncio.get_running_loop().time() + wait
            # Stop early if the session is torn down meanwhile (the audio is gone anyway).
            while not self.ending and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(play(), timeout=timeout)
        except asyncio.TimeoutError:
            _logger.warning("Failure apology did not finish within %.1fs", timeout)
            return False
        except Exception:
            _logger.warning("Failure apology playback failed", exc_info=True)
            return False
        return True

    def request_engine_fallback(self) -> bool:
        """Restart the pipeline on the cascaded engine (PR-30); False when not feasible.

        ``run_pipeline`` rebuilds the pipeline from the cascaded config once the current
        one has stopped, so the room, session and event stream carry on.
        """

        config, self._fallback_config = self._fallback_config, None
        if (
            config is None
            or self._pipeline_worker is None
            or self._pipeline_cancel_requested
            or self._pipeline_cancel_task is not None
        ):
            return False
        self._next_config = config
        self._restart_pending = True
        self._pipeline_cancel_task = asyncio.create_task(self.request_end(reason=FALLBACK_REASON))
        return True

    async def request_end(self, reason: str = "Session ended by request") -> bool:
        """Ask Pipecat to stop the pipeline through its native cancellation frame."""

        worker = self._pipeline_worker
        if worker is None:
            return False
        if self._restart_pending and reason != FALLBACK_REASON:
            # An /end, hang-up or other real stop arrived while the engine fallback was
            # pending: the session ends with this reason instead of restarting.
            self._restart_cancelled = True
            self._next_config = None
            self._requested_end_reason = reason
            self.cancel_requested = True
        if self._pipeline_cancel_requested:
            return True
        self._pipeline_cancel_requested = True
        self.cancel_requested = True
        self._requested_end_reason = reason
        try:
            cancellation = worker.cancel(reason=reason)
            if inspect.isawaitable(cancellation):
                await cancellation
        except Exception:
            self._pipeline_cancel_requested = False
            self._requested_end_reason = None
            raise
        return True

    def event(
        self,
        event_type: RuntimeEventType,
        payload: dict[str, object] | None = None,
    ) -> RuntimeEvent:
        return RuntimeEvent(
            event_type=event_type,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            agent_id=self.agent_id,
            payload=payload or {},
        )

    def enqueue_event(self, event: RuntimeEvent) -> None:
        """Queue a Platform event without waiting for network delivery."""

        if not self.event_client:
            return
        if not self._accepting_events:
            raise RuntimeError("runtime event delivery is no longer accepting events")
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull as exc:
            self._event_delivery_error = RuntimeError("runtime event queue capacity exceeded")
            self._accepting_events = False
            if self._pipeline_worker:
                self._pipeline_cancel_task = asyncio.create_task(
                    self.request_end(reason="Runtime event queue capacity exceeded")
                )
            raise self._event_delivery_error from exc
        if self._event_delivery_task is None:
            self._event_delivery_task = asyncio.create_task(self._deliver_events())

    async def close_event_dispatcher(self) -> bool:
        """Drain queued events and stop the per-session delivery task."""

        self._accepting_events = False
        if self._pipeline_cancel_task is not None:
            await asyncio.gather(self._pipeline_cancel_task, return_exceptions=True)
            self._pipeline_cancel_task = None
        task = self._event_delivery_task
        if task is None:
            return True
        if self._event_shutdown_expired and not task.done():
            task.cancel()
            return False
        drain_task = asyncio.create_task(self._event_queue.join())
        worker_stopped = False
        try:
            completed, _ = await asyncio.wait(
                {drain_task}, timeout=_EVENT_DRAIN_TIMEOUT_SECONDS
            )
            if completed:
                await drain_task
            else:
                if self._event_delivery_error is None:
                    self._event_delivery_error = RuntimeError(
                        "runtime event delivery drain timed out"
                    )
                drain_task.cancel()
                await asyncio.gather(drain_task, return_exceptions=True)
                task.cancel()
        finally:
            if not task.done():
                task.cancel()
            _, pending = await asyncio.wait(
                {task}, timeout=_EVENT_WORKER_STOP_TIMEOUT_SECONDS
            )
            if pending:
                if self._event_delivery_error is None:
                    self._event_delivery_error = RuntimeError(
                        "runtime event worker did not stop before shutdown deadline"
                    )
                _logger.error(
                    "Platform event worker did not stop before shutdown deadline for session %s",
                    self.session_id,
                )
            discarded = self._discard_pending_events()
            if discarded:
                _logger.error(
                    "Discarded %s queued Platform events during shutdown for session %s",
                    discarded,
                    self.session_id,
                )
            if not pending:
                self._event_delivery_task = None
                worker_stopped = True
            else:
                self._event_shutdown_expired = True
        return worker_stopped

    async def _deliver_events(self) -> None:
        try:
            while True:
                event = await self._event_queue.get()
                try:
                    if self._event_delivery_error is not None:
                        discarded = self._discard_pending_events()
                        if discarded:
                            _logger.error(
                                "Discarded %s queued Platform events after delivery failure for session %s",
                                discarded,
                                self.session_id,
                            )
                        return
                    assert self.event_client is not None
                    await self.event_client.emit(event)
                    if self._event_shutdown_expired:
                        return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._event_delivery_error = exc
                    self._accepting_events = False
                    _logger.exception(
                        "Platform event delivery failed for session %s", self.session_id
                    )
                    if self._pipeline_worker:
                        try:
                            await self.request_end(reason="Platform event delivery failed")
                        except Exception:
                            _logger.exception(
                                "Failed to cancel pipeline after Platform event delivery failure "
                                "for session %s",
                                self.session_id,
                            )
                    discarded = self._discard_pending_events()
                    if discarded:
                        _logger.error(
                            "Discarded %s queued Platform events after delivery failure for session %s",
                            discarded,
                            self.session_id,
                        )
                    return
                finally:
                    self._event_queue.task_done()
        finally:
            if self._close_event_client_when_worker_stops and self.event_client:
                try:
                    await self.event_client.aclose()
                except Exception:
                    _logger.exception(
                        "Failed to close Platform event client for session %s", self.session_id
                    )

    def _discard_pending_events(self) -> int:
        discarded = 0
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._event_queue.task_done()
            discarded += 1
        return discarded

    async def _reset_for_restart(self) -> None:
        """Clear the stop request of the pipeline that just ended before building the next."""

        task, self._pipeline_cancel_task = self._pipeline_cancel_task, None
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if self._lifecycle_monitor is not None:
            self._lifecycle_monitor.stop()
            self._lifecycle_monitor = None
        self._pipeline_worker = None
        self._transport = None
        self._restart_pending = False
        if self._restart_cancelled:
            # Ended while the restart was being prepared: keep the end reason, do not rebuild.
            return
        self._pipeline_cancel_requested = False
        self._requested_end_reason = None
        self.cancel_requested = False
        self.state = SessionState.CREATED

    async def run_pipeline(self, config: object) -> None:
        """Run the configured Pipecat pipeline and finalize lifecycle state."""

        event_type = "SessionEnded"
        outcome = "ended"
        reason: str | EndReason | None = None
        detail: str | None = None
        transport = None
        try:
            from pipecat.workers.runner import WorkerRunner

            from .lifecycle import attach_lifecycle
            from .pipeline import PipelineConfig, build_livekit_pipeline

            if not isinstance(config, PipelineConfig):
                raise TypeError("config must be a PipelineConfig")
            started = False
            self._fallback_config = cascaded_fallback_config(config)
            while True:
                transport, worker = build_livekit_pipeline(config, self)
                self._pipeline_worker = worker
                self._transport = transport
                self._pipeline_config = config
                # ST-13: idle prompt / idle end / max duration (an observer on the worker).
                self._lifecycle_monitor = attach_lifecycle(worker, self, config)

                async def on_connected(*_: object, **__: object) -> None:
                    nonlocal started
                    self.connect()
                    self.begin_listening()
                    if self.event_client and not started:
                        started = True  # a fallback restart is not a new RuntimeStarted
                        self.enqueue_event(
                            self.event("RuntimeStarted", {"room_name": config.room_name})
                        )

                async def on_disconnected(*_: object, **__: object) -> None:
                    if self.state != SessionState.ENDED:
                        await self.request_end("LiveKit disconnected")

                present: set[str] = set()

                async def on_participant_present(
                    _transport: object, participant_id: str
                ) -> None:
                    present.add(participant_id)

                async def on_participant_disconnected(
                    _transport: object, participant_id: str
                ) -> None:
                    present.discard(participant_id)
                    if self.participants is not None and present:
                        # Two-person call: the remaining participant keeps the session.
                        _logger.info("LiveKit participant left; session continues")
                        return
                    if self.state != SessionState.ENDED:
                        _logger.info("LiveKit participant disconnected; ending voice session")
                        await self.request_end("participant_disconnected")

                transport.add_event_handler("on_connected", on_connected)
                transport.add_event_handler("on_disconnected", on_disconnected)
                transport.add_event_handler(
                    "on_participant_disconnected", on_participant_disconnected
                )
                if self.participants is not None:
                    # Existing participants are reported by their audio track subscription.
                    transport.add_event_handler(
                        "on_participant_connected", on_participant_present
                    )
                    transport.add_event_handler(
                        "on_audio_track_subscribed", on_participant_present
                    )
                runner = WorkerRunner(handle_sigint=False)
                await runner.add_workers(worker)
                await runner.run()
                await transport.cleanup()
                transport = None
                if self._next_config is None or self._restart_cancelled:
                    break
                # PR-30: the realtime engine failed; carry on with the cascaded pipeline.
                config, self._next_config = self._next_config, None
                await self._reset_for_restart()
                if self._restart_cancelled:
                    break  # ended while the restart was being prepared
            dispatcher_stopped = await self.close_event_dispatcher()
            if not dispatcher_stopped:
                raise RuntimeError("Platform event worker did not stop during session shutdown")
            if self._event_delivery_error:
                raise RuntimeError("failed to deliver Runtime events") from (
                    self._event_delivery_error
                )
        except asyncio.CancelledError:
            if self._event_delivery_error:
                event_type = "SessionFailed"
                outcome = "failed"
                reason = EndReason.RUNTIME_ERROR
                detail = "event_delivery_failed"
                raise RuntimeError("Platform event delivery failed") from self._event_delivery_error
            outcome = "cancelled"
            reason = EndReason.CANCELLED
            raise
        except Exception as exc:
            event_type = "SessionFailed"
            outcome = "failed"
            reason = EndReason.RUNTIME_ERROR
            detail = type(exc).__name__
            raise
        finally:
            try:
                if transport is not None:
                    await transport.cleanup()
            finally:
                if self._lifecycle_monitor is not None:
                    self._lifecycle_monitor.stop()
                self._pipeline_worker = None
                self._transport = None
                self.end()
                if self.event_client:
                    dispatcher_stopped = await self.close_event_dispatcher()
                    if self._failure_reason and event_type == "SessionEnded":
                        event_type = "SessionFailed"
                        outcome = "failed"
                        reason, detail = classify_end_reason(self._failure_reason)
                    if self._event_delivery_error:
                        event_type = "SessionFailed"
                        outcome = "failed"
                        reason = EndReason.RUNTIME_ERROR
                        detail = "event_delivery_failed"
                    payload = terminal_payload(
                        outcome, reason or self._requested_end_reason, detail=detail
                    )
                    if dispatcher_stopped:
                        await self.event_client.emit(self.event(event_type, payload))
                    else:
                        _logger.error(
                            "Skipping terminal event because the Platform event worker is still active for session %s",
                            self.session_id,
                        )

"""In-memory lifecycle state for one live conversation."""

import asyncio
import inspect
import logging
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

from .events import RuntimeEvent, RuntimeEventType
from .participants import ParticipantDirectory

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
        self._pipeline_cancel_task: asyncio.Task[bool] | None = None
        self._pipeline_cancel_requested = False
        self._requested_end_reason: str | None = None
        self._failure_reason: str | None = None
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

    async def request_end(self, reason: str = "Session ended by request") -> bool:
        """Ask Pipecat to stop the pipeline through its native cancellation frame."""

        worker = self._pipeline_worker
        if worker is None:
            return False
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

    async def run_pipeline(self, config: object) -> None:
        """Run the configured Pipecat pipeline and finalize lifecycle state."""

        event_type = "SessionEnded"
        outcome = "ended"
        reason: str | None = None
        transport = None
        try:
            from pipecat.workers.runner import WorkerRunner

            from .pipeline import PipelineConfig, build_livekit_pipeline

            if not isinstance(config, PipelineConfig):
                raise TypeError("config must be a PipelineConfig")
            transport, worker = build_livekit_pipeline(config, self)
            self._pipeline_worker = worker

            async def on_connected(*_: object, **__: object) -> None:
                self.connect()
                self.begin_listening()
                if self.event_client:
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
                reason = "event_delivery_failed"
                raise RuntimeError("Platform event delivery failed") from self._event_delivery_error
            outcome = "cancelled"
            reason = "cancelled"
            raise
        except Exception as exc:
            event_type = "SessionFailed"
            outcome = "failed"
            reason = type(exc).__name__
            raise
        finally:
            try:
                if transport is not None:
                    await transport.cleanup()
            finally:
                self._pipeline_worker = None
                self.end()
                if self.event_client:
                    dispatcher_stopped = await self.close_event_dispatcher()
                    if self._failure_reason and event_type == "SessionEnded":
                        event_type = "SessionFailed"
                        outcome = "failed"
                        reason = self._failure_reason
                    if self._event_delivery_error:
                        event_type = "SessionFailed"
                        outcome = "failed"
                        reason = "event_delivery_failed"
                    payload: dict[str, object] = {"outcome": outcome}
                    terminal_reason = reason or self._requested_end_reason
                    if terminal_reason:
                        payload["reason"] = terminal_reason
                    if dispatcher_stopped:
                        await self.event_client.emit(self.event(event_type, payload))
                    else:
                        _logger.error(
                            "Skipping terminal event because the Platform event worker is still active for session %s",
                            self.session_id,
                        )

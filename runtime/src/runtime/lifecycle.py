"""Session lifecycle policy (ST-13): stable end reasons, idle timeout, max duration, admission.

* :class:`EndReason` is the closed set of values the ``reason`` of a terminal
  ``SessionEnded`` / ``SessionFailed`` event can take. Free text (exception class names,
  Pipecat cancel messages) never reaches the event: :func:`terminal_payload` maps it and
  keeps the original in a separate ``detail`` field when that is useful.
* :class:`LifecycleObserver` is a Pipecat observer (no pipeline processor, like the stall
  watchdog) that feeds :class:`LifecycleMonitor` with speech activity. The monitor prompts
  an idle agent ("Are you still there?"), ends an idle session and ends a session that
  reached its maximum length. Both are ended through ``RuntimeSession.request_end`` so the
  normal terminal event path reports ``idle_timeout`` / ``max_duration``.
* :func:`max_sessions` / :func:`max_session_secs` read the environment knobs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from functools import lru_cache
from typing import Any

from .fallback_lines import FALLBACK_LANGUAGE, IDLE, line

_logger = logging.getLogger(__name__)

DEFAULT_IDLE_PROMPT_SECS = 60.0
DEFAULT_IDLE_END_SECS = 90.0
DEFAULT_MAX_SESSION_SECS = 1800.0
DEFAULT_MAX_SESSIONS = 20
TICK_SECS = 1.0


class EndReason(str, Enum):
    """Why a session ended; the only values a terminal event's ``reason`` can carry."""

    USER_HANGUP = "user_hangup"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    IDLE_TIMEOUT = "idle_timeout"
    MAX_DURATION = "max_duration"
    PROVIDER_FAILED = "provider_failed"
    RUNTIME_ERROR = "runtime_error"
    CAPACITY = "capacity"
    PLATFORM_END = "platform_end"
    CANCELLED = "cancelled"


END_REASONS = frozenset(reason.value for reason in EndReason)

# Free-text end requests (Pipecat cancel messages) -> stable reason.
_KNOWN_TEXT: dict[str, EndReason] = {
    "LiveKit disconnected": EndReason.PARTICIPANT_DISCONNECTED,
    "Browser participant disconnected": EndReason.PARTICIPANT_DISCONNECTED,
    "Session ended by request": EndReason.PLATFORM_END,
}
# Internal reasons that are really a runtime failure; the original text becomes the detail.
_RUNTIME_ERROR_DETAIL: dict[str, str] = {
    "event_delivery_failed": "event_delivery_failed",
    "Platform event delivery failed": "event_delivery_failed",
    "Runtime event queue capacity exceeded": "event_queue_capacity_exceeded",
}


def classify_end_reason(raw: str | EndReason | None) -> tuple[EndReason, str | None]:
    """Map any internal end reason to ``(EndReason, detail)``.

    ``None`` (the pipeline stopped by itself, nobody asked) is ``participant_disconnected``:
    the media transport closing is the only way that happens. Unknown text is a
    ``runtime_error`` whose ``detail`` is the text.
    """

    if raw is None:
        return EndReason.PARTICIPANT_DISCONNECTED, None
    if isinstance(raw, EndReason):
        return raw, None
    if raw in END_REASONS:
        return EndReason(raw), None
    if raw in _KNOWN_TEXT:
        return _KNOWN_TEXT[raw], None
    return EndReason.RUNTIME_ERROR, _RUNTIME_ERROR_DETAIL.get(raw, raw)


def terminal_payload(
    outcome: str,
    reason: str | EndReason | None,
    *,
    detail: str | None = None,
    classify: bool = True,
) -> dict[str, object]:
    """The payload of a terminal ``SessionEnded`` / ``SessionFailed`` event."""

    end_reason, mapped_detail = classify_end_reason(reason) if classify else (reason, None)
    payload: dict[str, object] = {"outcome": outcome, "reason": end_reason.value}
    detail = detail or mapped_detail
    if detail:
        payload["detail"] = detail
    return payload


# -- environment knobs ------------------------------------------------------------------


def _env_number(name: str, default: float, *, minimum: float = 1.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def idle_prompt_secs() -> float:
    return _env_number("RUNTIME_IDLE_PROMPT_SECS", DEFAULT_IDLE_PROMPT_SECS)


def idle_end_secs() -> float:
    return _env_number("RUNTIME_IDLE_END_SECS", DEFAULT_IDLE_END_SECS)


def max_session_secs() -> float:
    return _env_number("RUNTIME_MAX_SESSION_SECS", DEFAULT_MAX_SESSION_SECS)


def max_sessions() -> int:
    return int(_env_number("RUNTIME_MAX_SESSIONS", DEFAULT_MAX_SESSIONS))


# -- monitor ----------------------------------------------------------------------------


class LifecycleMonitor:
    """Idle prompt / idle end / max duration for one session.

    ``check()`` evaluates the policy once at ``clock()``; ``run()`` calls it every tick.
    Tests drive ``check()`` with a fake clock.
    """

    def __init__(
        self,
        session: Any,
        *,
        speak: Callable[[str], Awaitable[Any]] | None,
        language: Callable[[], str | None] | None = None,
        interpreter: bool = False,
        prompt_secs: float | None = None,
        end_secs: float | None = None,
        max_secs: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._session = session
        self._speak = speak
        self._language = language
        self._interpreter = interpreter
        self._prompt_secs = idle_prompt_secs() if prompt_secs is None else prompt_secs
        self._end_secs = idle_end_secs() if end_secs is None else end_secs
        self._clock = clock
        self._sleep = sleep
        if max_secs is None:
            override = getattr(session, "max_duration_secs", None)
            max_secs = override if _positive(override) else max_session_secs()
        self._max_secs = float(max_secs)
        # The session length spans an engine-fallback restart: the first monitor stamps it.
        started = getattr(session, "lifecycle_started_at", None)
        if started is None:
            started = clock()
            _set(session, "lifecycle_started_at", started)
        self._started = started
        self._last_activity = clock()
        self._prompted = False
        self._user_speaking = False
        self._bot_speaking = False
        self._ended = False
        self._stopped = False
        self._task: asyncio.Task[None] | None = None

    @property
    def prompted(self) -> bool:
        return self._prompted

    # -- activity ---------------------------------------------------------------------

    def user_started(self) -> None:
        self._user_speaking = True
        self._last_activity = self._clock()
        self._prompted = False  # the caller answered: the next silence starts over

    def user_stopped(self) -> None:
        self._user_speaking = False
        self._last_activity = self._clock()

    def bot_started(self) -> None:
        self._bot_speaking = True
        self._bot_activity()

    def bot_stopped(self) -> None:
        self._bot_speaking = False
        self._bot_activity()

    def bot_speaking(self) -> None:
        self._bot_activity()

    def _bot_activity(self) -> None:
        # Once the "still there?" prompt went out, the bot's own voice is not activity.
        if not self._prompted:
            self._last_activity = self._clock()

    # -- policy -----------------------------------------------------------------------

    async def check(self) -> None:
        if self._ended:
            return
        now = self._clock()
        if now - self._started >= self._max_secs:
            await self._end(EndReason.MAX_DURATION)
            return
        if self._user_speaking or (self._bot_speaking and not self._prompted):
            self._last_activity = now
            return
        idle = now - self._last_activity
        if idle >= self._end_secs:
            await self._end(EndReason.IDLE_TIMEOUT)
        elif (
            idle >= self._prompt_secs
            and not self._prompted
            and not self._interpreter
            and self._speak is not None
        ):
            self._prompted = True
            try:
                await self._speak(line(IDLE, self._resolve_language()))
            except Exception:
                _logger.warning("Could not queue the idle prompt", exc_info=True)

    def _resolve_language(self) -> str:
        try:
            return (self._language() if self._language else None) or FALLBACK_LANGUAGE
        except Exception:
            return FALLBACK_LANGUAGE

    async def _end(self, reason: EndReason) -> None:
        self._ended = True
        _logger.info("Ending session %s: %s", getattr(self._session, "session_id", "?"), reason.value)
        try:
            await self._session.request_end(reason=reason.value)
        except Exception:
            _logger.warning("Could not end the session on %s", reason.value, exc_info=True)

    # -- loop -------------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None and not self._stopped:
            self._task = asyncio.get_running_loop().create_task(
                self.run(), name="session-lifecycle"
            )

    def stop(self) -> None:
        self._stopped = True
        task, self._task = self._task, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def run(self) -> None:
        while not self._ended:
            await self._sleep(TICK_SECS)
            await self.check()


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _set(obj: object, name: str, value: object) -> None:
    try:
        setattr(obj, name, value)
    except Exception:  # pragma: no cover - exotic session doubles
        pass


# -- observer ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def observer_class() -> type:
    """The :class:`LifecycleObserver` class (built lazily: Pipecat is an optional extra)."""

    from pipecat.frames.frames import (
        BotSpeakingFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        CancelFrame,
        EndFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.observers.base_observer import BaseObserver, FramePushed

    class LifecycleObserver(BaseObserver):
        """Feed speech activity to a :class:`LifecycleMonitor`; start it with the first frame."""

        def __init__(self, monitor: LifecycleMonitor) -> None:
            super().__init__()
            self.monitor = monitor

        async def on_push_frame(self, data: FramePushed) -> None:
            frame = data.frame
            self.monitor.start()  # needs a running loop; the StartFrame is the first frame
            if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
                self.monitor.user_started()
            elif isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)):
                self.monitor.user_stopped()
            elif isinstance(frame, BotStartedSpeakingFrame):
                self.monitor.bot_started()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self.monitor.bot_stopped()
            elif isinstance(frame, BotSpeakingFrame):
                self.monitor.bot_speaking()
            elif isinstance(frame, (CancelFrame, EndFrame)):
                self.monitor.stop()

        async def cleanup(self) -> None:
            self.monitor.stop()
            await super().cleanup()

    return LifecycleObserver


def idle_language(config: object, session: object) -> Callable[[], str | None]:
    """The language the idle prompt is spoken in: caller language, else TTS/STT setting."""

    def resolve() -> str | None:
        caller = getattr(session, "caller_language", None)
        if callable(caller):
            try:
                language = caller()
            except Exception:
                language = None
            if isinstance(language, str) and language and language != "multi":
                return language
        settings = getattr(config, "provider_settings", None) or {}
        for kind in ("tts", "stt"):
            language = (settings.get(kind) or {}).get("language")
            if isinstance(language, str) and language and language != "multi":
                return language
        return None

    return resolve


def attach_lifecycle(
    worker: Any,
    session: Any,
    config: object = None,
    *,
    interpreter: bool | None = None,
    **monitor_kwargs: Any,
) -> LifecycleMonitor | None:
    """Attach the lifecycle observer to a pipeline worker; no-op without session/observers."""

    if session is None or not hasattr(worker, "add_observer"):
        return None
    if interpreter is None:
        interpreter = getattr(config, "mode", "agent") == "interpreter"

    async def speak(text: str) -> None:
        from pipecat.frames.frames import TTSSpeakFrame

        await worker.queue_frame(TTSSpeakFrame(text, append_to_context=False))

    monitor = LifecycleMonitor(
        session,
        speak=speak,
        language=idle_language(config, session),
        interpreter=interpreter,
        **monitor_kwargs,
    )
    worker.add_observer(observer_class()(monitor))
    return monitor

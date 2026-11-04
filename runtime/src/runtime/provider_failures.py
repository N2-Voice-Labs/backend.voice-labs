"""Provider failures (PR-30, ST-10a): never go silent when a provider breaks.

A rejected key (401/403), exhausted credit (402, ``quota_exceeded``, ``insufficient_quota``),
an outage, a hang or any other error of an STT, LLM, TTS, translator or realtime provider
used to leave the session up but mute: Pipecat marks the service "no longer usable" and
later utterances are skipped with a log line only. This module is the one path every such
error takes. Nothing is log-only: **every** report emits a ``ProviderError`` event.

``ProviderError`` v1 payload (the error taxonomy)::

    {provider, kind, category, stage, retryable, attempt, message}

* ``kind``: ``stt|llm|tts|realtime|translation``.
* ``category``: ``auth|quota|unavailable|timeout|other``. ``timeout`` is a request that
  timed out or a stall found by :mod:`runtime.stall_watchdog`; ``unavailable`` is 5xx/429,
  a refused/reset connection or a dropped socket.
* ``stage``: where it broke, e.g. ``transcription``, ``generation``, ``synthesis``,
  ``translation``, ``connect``, ``stream`` or a watchdog stage (``no_final_transcript``,
  ``no_first_token``, ``no_first_audio``).
* ``retryable``: True when the call goes on and the provider gets another chance.
* ``attempt``: 1-based count of errors of this kind in the current
  :data:`TRANSIENT_WINDOW_SECONDS` window.
* ``message``: a fixed phrase, never provider error text, so no key or body leaks.

Policy:

1. ``auth``/``quota`` (and anything flagged fatal, e.g. a processor Pipecat marked
   unusable): terminal at once, ``retryable`` False.
2. ``unavailable``/``timeout`` (transient): the first one in the window is *retried*
   (``retryable`` True, the call goes on); the second one in the window is terminal.
3. ``other``: emitted and the call goes on; :data:`OTHER_FAILURE_LIMIT` in the window is
   terminal (an unclassified error says little about provider health on its own).
4. Terminal: realtime engine, category not auth/quota, one-way, cascaded providers
   configured: the session restarts its pipeline on the cascaded engine
   (``EngineFallback`` ``{from, to}``). Otherwise ``fail("provider_failed")`` then the
   normal session-ended path (``SessionFailed`` with ``reason: provider_failed``), so the
   room is closed and the worker freed. Auth/quota never fall back: the cascaded
   translator uses the same OpenAI key.
5. ST-10b: before ending, the caller hears a pre-rendered apology in their language
   (:mod:`runtime.apology`, played through ``session.play_apology``, bounded to a few
   seconds, skipped when the transport itself failed). Not on a retry, a noted error or an
   engine fallback: the call goes on there.

What "retry" really means: Pipecat STT/LLM/TTS services and the OpenAI client reconnect and
retry at the SDK level themselves, and a failed utterance is not re-issued at pipeline
level; for those the retry is "tolerate once, fail on the second", with the next utterance
being the retry. True request-level retries exist where this repository owns the call: the
inbound translator re-issues its request once, and the realtime engine reconnects once
(:func:`report_provider_retry`).

Wiring: :func:`watch_provider_failures` hooks a ``PipelineWorker`` (STT/LLM/TTS errors
arrive as upstream ``ErrorFrame``s) and attaches the stall watchdog; the translators and the
realtime processor call :func:`report_provider_exception` / :func:`report_provider_failure`
for errors they catch.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import PipelineConfig

_logger = logging.getLogger(__name__)

FAILURE_REASON = "provider_failed"
FALLBACK_REASON = "engine_fallback"
TRANSIENT_WINDOW_SECONDS = 30.0
OTHER_FAILURE_LIMIT = 3  # unclassified errors in the window that end the call

FATAL_REASONS = frozenset({"auth", "quota"})
TRANSIENT_REASONS = frozenset({"unavailable", "timeout"})
CATEGORIES = ("auth", "quota", "unavailable", "timeout", "other")
KINDS = ("stt", "llm", "tts", "realtime", "translation")
_PHRASES = {
    "auth": "rejected the credentials",
    "quota": "is out of credit or quota",
    "unavailable": "is unavailable",
    "timeout": "timed out",
    "other": "failed",
}
DEFAULT_STAGES = {
    "stt": "transcription",
    "llm": "generation",
    "tts": "synthesis",
    "realtime": "stream",
    "translation": "translation",
}

_AUTH_TEXT = re.compile(
    r"invalid[_ ]api[_ ]key|incorrect api key|unauthori[sz]ed|authentication|forbidden"
    r"|permission[_ ]denied|\b(?:401|403)\b",
    re.IGNORECASE,
)
_QUOTA_TEXT = re.compile(
    r"insufficient[_ ]quota|quota[_ ]exceeded|exceeded your current quota"
    r"|payment required|out of credit|insufficient[_ ](?:credit|funds)|billing|\b402\b",
    re.IGNORECASE,
)
# A 403/404 about the *model* (a tenant picked one their project cannot use) is a
# configuration error of that tenant, not a rejected provider key.
_MODEL_ACCESS_TEXT = re.compile(
    r"(?:does not|do not|don't|doesn't) have access to (?:the )?(?:model|[`'\"]?gpt)"
    r"|model[_ ]not[_ ](?:found|available)"
    r"|the model\b.{0,80}\b(?:does not exist|do(?:es)? not have access|not found)"
    r"|model\b.{0,40}\bis not supported",
    re.IGNORECASE,
)
_TIMEOUT_TEXT = re.compile(r"timed? ?out|timeout", re.IGNORECASE)
_UNAVAILABLE_TEXT = re.compile(
    r"unavailable|overloaded|rate[_ ]limit|connection (?:closed|reset|refused|error)"
    r"|server[_ ]error|bad gateway|\b(?:429|500|502|503|504)\b",
    re.IGNORECASE,
)
# OpenAI closes the realtime socket with 3000 invalid_api_key (private-use WebSocket range).
_AUTH_CLOSE_CODES = frozenset({3000, 3003})


def _status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    for candidate in (
        getattr(response, "status_code", None),
        getattr(response, "status", None),
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _close_code(exc: BaseException) -> int | None:
    # websockets' ConnectionClosed keeps the received close frame in ``rcvd``.
    holder = getattr(exc, "rcvd", None) if hasattr(exc, "rcvd") else exc
    code = getattr(holder, "code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _is_timeout(exc: BaseException) -> bool:
    # asyncio/builtin TimeoutError, httpx.*Timeout, openai.APITimeoutError, websockets timeouts.
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


def _reason_from_status(status: int) -> str | None:
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "quota"
    if status == 408:
        return "timeout"
    if status in (425, 429) or 500 <= status < 600:
        return "unavailable"
    return None


def classify_error(
    *, exception: BaseException | None = None, text: str = "", category: Any = None
) -> str:
    """``auth|quota|unavailable|timeout|other`` for a provider error (status, code, category, text)."""

    if category is not None:
        from pipecat.utils.errors import ErrorCategory

        if category in (ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORIZATION):
            return "auth"
        if category is ErrorCategory.QUOTA:
            return "quota"
    if exception is not None:
        status = _status_code(exception)
        # Bodies such as OpenAI's 429 insufficient_quota are quota, not a rate limit.
        code = str(getattr(exception, "code", "") or "")
        if status in (None, 403, 404) and _MODEL_ACCESS_TEXT.search(f"{code} {exception}"):
            return "other"
        if _QUOTA_TEXT.search(code):
            return "quota"
        if _AUTH_TEXT.search(code):
            return "auth"
        if status is not None and (reason := _reason_from_status(status)):
            if reason == "unavailable" and _QUOTA_TEXT.search(str(exception)):
                return "quota"
            return reason
        close = _close_code(exception)
        if close in _AUTH_CLOSE_CODES:
            return "auth"
        if _is_timeout(exception):
            return "timeout"
        text = f"{text} {type(exception).__name__} {exception}"
        if type(exception).__name__.startswith("ConnectionClosed") and _AUTH_TEXT.search(
            str(exception)
        ):
            # ElevenLabs accepts the WebSocket handshake and then closes with
            # 1008 "Invalid API key" (observed against the real API, 2026-11-05).
            return "auth"
        if isinstance(exception, OSError) or type(
            exception
        ).__name__.startswith("ConnectionClosed"):
            return "unavailable"  # timeouts, refused/reset connections, dropped sockets
    if category is not None and category in (
        ErrorCategory.CONNECTIVITY,
        ErrorCategory.SERVER,
        ErrorCategory.RATE_LIMIT,
    ):
        return "unavailable"
    if _MODEL_ACCESS_TEXT.search(text) and not re.search(r"\b(?:401|402)\b", text):
        return "other"
    if _QUOTA_TEXT.search(text):
        return "quota"
    if _AUTH_TEXT.search(text):
        return "auth"
    if _TIMEOUT_TEXT.search(text):
        return "timeout"
    if _UNAVAILABLE_TEXT.search(text):
        return "unavailable"
    return "other"


def provider_kind(processor: Any) -> str | None:
    """The slot (``stt|llm|tts|realtime|translation``) a Pipecat processor serves, from its class name."""

    name = type(processor).__name__
    for marker, kind in (
        ("Realtime", "realtime"),
        ("Translat", "translation"),  # the interpreter / agent translators call OpenAI chat
    ):
        if marker in name:
            return kind
    if name.endswith("Service"):  # Pipecat services only, not e.g. LLMUserAggregator
        for marker, kind in (("STT", "stt"), ("TTS", "tts"), ("LLM", "llm")):
            if marker in name:
                return kind
    return None


class _State:
    """Per-session failure bookkeeping (kept on the session object)."""

    def __init__(self) -> None:
        self.failed: dict[str, Any] | None = None
        # Error times per (kind, group) inside the sliding window; ``group`` is
        # ``transient`` (unavailable/timeout) or ``other``.
        self.recent: dict[tuple[str, str], list[float]] = {}


def _state(session: Any) -> _State:
    state = session.__dict__.get("_provider_failure_state")
    if state is None:
        state = session.__dict__["_provider_failure_state"] = _State()
    return state


def provider_failure(session: Any) -> dict[str, Any] | None:
    """The terminal ``ProviderError`` payload recorded for ``session``, if it failed."""

    state = session.__dict__.get("_provider_failure_state") if session is not None else None
    return state.failed if state is not None else None


_end_tasks: set[asyncio.Task] = set()


def _emit(session: Any, event_type: str, payload: dict[str, object]) -> None:
    if not getattr(session, "event_client", None):
        return
    try:
        session.enqueue_event(session.event(event_type, payload))
    except RuntimeError:
        _logger.warning("Dropped %s event for session %s", event_type, session.session_id)


APOLOGY_BACKSTOP_SECONDS = 8.0  # > apology.MAX_PLAYBACK_SECONDS, which session.play_apology honours
# A transport that broke cannot carry the apology either.
_TRANSPORT_PROVIDERS = frozenset({"transport", "livekit"})


def _apology_possible(kind: str, provider: str) -> bool:
    return kind != "transport" and provider.lower() not in _TRANSPORT_PROVIDERS


async def _play_apology(session: Any) -> None:
    """ST-10b: tell the caller why the call ends (bounded; never blocks the end)."""

    play = getattr(session, "play_apology", None)
    if play is None:
        return
    try:
        # The session bounds itself; this is the backstop for any other session object.
        await asyncio.wait_for(play(), timeout=APOLOGY_BACKSTOP_SECONDS)
    except asyncio.TimeoutError:
        _logger.warning("Failure apology timed out; ending the session without it")
    except Exception:
        _logger.exception("Could not play the failure apology")


def _spawn_end(
    session: Any,
    reason: str,
    if_not_ended: Callable[[], Awaitable[Any]] | None,
    *,
    apology: bool = False,
) -> None:
    async def end() -> None:
        if apology:
            await _play_apology(session)
        ended = False
        try:
            ended = bool(await session.request_end(reason))
        except Exception:
            _logger.exception("Could not end the session after a provider failure")
        if not ended and if_not_ended is not None:
            await if_not_ended()

    task = asyncio.get_running_loop().create_task(end())
    _end_tasks.add(task)
    task.add_done_callback(_end_tasks.discard)


def _record(state: _State, kind: str, category: str, now: float) -> int:
    """Count this error in the sliding window; returns its 1-based attempt number."""

    group = "transient" if category in TRANSIENT_REASONS else "other"
    times = [t for t in state.recent.get((kind, group), []) if now - t < TRANSIENT_WINDOW_SECONDS]
    times.append(now)
    state.recent[(kind, group)] = times
    return len(times)


def _payload(
    kind: str, provider: str, category: str, stage: str | None, retryable: bool, attempt: int
) -> dict[str, Any]:
    if category not in CATEGORIES:
        category = "other"
    return {
        "provider": provider,
        "kind": kind,
        "category": category,
        "stage": stage or DEFAULT_STAGES.get(kind, "unknown"),
        "retryable": retryable,
        "attempt": attempt,
        "message": f"{provider} {kind} {_PHRASES[category]}",
    }


def _note_health(provider: str, category: str) -> None:
    """Tell the provider-health cache (ST-00) about a live failure; best effort."""

    try:
        from .provider_health import record_provider_failure

        record_provider_failure(provider, category)
    except Exception:  # pragma: no cover - health is advisory, never break the failure path
        _logger.debug("Could not record provider health", exc_info=True)


def report_provider_retry(
    session: Any | None,
    kind: str,
    provider: str,
    category: str,
    *,
    stage: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Emit ``ProviderError`` (retryable) for an error the caller is about to retry itself.

    The caller owns the retry (a re-issued translation request, a socket reconnect), so no
    policy is applied here. Returns the attempt number (0 without a session).
    """

    if session is None or getattr(session, "ending", False):
        return 0
    state = _state(session)
    attempt = _record(state, kind, category, clock())
    payload = _payload(kind, provider, category, stage, True, attempt)
    _logger.warning("Provider error, retrying: %s", payload["message"])
    _emit(session, "ProviderError", payload)
    return attempt


def report_provider_failure(
    session: Any | None,
    kind: str,
    provider: str,
    reason: str,
    *,
    stage: str | None = None,
    fatal: bool | None = None,
    allow_fallback: bool = False,
    if_not_ended: Callable[[], Awaitable[Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Apply the failure policy to one provider error; always emits ``ProviderError``.

    ``reason`` is the ``category``. Returns ``"ignored"`` (no session or the session is
    already ending), ``"duplicate"`` (a terminal failure was already reported), ``"retry"``
    (transient error, first in the window: the call goes on), ``"noted"`` (an ``other``
    error below the limit: the call goes on), ``"fallback"`` (the session restarts on the
    cascaded engine) or ``"failed"`` (the session is marked failed and being ended).
    ``if_not_ended`` runs when the session cannot be ended through its pipeline worker
    (e.g. standalone processors).
    """

    if session is None or getattr(session, "ending", False):
        return "ignored"
    state = _state(session)
    if state.failed is not None:
        return "duplicate"
    category = reason if reason in CATEGORIES else "other"
    if fatal is None:
        fatal = category in FATAL_REASONS
    attempt = _record(state, kind, category, clock())
    terminal = fatal
    if not terminal and category in TRANSIENT_REASONS:
        terminal = attempt >= 2  # one retry, then the call fails
    elif not terminal:
        terminal = attempt >= OTHER_FAILURE_LIMIT
    payload = _payload(kind, provider, category, stage, not terminal and category != "other", attempt)
    _emit(session, "ProviderError", payload)
    if not terminal:
        _logger.warning("Provider error (%s); the call goes on", payload["message"])
        return "retry" if category in TRANSIENT_REASONS else "noted"
    state.failed = payload
    _logger.error("Provider failed: %s", payload["message"])
    _note_health(provider, category)
    fallback = getattr(session, "request_engine_fallback", None)
    if allow_fallback and category not in FATAL_REASONS and fallback is not None and fallback():
        _emit(session, "EngineFallback", {"from": "realtime", "to": "cascaded"})
        state.failed = None  # the cascaded pipeline may fail too, and then the call ends
        return "fallback"
    if hasattr(session, "fail"):
        session.fail(FAILURE_REASON)
    if hasattr(session, "request_end"):
        _spawn_end(
            session, FAILURE_REASON, if_not_ended, apology=_apology_possible(kind, provider)
        )
    return "failed"


def report_provider_exception(
    session: Any | None, kind: str, provider: str, exc: BaseException, **kwargs: Any
) -> str:
    """:func:`report_provider_failure` for a caught exception (classified; never logged raw)."""

    return report_provider_failure(session, kind, provider, classify_error(exception=exc), **kwargs)


def cascaded_fallback_config(config: "PipelineConfig") -> "PipelineConfig | None":
    """``config`` on the cascaded engine, or None when the session cannot run cascaded.

    Feasible only for a one-way realtime interpreter session (the realtime engine serves
    nothing else) with the cascaded engine's providers configured.
    """

    interpreter = config.interpreter
    if config.mode != "interpreter" or not interpreter or config.participants:
        return None
    if interpreter.get("engine") != "realtime" or interpreter.get("direction", "one_way") != "one_way":
        return None
    if not (
        config.deepgram_api_key
        and config.tts_api_key
        and config.tts_voice_id
        and config.openai_api_key
    ):
        return None
    return dataclasses.replace(config, interpreter={**interpreter, "engine": "cascaded"})


def watch_provider_failures(
    worker: Any,
    session: Any | None,
    providers: Mapping[str, str],
    *,
    stall_watchdog: bool = True,
) -> None:
    """Report STT/LLM/TTS errors reaching ``worker`` (upstream ``ErrorFrame``) to the policy.

    Also attaches the :class:`~runtime.stall_watchdog.StallWatchdog` observer (hangs raise no
    error at all) unless ``stall_watchdog`` is False, e.g. for the realtime engine that has
    no STT/LLM/TTS service.
    """

    if session is None:
        return

    if stall_watchdog and hasattr(worker, "add_observer"):
        from .stall_watchdog import StallWatchdog

        worker.add_observer(StallWatchdog(session, providers))

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(_worker: Any, frame: Any) -> None:
        processor = getattr(frame, "processor", None)
        kind = provider_kind(processor)
        if kind is None:
            return
        category = getattr(frame, "category", None)
        if getattr(category, "value", None) == "application":
            # Application code (e.g. a tool handler) failed, not the provider.
            _logger.warning("Application error in %s; not a provider failure", type(processor).__name__)
            return
        reason = classify_error(
            exception=getattr(frame, "exception", None),
            text=str(getattr(frame, "error", "")),
            category=category,
        )
        # Pipecat flags a service it gave up on (permanent category, or a service that
        # marked itself "no longer usable", e.g. Cartesia quota_exceeded) as unusable.
        unusable = processor is not None and getattr(processor, "is_usable", True) is False
        fatal = bool(getattr(frame, "fatal", False)) or unusable or reason in FATAL_REASONS
        report_provider_failure(
            session,
            kind,
            providers.get(kind) or ("openai" if kind == "translation" else kind),
            reason,
            fatal=fatal,
        )

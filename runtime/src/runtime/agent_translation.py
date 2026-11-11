"""Agent + translator (TR-040): an assistant that works in one language for callers who
speak another.

The Platform sends ``mode: "agent"`` plus ``translation: {working_language,
caller_languages}``. The LLM, its tools, knowledge and ``record_outcome`` all stay in the
working language; only the edges of the conversation are translated::

    transport.input -> STT -> CallerLanguageTap -> user aggregator -> InboundTranslator
      -> LLM -> OutboundTranslator -> TTS -> events -> transport.output -> assistant aggregator

Inbound
-------
Translating at the *user aggregator boundary* (not on the transcript segments) keeps
Pipecat's turn handling untouched: interim/final transcripts reach the aggregator
immediately, so turn start/stop analysis and barge-in behave exactly as in the normal
agent pipeline. The aggregator writes the caller's words to the shared ``LLMContext`` and
pushes an ``LLMContextFrame`` to the LLM; :class:`InboundTranslator` sits between them,
translates the trailing user message(s) of that context into the working language *in
place* and forwards the frame. The context therefore only ever holds working-language
text. :class:`CallerLanguageTap` (before the aggregator) records the language Deepgram
reports per finalized segment so the translator knows which language the turn was in.

Outbound and the assistant context
----------------------------------
The assistant aggregator records what reaches it as ``TextFrame``s with
``append_to_context=True``; normally those are the words the TTS service reports as
spoken, which here would be the *translated* words. So the translated speech is sent as
``TTSSpeakFrame(..., append_to_context=False)`` (the TTS service reports it with
``append_to_context=False``, which the aggregator ignores), while the original
working-language sentence travels on separately as an ``LLMTextFrame`` with
``skip_tts=True`` (TTS passes such frames through untouched, so it is never spoken but
reaches the aggregator). The ``LLMFullResponseStart``/``End`` frames are forwarded the
same way (``skip_tts=True``) so the aggregator still opens and closes its assistant turn.

An original sentence is released to the aggregator only when its first translated chunk
is handed to TTS, and each response is processed inline, in frame order. An
``InterruptionFrame`` cancels the in-flight translation and Pipecat drops the queued
text frames, so unspoken sentences are neither translated nor recorded; the sentences
already started stay in the context (a deliberate approximation of "what the caller
heard": the aggregator cannot see playback of a language it did not synthesize).

When the caller already speaks the working language nothing is translated: the LLM
frames pass through untouched and TTS is switched back to the working language.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings

from .interpreter import (
    _SCRIPT_LANGUAGES,
    SUPPORTED_LANGUAGES,
    EmptyTranslationHook,
    SentenceChunker,
    TranslateFn,
    _close_stream,
    close_translator,
    as_streaming_translator,
    build_translation_messages,
    _script_votes,
    language_label,
)
from .glossary import Glossary, parse_glossary
from .guard_events import current_turn_id
from .language_capabilities import auto_detect_languages
from .participants import normalize_language
from .provider_failures import (
    TRANSIENT_REASONS,
    classify_error,
    report_provider_exception,
    report_provider_retry,
)
from .stall_watchdog import guard_first_token
from .timeline import TimelineTracker, UtteranceTimeline

_logger = logging.getLogger(__name__)

MAX_CALLER_LANGUAGES = 4
# Mirrors the Platform's AUTO_DIRECTION_LANGUAGES: Deepgram "multi" (code-switching)
# detects these. Every caller language and the working language must be in the set for
# per-utterance detection; otherwise the single caller language is fixed.
AUTO_DETECTION_LANGUAGES = auto_detect_languages()
PROCESSOR_NAME = "agent_translation"
# Small, fast, non-reasoning model for the two translation legs (each adds a round trip
# to every turn). Override with ``provider_settings.llm.translation_model``.
AGENT_TRANSLATION_MODEL = "gpt-4.1-nano"
# Outbound sentences are translated with the last few sentence pairs for consistency.
OUTBOUND_HISTORY_PAIRS = 2
# Languages written in a non-Latin script; their STT label is checked against the text.
_NON_LATIN_LANGUAGES = frozenset({"ru", "ko", "ja", "zh", "hi"})


@dataclass(frozen=True)
class AgentTranslationSettings:
    """Validated ``translation`` block of the Platform session contract."""

    working_language: str
    caller_languages: tuple[str, ...]
    # NS-08: required renderings and never-translate terms; see ``glossary``.
    glossary: Glossary = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AgentTranslationSettings":
        if not isinstance(raw, Mapping):
            raise RuntimeError("translation must be an object")
        working = normalize_language(raw.get("working_language"))
        callers_raw = raw.get("caller_languages")
        if working not in SUPPORTED_LANGUAGES:
            raise RuntimeError("translation working_language is not supported")
        if (
            not isinstance(callers_raw, (list, tuple))
            or not 1 <= len(callers_raw) <= MAX_CALLER_LANGUAGES
        ):
            raise RuntimeError("translation caller_languages must list 1-4 languages")
        callers = tuple(normalize_language(code) for code in callers_raw)
        if (
            any(code not in SUPPORTED_LANGUAGES for code in callers)
            or len(set(callers)) != len(callers)
            or working in callers
        ):
            raise RuntimeError(
                "translation caller_languages must be distinct supported languages "
                "other than the working language"
            )
        settings = cls(  # type: ignore[arg-type]
            working_language=working,
            caller_languages=callers,
            glossary=parse_glossary(raw.get("glossary")),
        )
        if len(callers) > 1 and not settings.auto:
            raise RuntimeError(
                "several caller languages require automatic language detection"
            )
        return settings

    @property
    def auto(self) -> bool:
        """Per-utterance language detection (Deepgram ``multi``) is available."""

        return all(
            code in AUTO_DETECTION_LANGUAGES
            for code in (self.working_language, *self.caller_languages)
        )

    @property
    def stt_language(self) -> str:
        return "multi" if self.auto else self.caller_languages[0]

    @property
    def initial_tts_language(self) -> str:
        return self.caller_languages[0]

    @property
    def instruction(self) -> str:
        language = language_label(self.working_language)
        text = (
            "Callers speak other languages; their words reach you translated into "
            f"{language}. Always answer in {language}; your reply is translated for them."
        )
        # Names and terms that must survive translation: the agent should say them
        # exactly as the glossary writes them.
        terms = [entry.source for entry in self.glossary if entry.keep]
        if terms:
            text += " Never alter these names or terms: " + ", ".join(terms) + "."
        return text


@dataclass
class _Segment:
    text: str
    language: str | None
    user_id: str
    timestamp: str


class CallerLanguageState:
    """Shared by the pipeline's translation processors."""

    def __init__(
        self,
        settings: AgentTranslationSettings,
        timeline: TimelineTracker | None = None,
    ) -> None:
        self.settings = settings
        # RM-003: the caller's turn that the next LLM response answers.
        self.timeline = timeline
        self.utterance: UtteranceTimeline | None = None
        self.language: str = settings.caller_languages[0]
        self.history: deque[tuple[str, str, str, str]] = deque(maxlen=OUTBOUND_HISTORY_PAIRS)
        self.segments: list[_Segment] = []

    @property
    def candidates(self) -> tuple[str, ...]:
        if not self.settings.auto:
            return self.settings.caller_languages
        return (*self.settings.caller_languages, self.settings.working_language)

    def detect(self, segments: list[_Segment], text: str) -> str:
        """The language of a caller turn, always one of :attr:`candidates`.

        Fixed sessions always return the caller language. Otherwise the *text* wins over
        the STT label when they disagree, because Deepgram "multi" mislabels short
        phrases (a Russian greeting came back as ``en``):

        * text dominated by a non-Latin script (Cyrillic, Hangul, kana, Han, Devanagari)
          is that script's candidate language (the STT label only picks between
          languages sharing the script, e.g. ``zh``/``ja``);
        * otherwise (Latin text) the STT label counts only if it is a Latin-script
          candidate, then the previous language if Latin, then the working language;
        * text without letters keeps the previous turn's language.
        """

        if not self.settings.auto:
            return self.settings.caller_languages[0]
        candidates = self.candidates
        votes: Counter[str] = Counter()
        for segment in segments:
            code = normalize_language(segment.language)
            if code in candidates:
                votes[code] += max(len(segment.text), 1)
        script_votes = _script_votes(text)
        if script_votes:
            script, _ = script_votes.most_common(1)[0]
            allowed = [code for code in _SCRIPT_LANGUAGES[script] if code in candidates]
            if allowed:
                best = votes.most_common(1)[0][0] if votes else None
                return best if best in allowed else allowed[0]
        latin = [code for code in candidates if code not in _NON_LATIN_LANGUAGES]
        latin_votes = Counter({c: n for c, n in votes.items() if c in latin})
        if any(ch.isalpha() for ch in text):
            if latin_votes:
                return latin_votes.most_common(1)[0][0]
            pool = latin or list(candidates)
            working = self.settings.working_language
            return self.language if self.language in pool else (working if working in pool else pool[0])
        return self.language if self.language in candidates else self.settings.caller_languages[0]


def emit_event(session: Any | None, event_type: str, payload: dict[str, object]) -> None:
    """Queue a Platform event; telemetry failures never break the conversation."""

    if session is None or not getattr(session, "event_client", None):
        return
    try:
        session.enqueue_event(session.event(event_type, payload))
    except RuntimeError:
        _logger.warning("Dropped %s event for session %s", event_type, session.session_id)


def emit_latency(session: Any | None, metric: str, seconds: float, model: str) -> None:
    emit_event(
        session,
        "LatencyMeasured",
        {
            "metric": metric,
            "value_ms": round(seconds * 1000, 1),
            "processor": PROCESSOR_NAME,
            "model": model,
        },
    )


async def collect_translation(translate: Any, messages: list[dict[str, str]]) -> str:
    stream = translate(messages)
    parts: list[str] = []
    try:
        async for delta in stream:
            parts.append(delta)
    finally:
        await _close_stream(stream)
    return "".join(parts).strip()


INBOUND_RETRY_DELAY_SECONDS = 0.2  # pause before the inbound translator's one retry


class CallerLanguageTap(FrameProcessor):
    """Record the language of each finalized transcript segment; pass frames through."""

    def __init__(self, state: CallerLanguageState) -> None:
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        tracker = self._state.timeline
        if tracker is not None:
            # The aggregator (downstream) owns the VAD; its frames reach this tap
            # upstream. Either direction is the one copy this processor sees.
            if isinstance(frame, VADUserStartedSpeakingFrame):
                tracker.speech_started("", frame.start_secs)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                tracker.turn_stopped("")
            elif isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
                if frame.text.strip():
                    tracker.transcript("")
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._state.segments.append(
                _Segment(
                    text=frame.text.strip(),
                    language=frame.language.value
                    if hasattr(frame.language, "value")
                    else frame.language,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                )
            )
        await self.push_frame(frame, direction)


class InboundTranslator(FrameProcessor):
    """Translate the caller's turn into the working language before the LLM sees it."""

    # Messages translated so far (strong refs keep ``id`` unique); bounded.
    _DONE_LIMIT = 64

    def __init__(
        self,
        *,
        state: CallerLanguageState,
        translate: TranslateFn,
        model: str,
        session: Any | None = None,
        warm_up: Callable[[], Awaitable[None]] | None = None,
        on_empty_translation: EmptyTranslationHook | None = None,
        provider: str = "openai",
    ) -> None:
        super().__init__()
        self._provider = provider
        self._state = state
        self._translator = translate
        self._translate = as_streaming_translator(translate)
        self._model = model
        self._session = session
        self._warm_up = warm_up
        self._on_empty_translation = on_empty_translation
        self._done: deque[Any] = deque(maxlen=self._DONE_LIMIT)

    async def cleanup(self) -> None:
        await super().cleanup()
        await close_translator(self._translator)  # shared with OutboundTranslator; closing twice is harmless

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._warm_up is not None:
            # Connect to the translation provider while the call is being set up, so
            # the first caller turn does not pay the TLS handshake.
            self.create_task(self._warm_up(), "agent-translation-warm-up")
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._translate_context(frame)
        await self.push_frame(frame, direction)

    def _is_done(self, message: Any) -> bool:
        return any(message is done for done in self._done)

    def _pending_messages(self, frame: LLMContextFrame) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        for message in reversed(frame.context.messages):
            if (
                not isinstance(message, dict)
                or message.get("role") != "user"
                or not isinstance(message.get("content"), str)
                or self._is_done(message)
            ):
                break
            pending.append(message)
        pending.reverse()
        return pending

    async def _translate_context(self, frame: LLMContextFrame) -> None:
        pending = self._pending_messages(frame)
        if not pending:
            return
        segments, self._state.segments = self._state.segments, []
        tracker = self._state.timeline
        if tracker is not None:
            # A turn continued after a pause arrives as another context frame without a
            # new speech start: it keeps timing from the speech that is still unanswered.
            self._state.utterance = tracker.take("") or self._state.utterance
        settings = self._state.settings
        messages = frame.context.messages
        # A turn can reach the context as several user messages (the aggregator ends it
        # at a short pause, or an earlier translation was cut off by a barge-in). They
        # are one utterance for the LLM: join them into the newest message and translate
        # once, with a single short call.
        message = pending[-1]
        for earlier in pending[:-1]:
            messages[:] = [m for m in messages if m is not earlier]
        original = " ".join(m["content"].strip() for m in pending).strip()
        language = self._state.detect(segments, original)
        self._state.language = language
        translated = original
        if language != settings.working_language and original:
            if tracker is not None:
                tracker.mark(self._state.utterance, "translation_start")
            translated = await self._translate_message(original, language)
        self._done.append(message)
        message["content"] = translated
        event: dict[str, object] = {
            "content": original,
            "role": "user",
            "primary_language": language,
            "detected_languages": [language],
            "translated_text": translated,
            "working_language": settings.working_language,
        }
        if segments:
            event.update(user_id=segments[-1].user_id, timestamp=segments[-1].timestamp)
        turn_id = current_turn_id(self._session, self._state.utterance)  # ST-02a
        if turn_id is not None:
            event["turn_id"] = turn_id
        emit_event(self._session, "TranscriptFinal", event)

    async def _translate_message(self, text: str, source: str) -> str:
        """Translate ``text`` in one call; on provider errors the original goes on.

        No conversation history is sent: the caller's words are translated on their own,
        which keeps the prompt (and the round trip) as short as possible.
        """

        working = self._state.settings.working_language
        messages = build_translation_messages(
            text, source, working, glossary=self._state.settings.glossary
        )
        started = time.monotonic()
        try:
            try:
                translated = await collect_translation(
                    lambda m: guard_first_token(self._translate(m)), messages
                )
            except asyncio.CancelledError:
                raise
            except Exception as first:
                # ST-10a: the one place this request can be re-issued as is (nothing has
                # been spoken yet). A transient failure is retried once; anything else, or
                # the retry failing too, takes the normal failure path below.
                category = classify_error(exception=first)
                if category not in TRANSIENT_REASONS:
                    raise
                report_provider_retry(
                    self._session, "translation", self._provider, category, stage="inbound_translation"
                )
                _logger.warning(
                    "Inbound translation failed (%s); retrying once", type(first).__name__
                )
                await asyncio.sleep(INBOUND_RETRY_DELAY_SECONDS)
                translated = await collect_translation(
                    lambda m: guard_first_token(self._translate(m)), messages
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.warning(
                "Inbound translation failed (%s); passing the caller's words through",
                type(exc).__name__,
            )
            report_provider_exception(
                self._session,
                "translation",
                getattr(exc, "translator_provider", None) or self._provider,
                exc,
                stage="inbound_translation",
            )  # PR-30
            return text
        if not translated:
            # ST-12: the caller's words go on unchanged; let the orchestrator know.
            if self._on_empty_translation is not None:
                self._on_empty_translation(
                    {
                        "text": text,
                        "source_language": source,
                        "target_language": working,
                    }
                )
            return text
        emit_latency(
            self._session, "inbound_translation_ms", time.monotonic() - started, self._model
        )
        return translated


@dataclass
class _Response:
    """One LLM response being relayed to the caller."""

    language: str
    passthrough: bool
    chunker: SentenceChunker = field(default_factory=SentenceChunker)
    originals: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)
    first_chunk_measured: bool = False
    utterance: UtteranceTimeline | None = None


class OutboundTranslator(FrameProcessor):
    """Translate the LLM's reply per sentence and speak it in the caller's language."""

    def __init__(
        self,
        *,
        state: CallerLanguageState,
        translate: TranslateFn,
        model: str,
        session: Any | None = None,
        on_empty_translation: EmptyTranslationHook | None = None,
        provider: str = "openai",
    ) -> None:
        super().__init__()
        self._provider = provider
        self._state = state
        self._translator = translate
        self._translate = as_streaming_translator(translate)
        self._model = model
        self._session = session
        self._on_empty_translation = on_empty_translation
        self._tts_language = state.settings.initial_tts_language
        self._response: _Response | None = None

    # -- frame handling ----------------------------------------------------------

    async def cleanup(self) -> None:
        await super().cleanup()
        await close_translator(self._translator)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            # super() already cancelled the in-flight translation and dropped the
            # queued text frames; report what was handed to TTS and forget the rest.
            self._emit_assistant_transcript()
            if self._response is not None:
                self._interrupted(self._response)
            self._response = None
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._finish_response()
            await self._start_response(frame)
        elif isinstance(frame, LLMTextFrame) and not frame.skip_tts:
            await self._handle_text(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._end_response(frame)
        else:
            await self.push_frame(frame, direction)

    async def _start_response(self, frame: LLMFullResponseStartFrame) -> None:
        settings = self._state.settings
        language = self._state.language
        passthrough = language == settings.working_language
        self._response = _Response(language=language, passthrough=passthrough)
        tracker = self._state.timeline
        if tracker is not None:
            # The first response after a caller turn is the one timed (a later response of
            # the same turn, e.g. after a tool call, is not).
            self._response.utterance, self._state.utterance = self._state.utterance, None
            if passthrough:
                # No translation: the LLM text goes straight to TTS, in as many pieces as
                # the TTS service chooses, so the output going idle ends the utterance.
                tracker.enqueue_audio(self._response.utterance)
        await self._ensure_tts_language(language)
        if not passthrough:
            frame.skip_tts = True  # opens the assistant turn; never reaches TTS
        await self.push_frame(frame)

    async def _handle_text(self, frame: LLMTextFrame) -> None:
        if self._response is None:
            await self._start_response(LLMFullResponseStartFrame())
        response = self._response
        assert response is not None
        if response.passthrough:
            response.originals.append(frame.text)
            await self.push_frame(frame)
            return
        for sentence in response.chunker.feed(frame.text):
            await self._speak_sentence(response, sentence)

    async def _end_response(self, frame: LLMFullResponseEndFrame) -> None:
        response = self._response
        if response is not None and not response.passthrough:
            for sentence in response.chunker.flush():
                await self._speak_sentence(response, sentence)
            frame.skip_tts = True  # closes the assistant turn; never reaches TTS
        await self._finish_response()
        await self.push_frame(frame)

    async def _finish_response(self) -> None:
        if self._response is not None:
            self._emit_assistant_transcript()
            self._speech_done(self._response)
            self._response = None

    def _interrupted(self, response: _Response) -> None:
        """A barge-in cut the reply off: unheard speech keeps its timing for the next reply."""

        tracker = self._state.timeline
        if tracker is not None and tracker.abandon(response.utterance):
            self._state.utterance = response.utterance
        else:
            self._speech_done(response)

    def _speech_done(self, response: _Response) -> None:
        tracker = self._state.timeline
        if tracker is not None:
            tracker.speech_done(response.utterance, count_known=not response.passthrough)

    # -- translation -------------------------------------------------------------

    async def _ensure_tts_language(self, language: str) -> None:
        if language != self._tts_language:
            self._tts_language = language
            await self.push_frame(TTSUpdateSettingsFrame(delta=TTSSettings(language=language)))

    async def _speak_sentence(self, response: _Response, sentence: str) -> None:
        settings = self._state.settings
        messages = build_translation_messages(
            sentence,
            settings.working_language,
            response.language,
            tuple(self._state.history),
            glossary=settings.glossary,
        )
        started = time.monotonic()
        released = False
        chunker = SentenceChunker()
        deltas: list[str] = []
        tracker = self._state.timeline

        async def say(chunk: str) -> None:
            nonlocal released
            if not released:
                # The sentence is (about to be) heard: record its working-language text.
                released = True
                response.originals.append(sentence)
                context_frame = LLMTextFrame(sentence)
                context_frame.skip_tts = True
                context_frame.includes_inter_frame_spaces = False
                await self.push_frame(context_frame)
                if not response.first_chunk_measured:
                    response.first_chunk_measured = True
                    emit_latency(
                        self._session,
                        "outbound_translation_first_chunk_ms",
                        time.monotonic() - started,
                        self._model,
                    )
            response.spoken.append(chunk)
            if tracker is not None:
                tracker.mark(response.utterance, "translation_first_chunk")
                tracker.enqueue_audio(response.utterance)
                tracker.chunk_pushed(response.utterance)
                tracker.mark(response.utterance, "tts_request")
            await self.push_frame(TTSSpeakFrame(chunk, append_to_context=False))

        if tracker is not None:
            tracker.mark(response.utterance, "translation_start")
        stream = guard_first_token(self._translate(messages))
        try:
            async for delta in stream:
                if tracker is not None and delta:
                    tracker.mark(response.utterance, "translation_first_token")
                deltas.append(delta)
                for chunk in chunker.feed(delta):
                    await say(chunk)
            for chunk in chunker.flush():
                await say(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.warning(
                "Outbound translation failed (%s); %s",
                type(exc).__name__,
                "stopped after the chunks already spoken"
                if released
                else "speaking the sentence untranslated",
            )
            report_provider_exception(
                self._session,
                "translation",
                getattr(exc, "translator_provider", None) or self._provider,
                exc,
            )  # PR-30
            if not released:
                await say(sentence)
            return
        finally:
            await _close_stream(stream)
        translation = "".join(deltas).strip()
        if translation:
            self._state.history.append(
                (settings.working_language, sentence, response.language, translation)
            )
        elif self._on_empty_translation is not None:
            # ST-12: nothing was spoken for this sentence; let the orchestrator know.
            self._on_empty_translation(
                {
                    "text": sentence,
                    "source_language": settings.working_language,
                    "target_language": response.language,
                }
            )

    # -- events ------------------------------------------------------------------

    def _emit_assistant_transcript(self) -> None:
        response = self._response
        if response is None:
            return
        if response.passthrough:
            spoken = original = "".join(response.originals).strip()
        else:
            spoken = " ".join(response.spoken).strip()
            original = " ".join(response.originals).strip()
        if not spoken:
            return
        event: dict[str, object] = {
            "content": spoken,
            "role": "assistant",
            "primary_language": response.language,
            "detected_languages": [response.language],
            "original_text": original,
            "working_language": self._state.settings.working_language,
        }
        turn_id = current_turn_id(self._session, response.utterance)  # ST-02a
        if turn_id is not None:
            event["turn_id"] = turn_id
        emit_event(self._session, "TranscriptFinal", event)
        response.originals = []
        response.spoken = []

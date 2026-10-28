"""Answer guard (ST-12): an empty or broken LLM answer never becomes silence.

:class:`AnswerGuard` sits between the agent LLM (and the outbound translator, when
present) and the TTS service. It watches one LLM response at a time, delimited by
``LLMFullResponseStartFrame`` and ``LLMFullResponseEndFrame``:

* A response with no non-whitespace text and no function call gets a short spoken
  REPEAT line in the caller's language instead of nothing.
* Spoken text is capped at ``max_spoken_chars``: once the cap is reached, text is
  forwarded up to the next sentence boundary and the rest of the response is dropped.
* An interruption resets the state; everything else passes through unchanged.
"""

from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .fallback_lines import REPEAT, line

_logger = logging.getLogger(__name__)

MAX_SPOKEN_CHARS = 600
SENTENCE_BOUNDARIES = frozenset(".!?。！？")

LanguageSource = str | Callable[[], str] | None
LanguageCallback = Callable[[str], Any]
TruncatedCallback = Callable[[int], Any]


class AnswerGuard(FrameProcessor):
    """Replace empty LLM answers with a spoken fallback and cap overlong answers."""

    def __init__(
        self,
        *,
        language: LanguageSource = None,
        max_spoken_chars: int = MAX_SPOKEN_CHARS,
        on_empty: LanguageCallback | None = None,
        on_truncated: TruncatedCallback | None = None,
    ) -> None:
        super().__init__()
        self._language = language
        self._max_spoken_chars = max_spoken_chars
        self._on_empty = on_empty
        self._on_truncated = on_truncated
        self._reset()

    def _reset(self) -> None:
        self._in_response = False
        self._has_text = False
        self._has_function_call = False
        self._spoken_chars = 0
        self._truncated = False

    # -- frame handling ----------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            self._in_response = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._in_response and not (self._has_text or self._has_function_call):
                await self._speak_fallback()
            self._reset()
            await self.push_frame(frame, direction)
        elif self._in_response and isinstance(
            frame,
            (
                FunctionCallsStartedFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
            ),
        ):
            self._has_function_call = True
            await self.push_frame(frame, direction)
        elif self._in_response and isinstance(frame, (LLMTextFrame, TTSSpeakFrame)):
            await self._handle_text(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _handle_text(
        self, frame: LLMTextFrame | TTSSpeakFrame, direction: FrameDirection
    ) -> None:
        if isinstance(frame, LLMTextFrame) and frame.skip_tts:
            # Context-only text (e.g. the outbound translator's transcript): not spoken.
            await self.push_frame(frame, direction)
            return
        text = frame.text
        if text.strip():
            self._has_text = True
        if self._truncated:
            return
        cut = max(0, self._max_spoken_chars - self._spoken_chars)
        if cut >= len(text):
            self._spoken_chars += len(text)
            await self.push_frame(frame, direction)
            return
        boundary = _first_boundary(text, cut)
        if boundary is None:
            # Over the cap, mid-sentence: keep going until a sentence ends.
            self._spoken_chars += len(text)
            await self.push_frame(frame, direction)
            return
        kept = text[: boundary + 1]
        self._spoken_chars += len(kept)
        self._truncated = True
        if kept != text:
            frame = copy.copy(frame)
            frame.text = kept
        await self.push_frame(frame, direction)
        await self._notify(self._on_truncated, self._spoken_chars)

    async def _speak_fallback(self) -> None:
        text = line(REPEAT, self._current_language())
        await self.push_frame(TTSSpeakFrame(text, append_to_context=False))
        await self._notify(self._on_empty, self._current_language())

    def _current_language(self) -> str | None:
        source = self._language
        try:
            return source() if callable(source) else source
        except Exception:
            _logger.warning(
                "Answer guard could not resolve the caller language; using English"
            )
            return None

    @staticmethod
    async def _notify(callback: Callable[[Any], Any] | None, value: Any) -> None:
        if callback is None:
            return
        try:
            result = callback(value)
            if inspect.isawaitable(result):
                await result
        except Exception:
            _logger.warning("Answer guard callback failed", exc_info=True)


def _first_boundary(text: str, start: int) -> int | None:
    for index in range(start, len(text)):
        if text[index] in SENTENCE_BOUNDARIES:
            return index
    return None

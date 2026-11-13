"""Drop transcripts that are the call's own translated speech coming back through a microphone.

When a participant listens on speakers (or two people share a laptop), the microphone picks up the
translation that was just played. Without a guard that audio is transcribed as if the person had
said it, translated again, played again, and the call loops: observed 2026-11-13, "클리핑이
불가능합니다" / "Clipping is not possible" twelve times in 30 s. Browser echo cancellation does not
help when the translation plays in another tab or device.

The guard remembers every translation this session spoke for ``window_s`` seconds. A new
transcript that closely matches one of them is treated as an echo and dropped before it is
translated, spoken, or stored. Short phrases (under ``min_chars`` letters/digits, e.g. "Hello.")
are never dropped, so a greeting answered with the same word still goes through.

Env: ``RUNTIME_ECHO_GUARD=0`` turns it off; ``RUNTIME_ECHO_GUARD_SECS`` sets the window (default 10).
"""

from __future__ import annotations

import os
import time
from collections import deque
from difflib import SequenceMatcher

DEFAULT_WINDOW_S = 10.0
DEFAULT_MIN_CHARS = 8
DEFAULT_SIMILARITY = 0.8


def normalize(text: str) -> str:
    """Casefold and keep only letters and digits (any script), so punctuation and spacing don't matter."""

    return "".join(ch for ch in text.casefold() if ch.isalnum())


class EchoGuard:
    def __init__(
        self,
        window_s: float = DEFAULT_WINDOW_S,
        min_chars: int = DEFAULT_MIN_CHARS,
        similarity: float = DEFAULT_SIMILARITY,
        clock=time.monotonic,
    ) -> None:
        self._window_s = window_s
        self._min_chars = min_chars
        self._similarity = similarity
        self._clock = clock
        self._spoken: deque[tuple[float, str]] = deque(maxlen=64)

    @classmethod
    def from_env(cls) -> "EchoGuard | None":
        if os.getenv("RUNTIME_ECHO_GUARD", "1").strip().lower() in {"0", "false", "no", "off"}:
            return None
        try:
            window = float(os.getenv("RUNTIME_ECHO_GUARD_SECS", str(DEFAULT_WINDOW_S)))
        except ValueError:
            window = DEFAULT_WINDOW_S
        return cls(window_s=max(0.0, window))

    def _prune(self, now: float) -> None:
        while self._spoken and now - self._spoken[0][0] > self._window_s:
            self._spoken.popleft()

    def record(self, text: str) -> None:
        """Remember text the session just spoke to someone."""

        norm = normalize(text)
        if norm:
            now = self._clock()
            self._prune(now)
            self._spoken.append((now, norm))

    def is_echo(self, text: str) -> bool:
        """True when ``text`` closely matches something spoken within the window."""

        norm = normalize(text)
        if len(norm) < self._min_chars:
            return False
        now = self._clock()
        self._prune(now)
        for _, spoken in self._spoken:
            if norm == spoken or norm in spoken:
                return True
            if SequenceMatcher(None, norm, spoken, autojunk=False).ratio() >= self._similarity:
                return True
        return False

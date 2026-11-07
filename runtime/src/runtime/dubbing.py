"""Cinematic dubbing: the speaker's real voice quietly under the translation (opt-in).

Off by default. With ``interpreter.dubbing`` set, each listener's translation track also carries
the *other* person's original voice, delayed a little, kept low, and ducked further while the
translation speaks, so the listener hears a real person with a translator over them rather than
only a synthetic voice.

Design (``docs/dubbing-feasibility.md``, option B):

* The translation path is never touched: STT, translation and TTS run as before, and a TTS
  frame is mixed the moment it is written. The original adapts to the translation, never the
  other way round. Idle, a translation frame goes out at once; while the original is playing it
  waits for at most the next block (20 ms).
* The speaker's microphone is **copied** after the per-participant resampler that feeds the
  lane's VAD and STT. Nothing is changed or delayed on that path.
* ``DubMixer`` is plain synchronous DSP with an injectable clock (fast to test): delay line,
  voiced gate, ducking envelope, soft limiter. ``DubbingChannel`` drives one listener's
  ``rtc.AudioSource`` with it, and only while there is something to play: an idle call
  publishes nothing, exactly as without dubbing.
* A listener only ever gets the other person's voice (``ListenerAudioRouter.feed_original``),
  so nobody hears their own delayed voice.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

RATE = 24000  # the translation tracks' sample rate
BLOCK_MS = 10
BLOCK = RATE * BLOCK_MS // 1000
FULL_SCALE = 32768.0
MIN_JITTER_MS = 40.0  # even "no delay" keeps a small buffer: network frames do not arrive evenly
MAX_TRANSLATION_S = 60.0  # bound on buffered translation audio
LIMITER_KNEE = 0.8  # fraction of full scale where soft limiting starts


@dataclass(frozen=True)
class DubbingProfile:
    """How the original voice sounds. Levels are dB relative to the speaker's own level."""

    name: str = "balanced"
    delay_ms: float = 150.0  # extra delay of the original, on top of the network path
    base_gain_db: float = -10.0  # original while nothing is being translated
    duck_gain_db: float = -20.0  # original while the translation speaks
    attack_ms: float = 120.0  # time to duck once the translation starts
    release_ms: float = 500.0  # time to come back after the translation ends
    gate_db: float = -50.0  # the original is silent below this level (microphone noise)
    gate_hold_ms: float = 200.0  # keeps the gate open across short gaps between words

    def __post_init__(self) -> None:
        if not 0 <= self.delay_ms <= 2000:
            raise ValueError("dubbing delay_ms must be between 0 and 2000")
        if not (self.duck_gain_db <= self.base_gain_db <= 0):
            raise ValueError("dubbing gains must satisfy duck_gain_db <= base_gain_db <= 0")
        if not (5 <= self.attack_ms <= 5000 and 5 <= self.release_ms <= 5000):
            raise ValueError("dubbing attack_ms and release_ms must be between 5 and 5000")
        if not (-90 <= self.gate_db <= -20 and 0 <= self.gate_hold_ms <= 2000):
            raise ValueError("dubbing gate_db must be -90..-20 and gate_hold_ms 0..2000")


# Experiment presets (``scripts/dubbing_lab.py`` compares them; none is a proven best).
PROFILES: dict[str, DubbingProfile] = {
    "subtle": DubbingProfile("subtle", delay_ms=0, base_gain_db=-16, duck_gain_db=-26, attack_ms=150),
    "balanced": DubbingProfile("balanced"),
    # The shareable-link calls' test setting: the balanced profile with the original 24 dB down
    # while the translation speaks (goal: start near -24 dB; docs/dubbing-listening.md).
    "cinematic": DubbingProfile("cinematic", duck_gain_db=-24),
    "present": DubbingProfile("present", delay_ms=300, base_gain_db=-6, duck_gain_db=-18, attack_ms=100),
}
_NUMBERS = ("delay_ms", "base_gain_db", "duck_gain_db", "attack_ms", "release_ms", "gate_db", "gate_hold_ms")


def parse_dubbing(raw: Any) -> DubbingProfile | None:
    """``interpreter.dubbing``: absent/false is off; true a preset; a name or a mapping tunes it.

    ``{"profile": "balanced", "duck_gain_db": -28}`` starts from a preset and overrides numbers;
    ``{"enabled": false}`` is off. Raises ``RuntimeError`` (like the other interpreter settings).
    """

    if raw is None or raw is False:
        return None
    if raw is True:
        return PROFILES["balanced"]
    if isinstance(raw, str):
        raw = {"profile": raw}
    if not isinstance(raw, Mapping):
        raise RuntimeError("interpreter dubbing must be a boolean, a profile name or an object")
    if raw.get("enabled", True) is False:
        return None
    name = raw.get("profile", "balanced")
    if name not in PROFILES:
        raise RuntimeError(f"unsupported dubbing profile: {name}")
    unknown = set(raw) - {"enabled", "profile", *_NUMBERS}
    if unknown:
        raise RuntimeError(f"unknown dubbing setting: {sorted(unknown)[0]}")
    overrides: dict[str, float] = {}
    for key in _NUMBERS:
        if key in raw:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"dubbing {key} must be a number")
            overrides[key] = float(value)
    try:
        return replace(PROFILES[name], **overrides)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _level_db(samples: np.ndarray) -> float:
    x = samples.astype(np.float32)
    return 20 * math.log10(max(float(np.sqrt(np.mean(x * x))) / FULL_SCALE, 1e-9))


def _db(db: float) -> float:
    return 10 ** (db / 20)


def _smooth(current: float, target: float, tau_ms: float) -> float:
    """One block of an exponential approach; ``tau_ms`` is the time to get ~95% of the way."""

    return target + (current - target) * math.exp(-3.0 * BLOCK_MS / tau_ms)


class DubMixer:
    """Original voice + translation -> one stream, 20 ms at a time. No I/O, no threads."""

    def __init__(self, profile: DubbingProfile, *, rate: int = RATE) -> None:
        if rate != RATE:
            raise ValueError(f"DubMixer works at {RATE} Hz")
        self.profile = profile
        self._delay_s = max(profile.delay_ms, MIN_JITTER_MS) / 1000
        self._original: deque[tuple[float, np.ndarray]] = deque()  # (time it may play, int16)
        self._original_samples = 0
        self._translation: deque[np.ndarray] = deque()
        self._translation_samples = 0
        self._base = _db(profile.base_gain_db)
        self._duck = _db(profile.duck_gain_db)
        self.gain = self._base  # current gain of the original (ducking envelope)
        self.gate = 0.0  # current voiced gate, 0..1
        self._hold = 0.0  # ms of gate hold left
        self.dropped_samples = 0
        self._last_voiced = -math.inf  # when the speaker last made a sound (for the ingest gate)

    # -- input ---------------------------------------------------------------------

    def push_original(self, pcm: bytes, now: float) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16)
        if not samples.size:
            return
        # A microphone delivers frames in silence too. Quiet audio far from any speech is not
        # queued, so a silent speaker keeps the mixer (and the track) idle.
        if _level_db(samples) > self.profile.gate_db:
            self._last_voiced = now
        elif now - self._last_voiced > self.profile.gate_hold_ms / 1000 + 0.1:
            return
        self._original.append((now + self._delay_s, samples.copy()))
        self._original_samples += samples.size
        limit = int((self._delay_s + 2.0) * RATE)  # a stalled consumer never grows memory
        while self._original_samples > limit and self._original:
            _, old = self._original.popleft()
            self._original_samples -= old.size
            self.dropped_samples += old.size

    def push_translation(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16)
        if not samples.size:
            return
        self._translation.append(samples.copy())
        self._translation_samples += samples.size
        limit = int(MAX_TRANSLATION_S * RATE)
        while self._translation_samples > limit and self._translation:
            old = self._translation.popleft()
            self._translation_samples -= old.size
            self.dropped_samples += old.size

    def clear_translation(self) -> None:
        """An interruption: drop translated audio that has not played yet."""

        self._translation.clear()
        self._translation_samples = 0

    def take_translation(self) -> bytes:
        """The translation audio still waiting (used when dubbing is switched off mid-sentence)."""

        data = b"".join(chunk.tobytes() for chunk in self._translation)
        self.clear_translation()
        return data

    def clear(self) -> None:
        """Release every buffer (end of call)."""

        self.clear_translation()
        self._original.clear()
        self._original_samples = 0

    # -- output --------------------------------------------------------------------

    @property
    def pending(self) -> bool:
        """Whether rendering another block would produce sound (or finish a fade)."""

        return bool(self._translation_samples or self._original_samples or self._hold > 0 or self.gate > 0.01)

    @property
    def buffered_bytes(self) -> int:
        return 2 * (self._translation_samples + self._original_samples)

    def _take_original(self, now: float) -> np.ndarray:
        out = np.zeros(BLOCK, dtype=np.int16)
        filled = 0
        while filled < BLOCK and self._original and self._original[0][0] <= now:
            due, chunk = self._original.popleft()
            n = min(BLOCK - filled, chunk.size)
            out[filled : filled + n] = chunk[:n]
            filled += n
            self._original_samples -= n
            if n < chunk.size:
                self._original.appendleft((due, chunk[n:]))
        return out

    def _take_translation(self) -> tuple[np.ndarray, bool]:
        out = np.zeros(BLOCK, dtype=np.int16)
        filled = 0
        while filled < BLOCK and self._translation:
            chunk = self._translation.popleft()
            n = min(BLOCK - filled, chunk.size)
            out[filled : filled + n] = chunk[:n]
            filled += n
            self._translation_samples -= n
            if n < chunk.size:
                self._translation.appendleft(chunk[n:])
        return out, filled > 0

    def render(self, now: float) -> bytes:
        """The next 20 ms (int16, mono, 24 kHz). ``now`` decides which original audio is due."""

        p = self.profile
        translation, translating = self._take_translation()
        translating = translating or self._translation_samples > 0
        original = self._take_original(now).astype(np.float32)

        # Voiced gate: microphone hiss and room noise between words stay out of the mix.
        level_db = _level_db(original)
        if level_db > p.gate_db:
            self._hold = p.gate_hold_ms
            gate_target = 1.0
        elif self._hold > 0:
            self._hold -= BLOCK_MS
            gate_target = 1.0
        else:
            gate_target = 0.0
        gate_new = _smooth(self.gate, gate_target, 20.0 if gate_target > self.gate else 150.0)

        # Ducking: fast down when the translation speaks, slow back up afterwards.
        target = self._duck if translating else self._base
        gain_new = _smooth(self.gain, target, p.attack_ms if target < self.gain else p.release_ms)

        # Linear ramps across the block: no zipper noise, no step in the volume.
        ramp = np.linspace(0.0, 1.0, BLOCK, endpoint=False, dtype=np.float32)
        envelope = (self.gain + (gain_new - self.gain) * ramp) * (self.gate + (gate_new - self.gate) * ramp)
        self.gain, self.gate = gain_new, gate_new

        mixed = translation.astype(np.float32) + original * envelope
        return _soft_limit(mixed).tobytes()


def _soft_limit(samples: np.ndarray) -> np.ndarray:
    """Smoothly compress peaks above the knee so a loud overlap never hard-clips."""

    knee = LIMITER_KNEE * FULL_SCALE
    over = np.abs(samples) > knee
    if over.any():
        room = FULL_SCALE - 1 - knee
        magnitude = np.abs(samples)
        limited = knee + room * np.tanh((magnitude - knee) / room)
        samples = np.where(over, np.sign(samples) * limited, samples)
    return np.clip(samples, -FULL_SCALE, FULL_SCALE - 1).astype(np.int16)


class DubbingChannel:
    """Plays one listener's mix into their ``AudioSource`` while there is something to play."""

    def __init__(
        self,
        source: Any,
        profile: DubbingProfile,
        *,
        make_frame: Callable[[bytes], Any],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self.mixer = DubMixer(profile)
        self._make_frame = make_frame
        self._clock = clock
        self._wake = asyncio.Event()
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    def _ensure_task(self) -> None:
        if not self._closed and (self._task is None or self._task.done()):
            self._task = asyncio.get_running_loop().create_task(self._run())
        self._wake.set()

    def push_translation(self, pcm: bytes) -> None:
        if self._closed:
            return
        self.mixer.push_translation(pcm)
        self._ensure_task()

    def push_original(self, pcm: bytes) -> None:
        if self._closed:
            return
        self.mixer.push_original(pcm, self._clock())
        self._ensure_task()

    def clear(self) -> None:
        self.mixer.clear_translation()

    async def _run(self) -> None:
        period = BLOCK_MS / 1000
        try:
            while not self._closed:
                if not self.mixer.pending:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                # Start of a burst: render one block now and one ahead, so the source is
                # never starved by timer jitter, then one block per period (staying one
                # block ahead of what is playing).
                started = self._clock()
                for _ in range(2):
                    await self._source.capture_frame(self._make_frame(self.mixer.render(self._clock())))
                next_at = started + period
                while self.mixer.pending and not self._closed:
                    now = self._clock()
                    if next_at - now > 0:
                        await asyncio.sleep(next_at - now)
                    elif now - next_at > 0.25:  # we stalled: resynchronise instead of bursting
                        next_at = now
                    await self._source.capture_frame(self._make_frame(self.mixer.render(self._clock())))
                    next_at += period
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.warning("Dubbing mixer stopped", exc_info=True)

    async def stop(self) -> bytes:
        """Stop mixing; returns translation audio not yet played so the caller can send it directly."""

        self._closed = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        rest = self.mixer.take_translation()
        self.mixer.clear()
        return rest

    async def aclose(self) -> None:
        await self.stop()

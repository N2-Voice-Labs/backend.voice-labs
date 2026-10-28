"""Pre-rendered apology (ST-10b): the caller always hears *why* the call is ending.

When a provider fails terminally (``provider_failures``) the TTS may be the very thing that
broke, so the apology is never synthesised live. A short clip per language ships with the
Runtime (``assets/apology/<lang>.wav``, rendered offline by ``scripts/render_apologies.py``)
and is written straight to the transport output, bypassing the pipeline and TTS, before the
session is failed and ended.

This module is pure Python (no Pipecat import) so it can be unit-tested and used without the
voice extras; :meth:`runtime.session.RuntimeSession.play_apology` turns the plans into Pipecat
frames.

Format: 16-bit mono PCM WAV. Clips may be stored at a lower rate than the pipeline output
(:data:`OUTPUT_SAMPLE_RATE`, Pipecat's ``PipelineParams.audio_out_sample_rate`` default, which
no Runtime pipeline overrides) to keep them small; :func:`load_apology` converts them with
linear interpolation. Anything longer than :data:`MAX_CLIP_SECONDS` is truncated.

Who hears what
--------------
* two-person interpreter session: each listener hears the apology in their own language, on
  their own output destination (``translation-lane-N``), so it reaches only them (private
  listen rooms and phone rooms included: the audio router owns those tracks);
* one-way interpreter session: the listener's language (``languages[1]``), main output;
* agent + translator session: the caller's current language, main output;
* anything else, or an unknown language: English.
"""

from __future__ import annotations

import array
import functools
import logging
import sys
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .participants import ParticipantDirectory, normalize_language

_logger = logging.getLogger(__name__)

ASSET_DIR = Path(__file__).parent / "assets" / "apology"
FALLBACK_LANGUAGE = "en"
OUTPUT_SAMPLE_RATE = 24000
# The whole apology (queueing plus playback) is bounded so a broken transport cannot hold
# the session open.
MAX_PLAYBACK_SECONDS = 6.0
MAX_CLIP_SECONDS = 5.5  # longer clips are truncated so clip + margin fits the bound
PLAYBACK_MARGIN_SECONDS = 0.4

# What each clip says. Used by ``scripts/render_apologies.py``; the English text is the
# product wording, the others are its translations.
APOLOGY_TEXTS: dict[str, str] = {
    "en": "Sorry, I'm having trouble right now. Please try again in a moment.",
    "ru": "Извините, у меня сейчас неполадки. Повторите попытку чуть позже.",
    "ko": "죄송합니다. 지금 문제가 발생했습니다. 잠시 후에 다시 시도해 주세요.",
    "es": "Lo siento, tengo problemas en este momento. Por favor, inténtalo de nuevo en un momento.",
    "fr": "Désolé, je rencontre un problème en ce moment. Veuillez réessayer dans un instant.",
    "de": "Entschuldigung, ich habe gerade ein Problem. Bitte versuchen Sie es gleich noch einmal.",
    "ja": "申し訳ありません。問題が発生しています。しばらくしてからお試しください。",
    "zh": "抱歉，我现在遇到了一些问题。请稍后再试。",
    "pt": "Desculpe, estou com problemas agora. Tente novamente em instantes.",
    "it": "Mi dispiace, al momento ho un problema. Per favore, riprova tra un attimo.",
    "nl": "Sorry, ik heb op dit moment problemen. Probeer het zo meteen nog een keer.",
    "hi": "क्षमा करें, अभी मुझे समस्या हो रही है। कृपया बाद में प्रयास करें।",
    "tr": "Üzgünüm, şu anda bir sorun yaşıyorum. Lütfen birazdan tekrar deneyin.",
    "pl": "Przepraszam, mam teraz problem. Spróbuj ponownie za chwilę.",
}


@dataclass(frozen=True)
class ApologyClip:
    """One ready-to-play clip: 16-bit mono PCM at :data:`OUTPUT_SAMPLE_RATE`."""

    language: str
    audio: bytes
    sample_rate: int = OUTPUT_SAMPLE_RATE
    num_channels: int = 1

    @property
    def duration(self) -> float:
        return len(self.audio) / 2 / self.num_channels / self.sample_rate


@dataclass(frozen=True)
class ApologyPlan:
    """One clip for one output: ``destination`` None is the transport's main output."""

    destination: str | None
    clip: ApologyClip


def asset_path(language: str, directory: Path | None = None) -> Path:
    return (directory or ASSET_DIR) / f"{language}.wav"


def available_languages(directory: Path | None = None) -> list[str]:
    """Languages with a shipped clip."""

    folder = directory or ASSET_DIR
    return sorted(path.stem for path in folder.glob("*.wav")) if folder.is_dir() else []


def resample_linear(samples: "array.array[int]", source_rate: int, target_rate: int) -> "array.array[int]":
    """Linear-interpolation resampling of mono 16-bit samples (clips are tiny)."""

    if source_rate == target_rate or not samples:
        return array.array("h", samples)
    count = max(1, round(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    last = len(samples) - 1
    out = array.array("h")
    for index in range(count):
        position = index * ratio
        base = int(position)
        if base >= last:
            out.append(samples[last])
            continue
        fraction = position - base
        out.append(round(samples[base] + (samples[base + 1] - samples[base]) * fraction))
    return out


def read_wav(path: Path, *, target_rate: int = OUTPUT_SAMPLE_RATE) -> tuple[bytes, int]:
    """PCM bytes of a 16-bit mono WAV, converted to ``target_rate`` and capped in length."""

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(f"{path.name}: apology clips must be 16-bit mono PCM")
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) // 2 * 2])
    if sys.byteorder == "big":  # WAV is little-endian
        samples.byteswap()
    samples = samples[: int(MAX_CLIP_SECONDS * rate)]
    samples = resample_linear(samples, rate, target_rate)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes(), target_rate


@functools.lru_cache(maxsize=32)
def _load(language: str, directory: Path | None) -> ApologyClip | None:
    path = asset_path(language, directory)
    if not path.is_file():
        return None
    try:
        audio, rate = read_wav(path)
    except (OSError, EOFError, wave.Error, ValueError):
        _logger.warning("Apology clip %s is unreadable", path.name, exc_info=True)
        return None
    return ApologyClip(language, audio, rate) if audio else None


def load_apology(language: str | None, *, directory: Path | None = None) -> ApologyClip | None:
    """The clip for ``language`` (``"ru-RU"`` and ``"ru"`` alike), else English, else None."""

    code = normalize_language(language)
    for candidate in dict.fromkeys(filter(None, (code, FALLBACK_LANGUAGE))):
        clip = _load(candidate, directory)
        if clip is not None:
            return clip
    return None


def apology_plans(
    config: Any,
    participants: ParticipantDirectory | None = None,
    caller_language: Callable[[], str | None] | None = None,
    *,
    directory: Path | None = None,
) -> list[ApologyPlan]:
    """What to play where for a session running ``config`` (see the module docstring)."""

    targets = _targets(config, participants, caller_language)
    plans: list[ApologyPlan] = []
    for destination, language in targets:
        clip = load_apology(language, directory=directory)
        if clip is not None:
            plans.append(ApologyPlan(destination, clip))
    return plans


def _targets(
    config: Any,
    participants: ParticipantDirectory | None,
    caller_language: Callable[[], str | None] | None,
) -> list[tuple[str | None, str | None]]:
    interpreter = getattr(config, "interpreter", None)
    if getattr(config, "mode", "agent") == "interpreter" and isinstance(interpreter, Mapping):
        directory = participants
        spec = getattr(config, "participants", None)
        if directory is None and spec:
            try:
                directory = ParticipantDirectory.from_mapping(interpreter.get("languages") or (), spec)
            except Exception:
                directory = None
        if directory is not None and len(directory):
            # A listener hears the other person, translated into their own language.
            return [(c.output_destination, c.source_language) for c in directory.contexts()]
        languages = interpreter.get("languages")
        if isinstance(languages, (list, tuple)) and len(languages) == 2:
            return [(None, languages[1])]
        return [(None, None)]
    language = None
    if caller_language is not None:
        try:
            language = caller_language()
        except Exception:
            language = None
    if not language:
        translation = getattr(config, "translation", None)
        callers = translation.get("caller_languages") if isinstance(translation, Mapping) else None
        language = callers[0] if callers else None
    return [(None, language)]


def to_frames(plans: list[ApologyPlan]) -> list[Any]:
    """Pipecat ``OutputAudioRawFrame`` (+ flushing ``TTSStoppedFrame``) per plan (needs the voice extras)."""

    from pipecat.frames.frames import OutputAudioRawFrame, TTSStoppedFrame

    frames = []
    for plan in plans:
        frame = OutputAudioRawFrame(
            audio=plan.clip.audio,
            sample_rate=plan.clip.sample_rate,
            num_channels=plan.clip.num_channels,
        )
        frame.transport_destination = plan.destination
        frame.interruptible = False  # a user talking over it must not cut the apology
        frames.append(frame)
        # The output transport holds back a trailing partial chunk until a TTSStoppedFrame
        # flushes it (padded with silence); without it the last <100 ms would never play.
        stop = TTSStoppedFrame()
        stop.transport_destination = plan.destination
        frames.append(stop)
    return frames

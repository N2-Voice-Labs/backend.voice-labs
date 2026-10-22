"""Render the evaluation manifest to 16 kHz mono WAV with macOS ``say`` (synthetic voices).

    python -m src.evals.generate_audio [--out DIR] [--limit N] [--categories a,b]

Audio is written outside the repository (default ``~/.cache/voice-labs-evals/v1``) together
with a sidecar ``audio_manifest.jsonl`` (id, file, duration, voices, degradation). These
are synthetic voices: good for regression and relative comparison, NOT a substitute for
real consented speakers. Real recordings can be added later by placing ``<id>.wav`` files
(16 kHz mono) in the same folder and keeping the same manifest fields; the runner only
needs the file.

Code-switched items are split by script (Cyrillic -> Russian voice, Latin -> English
voice) and the segments concatenated. Interruption items are rendered as the first
utterance, a short pause, then the interrupting utterance. Noisy/phone items re-use the
clean rendering of ``derived_from`` and degrade it (white noise at a fixed SNR and/or a
300-3400 Hz telephone band, 8 kHz resample and 8-bit mu-law companding). Pure Python
(``array``/``wave``); no numpy needed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
import wave
import zlib
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .dataset import (
    DEFAULT_MANIFEST,
    load_manifest,
    parse_categories,
    select_items,
    validate_manifest,
)

SAMPLE_RATE = 16000
DEFAULT_OUT = Path.home() / ".cache" / "voice-labs-evals" / "v1"
DEFAULT_VOICES = {"en": "Samantha", "ru": "Milena", "ko": "Yuna"}
LEAD_TRAIL_SECONDS = 0.25
SEGMENT_GAP_SECONDS = 0.05
INTERRUPTION_GAP_SECONDS = 0.45

Samples = array  # array("h")


# ---------------------------------------------------------------------------- text


def script_language(char: str) -> str | None:
    """``ko``/``ru``/``en`` for a Hangul/Cyrillic/Latin letter, else ``None``."""

    code = ord(char)
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return "ko"
    if 0x0400 <= code <= 0x04FF:
        return "ru"
    if char.isalpha() and code < 0x250:
        return "en"
    return None


def split_by_script(text: str) -> list[tuple[str, str]]:
    """Split mixed text into ``(language, segment)`` runs; neutral characters (spaces,
    digits, punctuation) stay with the run they follow."""

    runs: list[tuple[str, str]] = []
    pending = ""  # neutral prefix before the first letter
    for char in text:
        language = script_language(char)
        if language is None:
            if runs:
                runs[-1] = (runs[-1][0], runs[-1][1] + char)
            else:
                pending += char
        elif runs and runs[-1][0] == language:
            runs[-1] = (language, runs[-1][1] + char)
        else:
            runs.append((language, pending + char))
            pending = ""
    return [(lang, seg.strip()) for lang, seg in runs if seg.strip()]


def segments_for(item_language: str, text: str) -> list[tuple[str, str]]:
    """``[(language, text)]`` to render: whole text for single-language items, script runs
    for code-switched ones."""

    if "+" in item_language:
        return split_by_script(text)
    return [(item_language, text)]


# --------------------------------------------------------------------------- audio


def read_wav(path: Path) -> tuple[Samples, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        channels = handle.getnchannels()
        rate = handle.getframerate()
        samples = array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    if channels > 1:  # keep the first channel
        samples = samples[::channels]
    return samples, rate


def write_wav(path: Path, samples: Samples, rate: int = SAMPLE_RATE) -> None:
    data = array("h", samples)
    if sys.byteorder == "big":
        data.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def resample_linear(samples: Samples, src_rate: int, dst_rate: int) -> Samples:
    if src_rate == dst_rate or not samples:
        return samples
    ratio = src_rate / dst_rate
    count = int(len(samples) / ratio)
    out = array("h")
    for i in range(count):
        position = i * ratio
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        frac = position - left
        out.append(round(samples[left] * (1 - frac) + samples[right] * frac))
    return out


def silence(seconds: float) -> Samples:
    return array("h", [0]) * int(seconds * SAMPLE_RATE)


def duration_seconds(samples: Samples, rate: int = SAMPLE_RATE) -> float:
    return len(samples) / rate


def _clip(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def rms(samples: Samples) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def add_white_noise(samples: Samples, snr_db: float, seed: int) -> Samples:
    """Add Gaussian white noise so that signal RMS / noise RMS = ``snr_db``."""

    signal = rms(samples)
    if signal == 0:
        return samples
    sigma = signal / (10 ** (snr_db / 20))
    rng = random.Random(seed)
    return array("h", (_clip(s + rng.gauss(0, sigma)) for s in samples))


def _biquad(samples: list[float], kind: str, cutoff: float, rate: int, q: float = 0.7071) -> list[float]:
    """RBJ cookbook second-order low-pass/high-pass filter."""

    w0 = 2 * math.pi * cutoff / rate
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    if kind == "lowpass":
        b0 = b2 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
    elif kind == "highpass":
        b0 = b2 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
    else:
        raise ValueError(kind)
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    x1 = x2 = y1 = y2 = 0.0
    out = []
    for x in samples:
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x
        y2, y1 = y1, y
        out.append(y)
    return out


def _mulaw_roundtrip(value: float, mu: float = 255.0) -> float:
    x = max(-1.0, min(1.0, value / 32768))
    compressed = math.copysign(math.log1p(mu * abs(x)) / math.log1p(mu), x)
    quantized = round(compressed * 127) / 127
    expanded = math.copysign((math.expm1(abs(quantized) * math.log1p(mu))) / mu, quantized)
    return expanded * 32768


def phone_quality(samples: Samples) -> Samples:
    """Telephone line: 300-3400 Hz band, 8 kHz resample (down and back up), mu-law."""

    data = [float(s) for s in samples]
    data = _biquad(data, "highpass", 300, SAMPLE_RATE)
    data = _biquad(data, "highpass", 300, SAMPLE_RATE)
    data = _biquad(data, "lowpass", 3400, SAMPLE_RATE)
    data = _biquad(data, "lowpass", 3400, SAMPLE_RATE)
    narrow = [data[i] for i in range(0, len(data), 2)]  # 8 kHz
    narrow = [_mulaw_roundtrip(v) for v in narrow]
    up: list[float] = []
    for i, value in enumerate(narrow):  # back to 16 kHz, linear interpolation
        nxt = narrow[i + 1] if i + 1 < len(narrow) else value
        up.append(value)
        up.append((value + nxt) / 2)
    return array("h", (_clip(v) for v in up))


def degrade(samples: Samples, degradation: dict[str, Any], seed: int) -> Samples:
    """Apply a manifest ``degradation`` (``type`` noise|phone, optional ``snr_db``)."""

    result = samples
    if degradation.get("type") == "phone":
        result = phone_quality(result)
    snr = degradation.get("snr_db")
    if snr is not None:
        result = add_white_noise(result, float(snr), seed)
    return result


# ---------------------------------------------------------------------------- say


def synthesize_segment(
    text: str,
    voice: str,
    say_bin: str = "say",
    runner: Callable[..., Any] = subprocess.run,
) -> Samples:
    """Render ``text`` with ``say`` into 16 kHz mono samples."""

    with tempfile.TemporaryDirectory(prefix="vl-say-") as tmp:
        wav_path = Path(tmp) / "segment.wav"
        runner(
            [say_bin, "-v", voice, "-o", str(wav_path), f"--data-format=LEI16@{SAMPLE_RATE}", "--", text],
            check=True,
            capture_output=True,
        )
        samples, rate = read_wav(wav_path)
    return resample_linear(samples, rate, SAMPLE_RATE)


def resolve_voice(item: dict[str, Any], language: str, overrides: dict[str, str]) -> str:
    """CLI override > the item's ``tts_voice`` hint (single-language items) > default."""

    if language in overrides:
        return overrides[language]
    hint = item.get("tts_voice", "")
    if hint and "+" not in hint and language == item["source_language"]:
        return hint
    return DEFAULT_VOICES[language]


def render_utterance(
    item: dict[str, Any],
    text: str,
    overrides: dict[str, str],
    say_bin: str,
    runner: Callable[..., Any],
) -> tuple[Samples, list[str]]:
    pieces = array("h")
    voices: list[str] = []
    for index, (language, segment) in enumerate(segments_for(item["source_language"], text)):
        voice = resolve_voice(item, language, overrides)
        if voice not in voices:
            voices.append(voice)
        if index:
            pieces.extend(silence(SEGMENT_GAP_SECONDS))
        pieces.extend(synthesize_segment(segment, voice, say_bin, runner))
    return pieces, voices


def render_clean(
    item: dict[str, Any],
    overrides: dict[str, str],
    say_bin: str,
    runner: Callable[..., Any],
) -> tuple[Samples, list[str]]:
    """Clean rendering of an item (both utterances for interruption items)."""

    samples = array("h")
    samples.extend(silence(LEAD_TRAIL_SECONDS))
    first, voices = render_utterance(item, item["text"], overrides, say_bin, runner)
    samples.extend(first)
    follow = item.get("followup")
    if follow:
        samples.extend(silence(INTERRUPTION_GAP_SECONDS))
        second, more = render_utterance(item, follow["text"], overrides, say_bin, runner)
        samples.extend(second)
        voices += [v for v in more if v not in voices]
    samples.extend(silence(LEAD_TRAIL_SECONDS))
    return samples, voices


def render_item(
    item: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    overrides: dict[str, str],
    say_bin: str = "say",
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[Samples, list[str]]:
    """Final samples for an item, applying its degradation when present."""

    degradation = item.get("degradation")
    base = by_id[item["derived_from"]] if item.get("derived_from") else item
    samples, voices = render_clean(base, overrides, say_bin, runner)
    if degradation:
        seed = zlib.crc32(item["id"].encode("utf-8"))
        samples = degrade(samples, degradation, seed)
    return samples, voices


# ---------------------------------------------------------------------------- CLI


def parse_voice_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        language, _, voice = value.partition("=")
        if language not in DEFAULT_VOICES or not voice:
            raise ValueError(f"--voice expects LANG=VOICE with LANG in en/ru/ko, got {value!r}")
        overrides[language] = voice
    return overrides


def generate(
    items: list[dict[str, Any]],
    all_items: list[dict[str, Any]],
    out_dir: Path,
    *,
    dataset_version: str,
    overrides: dict[str, str] | None = None,
    force: bool = False,
    say_bin: str = "say",
    runner: Callable[..., Any] = subprocess.run,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Render ``items`` into ``out_dir`` and write ``audio_manifest.jsonl``."""

    overrides = overrides or {}
    by_id = {item["id"]: item for item in all_items}
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        path = out_dir / f"{item['id']}.wav"
        voices: list[str]
        if path.exists() and not force:
            samples, rate = read_wav(path)
            voices = []
            note = "cached"
        else:
            samples, voices = render_item(item, by_id, overrides, say_bin, runner)
            write_wav(path, samples)
            rate = SAMPLE_RATE
            note = "rendered"
        log(f"[{index}/{len(items)}] {item['id']} {note} {duration_seconds(samples, rate):.2f}s")
        records.append(
            {
                "id": item["id"],
                "file": path.name,
                "duration_s": round(duration_seconds(samples, rate), 3),
                "sample_rate": rate,
                "voices": voices,
                "degradation": item.get("degradation"),
                "synthetic": True,
                "generator": "macos-say",
                "dataset_version": dataset_version,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    manifest_path = out_dir / "audio_manifest.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = row
    for record in records:
        if record["voices"] == [] and record["id"] in existing:
            record["voices"] = existing[record["id"]].get("voices", [])
        existing[record["id"]] = record
    ordered = [existing[item["id"]] for item in all_items if item["id"] in existing]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def load_audio_manifest(audio_dir: Path) -> dict[str, dict[str, Any]]:
    path = audio_dir / "audio_manifest.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (outside git)")
    parser.add_argument("--categories", help="comma-separated category filter")
    parser.add_argument("--limit", type=int, help="max items per category")
    parser.add_argument("--voice", action="append", default=[], metavar="LANG=VOICE",
                        help="override a voice, e.g. --voice ko=Yuna (repeatable)")
    parser.add_argument("--say-bin", default="say")
    parser.add_argument("--force", action="store_true", help="re-render existing files")
    args = parser.parse_args(argv)

    all_items = load_manifest(args.manifest)
    problems = validate_manifest(all_items)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    items = select_items(all_items, parse_categories(args.categories), args.limit)
    records = generate(
        items,
        all_items,
        args.out.expanduser(),
        dataset_version=args.manifest.parent.name,
        overrides=parse_voice_overrides(args.voice),
        force=args.force,
        say_bin=args.say_bin,
    )
    total = sum(r["duration_s"] for r in records)
    print(f"{len(records)} files, {total / 60:.1f} min -> {args.out.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

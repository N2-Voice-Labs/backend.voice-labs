#!/usr/bin/env python
"""Render the failure-apology clips shipped in ``src/runtime/assets/apology/`` (ST-10b).

The Runtime plays one short clip per language when a provider fails terminally and the call
is about to end (see ``runtime.apology``). The clips are rendered OFFLINE, never at call time:
at that moment the TTS may be the thing that broke. Texts live in
``runtime.apology.APOLOGY_TEXTS``; output is 16-bit mono PCM WAV named ``<lang>.wav``.

Two renderers:

``--macos-placeholders``
    Local and free: macOS ``say`` + ``afconvert``, no network, no credentials. The clips
    currently committed under ``assets/apology/`` are THESE PLACEHOLDERS (system voices, stored
    at 8 kHz to stay under 100 KB each; the Runtime upsamples them to its 24 kHz output).
    Replace them with branded TTS renders (below) before relying on them in production.

``--cartesia --confirm-billable``
    Calls the configured TTS provider (Cartesia ``/tts/bytes``, the API Pipecat's service
    uses) once per language. THIS IS BILLABLE and is refused without ``--confirm-billable``.
    Needs ``CARTESIA_API_KEY`` and a voice: ``--voice-id`` for every language (a multilingual
    Cartesia voice), or ``--voice en=<id> --voice ru=<id> ...`` per language. Output rate
    defaults to the Runtime's 24 kHz. Review every clip by ear before committing it.

Examples::

    python scripts/render_apologies.py --macos-placeholders
    python scripts/render_apologies.py --macos-placeholders --languages en ru ko
    CARTESIA_API_KEY=... python scripts/render_apologies.py --cartesia --confirm-billable \\
        --voice-id <multilingual-voice-id> --languages en ru ko
"""

from __future__ import annotations

import argparse
import array
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runtime.apology import APOLOGY_TEXTS, ASSET_DIR  # noqa: E402

# macOS system voices, one per language (``say -v '?'``).
MACOS_VOICES = {
    "en": "Samantha",
    "ru": "Milena",
    "ko": "Yuna",
    "es": "Mónica",
    "fr": "Thomas",
    "de": "Anna",
    "ja": "Kyoko",
    "zh": "Tingting",
    "pt": "Luciana",
    "it": "Alice",
    "nl": "Xander",
    "hi": "Lekha",
    "tr": "Yelda",
    "pl": "Zosia",
}
PLACEHOLDER_RATE = 8000
SAY_WORDS_PER_MINUTE = 200  # brisk, so a clip stays small
PLACEHOLDER_MAX_BYTES = 100_000
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2026-03-01"  # the version pipecat 1.12's CartesiaHttpTTSService sends
CARTESIA_MODEL = "sonic-3.6"
SILENCE_THRESHOLD = 300  # int16 amplitude


def trim_silence(path: Path, pad_seconds: float = 0.08) -> None:
    """Cut leading and trailing silence in place (keeps the clips small and snappy)."""

    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        samples = array.array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    loud = [i for i, value in enumerate(samples) if abs(value) > SILENCE_THRESHOLD]
    if not loud:
        return
    pad = int(pad_seconds * params.framerate)
    trimmed = samples[max(0, loud[0] - pad) : loud[-1] + pad]
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(trimmed.tobytes())


def render_macos(language: str, destination: Path, rate: int) -> None:
    voice = MACOS_VOICES[language]
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "clip.aiff"
        subprocess.run(
            ["say", "-v", voice, "-r", str(SAY_WORDS_PER_MINUTE), "-o", str(source), APOLOGY_TEXTS[language]], check=True
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{rate}", "-c", "1", str(source), str(destination)],
            check=True,
        )
    trim_silence(destination)


def render_cartesia(language: str, destination: Path, rate: int, voice_id: str, api_key: str) -> None:
    import httpx

    response = httpx.post(
        CARTESIA_URL,
        headers={"Cartesia-Version": CARTESIA_VERSION, "X-API-Key": api_key},
        json={
            "model_id": CARTESIA_MODEL,
            "transcript": APOLOGY_TEXTS[language],
            "voice": {"mode": "id", "id": voice_id},
            "language": language,
            "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": rate},
        },
        timeout=30,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    trim_silence(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--macos-placeholders", action="store_true", help="local say+afconvert placeholders")
    mode.add_argument("--cartesia", action="store_true", help="render with Cartesia (billable)")
    parser.add_argument("--confirm-billable", action="store_true", help="required for --cartesia")
    parser.add_argument("--languages", nargs="+", default=list(APOLOGY_TEXTS), choices=sorted(APOLOGY_TEXTS))
    parser.add_argument("--voice-id", help="Cartesia voice used for every language")
    parser.add_argument("--voice", action="append", default=[], metavar="LANG=ID", help="Cartesia voice per language")
    parser.add_argument("--rate", type=int, help="output sample rate (8000 placeholders, 24000 Cartesia)")
    parser.add_argument("--out", type=Path, default=ASSET_DIR, help="output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.macos_placeholders:
        rate = args.rate or PLACEHOLDER_RATE
        for language in args.languages:
            destination = args.out / f"{language}.wav"
            render_macos(language, destination, rate)
            size = destination.stat().st_size
            flag = "" if size <= PLACEHOLDER_MAX_BYTES else "  (over 100 KB: lower --rate)"
            print(f"{destination.name}: {size} bytes{flag}")
        return 0
    if not args.confirm_billable:
        print("--cartesia calls the TTS provider and is billable; pass --confirm-billable.", file=sys.stderr)
        return 2
    api_key = os.environ.get("CARTESIA_API_KEY", "")
    voices = dict(item.split("=", 1) for item in args.voice)
    if not api_key:
        print("CARTESIA_API_KEY is not set.", file=sys.stderr)
        return 2
    rate = args.rate or 24000
    for language in args.languages:
        voice_id = voices.get(language) or args.voice_id
        if not voice_id:
            print(f"No voice for {language}: pass --voice-id or --voice {language}=<id>.", file=sys.stderr)
            return 2
        destination = args.out / f"{language}.wav"
        render_cartesia(language, destination, rate, voice_id, api_key)
        print(f"{destination.name}: {destination.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

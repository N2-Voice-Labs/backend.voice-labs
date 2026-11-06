#!/usr/bin/env python
"""Real-provider smoke test of the ElevenLabs STT + Gemini translation + ElevenLabs voice stack.

BILLABLE but tiny (a few hundred ElevenLabs characters, a few Gemini requests). It needs no
LiveKit and no browser. For each direction (ko to en, en to ko) it runs the chain a
translated call runs:

    speech (made with ElevenLabs TTS) -> ElevenLabs realtime STT -> Gemini translation
    (the Runtime's own ``OpenAITranslator``) -> ElevenLabs TTS -> ElevenLabs realtime STT

and prints what each stage returned with its timing. The last STT pass is the check that the
spoken translation is intelligible. It judges nothing about quality: read the text.

Environment (values are never printed):
  ELEVENLABS_API_KEY, GEMINI_API_KEY, GEMINI_TRANSLATION_MODEL
  ELEVENLABS_VOICE_ID (optional, default: a premade voice)
  GEMINI_BASE_URL (optional)
``--env-file PATH`` loads KEY=VALUE lines from a file first (default: ``../.env`` when present;
variables already in the environment win).

    cd runtime && python scripts/stack_smoke.py --confirm-billable
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import websockets  # noqa: E402

API = "https://api.elevenlabs.io"
DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"
STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v2_realtime")
TTS_MODEL = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
LANGUAGES = {"ko": ("kor", "Korean"), "en": ("eng", "English")}
SENTENCES = {
    "ko": "죄송하지만 그 계약서의 마지막 조항을 다시 한번 설명해 주시겠어요?",
    "en": "Could you please send me the updated schedule before the meeting tomorrow?",
}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def need(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        sys.exit(f"missing environment: {', '.join(missing)}")


async def tts(client: httpx.AsyncClient, text: str, language: str) -> tuple[bytes, float]:
    voice = os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE
    started = time.monotonic()
    response = await client.post(
        f"{API}/v1/text-to-speech/{voice}",
        params={"output_format": "pcm_16000"},
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        json={"text": text, "model_id": TTS_MODEL, "language_code": language},
    )
    response.raise_for_status()
    return response.content, time.monotonic() - started


async def stt(pcm: bytes, language_code: str) -> tuple[str, float]:
    """Stream ``pcm`` (16 kHz mono) to the realtime STT, commit, return (text, commit to final)."""

    url = (
        f"wss://{API.removeprefix('https://').removeprefix('http://')}/v1/speech-to-text/realtime"
        f"?model_id={STT_MODEL}&language_code={language_code}"
        "&audio_format=pcm_16000&commit_strategy=manual"
    )
    headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
    async with websockets.connect(url, additional_headers=headers) as ws:
        for offset in range(0, len(pcm), 3200):  # 100 ms chunks
            chunk = base64.b64encode(pcm[offset : offset + 3200]).decode()
            await ws.send(json.dumps({
                "message_type": "input_audio_chunk", "audio_base_64": chunk,
                "commit": False, "sample_rate": 16000,
            }))
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({
            "message_type": "input_audio_chunk", "audio_base_64": "",
            "commit": True, "sample_rate": 16000,
        }))
        committed_at = time.monotonic()
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), 20))
            kind = message.get("message_type")
            if kind == "committed_transcript":
                return message.get("text", ""), time.monotonic() - committed_at
            if kind not in ("session_started", "partial_transcript"):
                raise RuntimeError(f"STT returned {kind}: {message.get('error', '')}")


async def translate(text: str, source: str, target: str) -> tuple[str, float, float]:
    from src.runtime.interpreter import OpenAITranslator
    from src.runtime.speech_providers import gemini_base_url

    translator = OpenAITranslator(
        api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ["GEMINI_TRANSLATION_MODEL"],
        base_url=gemini_base_url(),
    )
    await translator.warm_up()  # as the pipeline does at call setup
    messages = [
        {"role": "system", "content": (
            f"Translate the user's {LANGUAGES[source][1]} text to {LANGUAGES[target][1]}. "
            "Output only the translation.")},
        {"role": "user", "content": text},
    ]
    started, first, parts = time.monotonic(), None, []
    async for delta in translator(messages):
        first = first if first is not None else time.monotonic() - started
        parts.append(delta)
    return "".join(parts), first or 0.0, time.monotonic() - started


async def direction(client: httpx.AsyncClient, source: str, target: str) -> bool:
    source_code, target_code = LANGUAGES[source][0], LANGUAGES[target][0]
    print(f"\n== {LANGUAGES[source][1]} -> {LANGUAGES[target][1]}")
    spoken, took = await tts(client, SENTENCES[source], source)
    print(f"speaker (TTS {source}, {len(SENTENCES[source])} chars, {len(spoken) / 32000:.1f}s audio, {took:.2f}s): {SENTENCES[source]}")
    heard, took = await stt(spoken, source_code)
    print(f"1 STT {source_code}  commit->final {took:.2f}s: {heard}")
    translated, first, total = await translate(heard, source, target)
    print(f"2 Gemini first text {first:.2f}s, total {total:.2f}s: {translated}")
    audio, took = await tts(client, translated, target)
    print(f"3 TTS {target}  ({len(translated)} chars, {len(audio) / 32000:.1f}s audio, {took:.2f}s)")
    back, took = await stt(audio, target_code)
    print(f"4 STT {target_code} of the spoken translation  commit->final {took:.2f}s: {back}")
    return bool(heard and translated and back)


async def main() -> int:
    async with httpx.AsyncClient(timeout=60) as client:
        results = [await direction(client, "ko", "en"), await direction(client, "en", "ko")]
    print("\nresult:", "OK" if all(results) else "SOMETHING RETURNED NO TEXT")
    return 0 if all(results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-billable", action="store_true", help="required: this spends a few cents of quota")
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    args = parser.parse_args()
    if not args.confirm_billable:
        sys.exit("refusing to run without --confirm-billable")
    load_env_file(args.env_file)
    need("ELEVENLABS_API_KEY", "GEMINI_API_KEY", "GEMINI_TRANSLATION_MODEL")
    sys.exit(asyncio.run(main()))

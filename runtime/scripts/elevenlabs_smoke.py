#!/usr/bin/env python
"""Real-provider smoke test for the ElevenLabs integration. BILLABLE (a few hundred characters).

Checks, against the REAL ElevenLabs API, what the mocked tests cannot:

 1. the key works (subscription) and which voices it can use;
 2. REST preview (``runtime.tts_preview``) in en/ru/ko, with ``language_code`` on Flash;
 3. whether ``pcm_24000`` responses are headerless raw PCM (the docs do not say);
 4. ``eleven_multilingual_v2`` accepts a request without ``language_code``;
 5. the live Pipecat ``ElevenLabsTTSService`` over the real websocket: first-audio latency,
    sample rate, a voice + language switch, an interruption, a second utterance on the same
    connection, and clean shutdown;
 6. a rejected key fails fast (REST 401 and websocket handshake) instead of hanging.

It makes only objective measurements (durations, levels, timings). It does NOT judge audio
quality: listen to the saved files (``--out-dir``) for that.

Credentials (never printed, never written to disk by this script), first match wins:
  * the ``ELEVENLABS_API_KEY`` environment variable;
  * the file named by ``--key-file`` or ``ELEVENLABS_API_KEY_FILE``;
  * ``~/Developer/11labs-api-key.txt``.
To rotate the key, replace the file's contents or export a new ``ELEVENLABS_API_KEY``.

    cd runtime && python scripts/elevenlabs_smoke.py --confirm-billable --out-dir /tmp/el-smoke
"""

from __future__ import annotations

import argparse
import array
import asyncio
import io
import json
import math
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

DEFAULT_KEY_FILE = Path.home() / "Developer" / "11labs-api-key.txt"
API = "https://api.elevenlabs.io"
TEXTS = {
    "en": "Hello, this is a short test.",
    "ru": "Здравствуйте, это короткая проверка.",
    "ko": "안녕하세요, 짧은 테스트입니다.",
}
LONG_EN = (
    "This is a deliberately longer sentence so that there is still audio being generated "
    "when the interruption arrives, and then even more words follow after it."
)
TIMEOUT = 40.0

_secret = ""


def redact(value: object) -> str:
    text = str(value)
    return text.replace(_secret, "***") if _secret else text


def load_key(key_file: str | None) -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if key:
        return key
    path = Path(key_file or os.environ.get("ELEVENLABS_API_KEY_FILE") or DEFAULT_KEY_FILE)
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def analyse_pcm(pcm: bytes, rate: int) -> dict:
    """Objective numbers about 16-bit mono PCM; says nothing about how it sounds."""

    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return {"seconds": 0.0, "rms": 0.0, "peak": 0, "silent_fraction": 1.0}
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    quiet = sum(1 for s in samples if abs(s) < 300)
    return {
        "seconds": round(len(samples) / rate, 2),
        "rms": round(rms, 1),
        "peak": max(abs(s) for s in samples),
        "silent_fraction": round(quiet / len(samples), 2),
    }


def audible(stats: dict, lo: float = 0.4, hi: float = 12.0) -> bool:
    return lo <= stats["seconds"] <= hi and stats["rms"] > 150 and stats["silent_fraction"] < 0.95


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool, detail: object, seconds: float | None = None) -> None:
        row = {"check": name, "ok": bool(ok), "detail": redact(detail)}
        if seconds is not None:
            row["seconds"] = round(seconds, 2)
        self.rows.append(row)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {row['detail']}", flush=True)


def save(out_dir: Path | None, name: str, pcm: bytes, rate: int) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_dir / name), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def rest_checks(report: Report, key: str, voice: str, out_dir: Path | None) -> None:
    from runtime import tts_preview

    for language, text in TEXTS.items():
        started = time.monotonic()
        try:
            wav = asyncio.run(
                tts_preview.synthesize_elevenlabs_preview(
                    api_key=key, provider_voice_id=voice, text=text, language=language
                )
            )
            with wave.open(io.BytesIO(wav), "rb") as handle:
                rate = handle.getframerate()
                pcm = handle.readframes(handle.getnframes())
            stats = analyse_pcm(pcm, rate)
            save(out_dir, f"rest-{language}.wav", pcm, rate)
            report.add(f"REST preview {language} (flash_v2_5, language_code)", audible(stats, 0.3), stats, time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001
            report.add(f"REST preview {language}", False, f"{type(exc).__name__}: {exc}")

    # Is a pcm_24000 body headerless? (the docs do not say)
    try:
        response = httpx.post(
            f"{API}/v1/text-to-speech/{voice}",
            params={"output_format": "pcm_24000"},
            headers={"xi-api-key": key},
            json={"text": TEXTS["en"], "model_id": "eleven_flash_v2_5"},
            timeout=TIMEOUT,
        )
        body = response.content
        headerless = response.status_code == 200 and body[:4] != b"RIFF"
        stats = analyse_pcm(body, 24000)
        report.add(
            "pcm_24000 body is headerless raw PCM",
            headerless and audible(stats, 0.3),
            {"status": response.status_code, "first4": list(body[:4]), "bytes": len(body), **stats},
        )
    except Exception as exc:  # noqa: BLE001
        report.add("pcm_24000 body is headerless raw PCM", False, f"{type(exc).__name__}: {exc}")

    # multilingual_v2 must work without a language_code (we never send one for it)
    try:
        wav = asyncio.run(
            tts_preview.synthesize_elevenlabs_preview(
                api_key=key,
                provider_voice_id=voice,
                text=TEXTS["ko"],
                language="ko",
                model="eleven_multilingual_v2",
            )
        )
        with wave.open(io.BytesIO(wav), "rb") as handle:
            stats = analyse_pcm(handle.readframes(handle.getnframes()), handle.getframerate())
        report.add("REST eleven_multilingual_v2 ko (no language_code)", audible(stats, 0.3), stats)
    except Exception as exc:  # noqa: BLE001
        report.add("REST eleven_multilingual_v2 ko", False, f"{type(exc).__name__}: {exc}")


async def live_checks(report: Report, key: str, voice_a: str, voice_b: str, out_dir: Path | None) -> None:
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        EndFrame,
        ErrorFrame,
        InterruptionFrame,
        TTSAudioRawFrame,
        TTSSpeakFrame,
        TTSUpdateSettingsFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.services.settings import TTSSettings
    from pipecat.workers.runner import WorkerRunner

    from runtime.pipeline import PipelineConfig
    from runtime.speech_providers import build_tts

    config = PipelineConfig(
        livekit_url="ws://unused",
        livekit_token="unused",
        room_name="smoke",
        deepgram_api_key="unused",
        openai_api_key="unused",
        cartesia_api_key="",
        cartesia_voice_id="",
        system_instruction="",
        providers={"stt": "deepgram", "llm": "openai-compatible", "tts": "elevenlabs"},
        elevenlabs_api_key=key,
        elevenlabs_voice_id=voice_a,
    )

    class Sink(FrameProcessor):
        def __init__(self) -> None:
            super().__init__(name="Sink")
            self.audio: list[tuple[float, int, bytes]] = []
            self.errors: list[str] = []

        async def process_frame(self, frame, direction) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSAudioRawFrame):
                self.audio.append((time.monotonic(), frame.sample_rate, bytes(frame.audio)))
            if isinstance(frame, ErrorFrame):
                self.errors.append(redact(frame.error))
            await self.push_frame(frame, direction)
            if type(frame).__name__ == "TTSStoppedFrame":
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    service = build_tts(config, "en")
    sink = Sink()
    worker = PipelineWorker(
        Pipeline([service, sink]),
        params=PipelineParams(enable_metrics=False),
        cancel_on_idle_timeout=False,
    )
    runner = WorkerRunner()
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(*_):
        started.set()

    await runner.add_workers(worker)
    run = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(started.wait(), TIMEOUT)

        async def speak(text: str, wait: float = 12.0) -> tuple[float | None, bytes, int]:
            """Queue text; return (seconds to first audio, audio bytes, sample rate)."""
            mark = len(sink.audio)
            t0 = time.monotonic()
            await worker.queue_frame(TTSSpeakFrame(text, append_to_context=False))
            deadline = t0 + wait
            quiet_since = None
            while time.monotonic() < deadline:
                await asyncio.sleep(0.1)
                if len(sink.audio) > mark:
                    if quiet_since is None or len(sink.audio) != getattr(speak, "last", -1):
                        quiet_since = time.monotonic()
                    speak.last = len(sink.audio)  # type: ignore[attr-defined]
                    if time.monotonic() - quiet_since > 1.5:
                        break
            frames = sink.audio[mark:]
            if not frames:
                return None, b"", 0
            return frames[0][0] - t0, b"".join(f[2] for f in frames), frames[0][1]

        # 1. English with the default voice
        ttfa, pcm, rate = await speak(TEXTS["en"])
        stats = analyse_pcm(pcm, rate or 24000)
        save(out_dir, "live-en-voiceA.wav", pcm, rate or 24000)
        report.add("live websocket en, voice A", ttfa is not None and audible(stats, 0.3),
                   {"first_audio_s": None if ttfa is None else round(ttfa, 2), "sample_rate": rate, **stats})

        # 2. switch voice + language (URL fields => reconnect)
        await worker.queue_frame(TTSUpdateSettingsFrame(delta=TTSSettings(voice=voice_b, language="ru")))
        await asyncio.sleep(0.3)
        ttfa, pcm, rate = await speak(TEXTS["ru"])
        stats = analyse_pcm(pcm, rate or 24000)
        save(out_dir, "live-ru-voiceB.wav", pcm, rate or 24000)
        report.add("live voice+language switch (voice B, ru)", ttfa is not None and audible(stats, 0.3),
                   {"first_audio_s_including_reconnect": None if ttfa is None else round(ttfa, 2), **stats})

        # 3. interruption while synthesis is genuinely IN FLIGHT. A complete sentence is generated
        #    in ~0.3 s, i.e. before any interruption could arrive, so keep the LLM turn open
        #    (start + a text fragment, no end frame): the ElevenLabs context stays open until
        #    the interruption closes it (close_context).
        from pipecat.frames.frames import LLMFullResponseStartFrame, TextFrame

        mark = len(sink.audio)
        await worker.queue_frame(LLMFullResponseStartFrame())
        await worker.queue_frame(TextFrame("This answer is still being written. And then there is more"))
        wait_until = time.monotonic() + 10
        while len(sink.audio) == mark and time.monotonic() < wait_until:
            await asyncio.sleep(0.05)
        got_audio = len(sink.audio) > mark
        await asyncio.sleep(0.2)
        t_int = time.monotonic()
        before = len(sink.audio)
        await worker.queue_frame(InterruptionFrame())
        await asyncio.sleep(2.5)
        late = [f for f in sink.audio[before:] if f[0] > t_int + 0.5]
        report.add("interruption of an OPEN context: audio had started, none >0.5 s after InterruptionFrame",
                   got_audio and not late,
                   {"audio_started_before_interrupt": got_audio, "frames_before_interrupt": before - mark,
                    "frames_after_0.5s": len(late), "sink_errors": sink.errors})
        ttfa, pcm, rate = await speak("Next.")
        stats = analyse_pcm(pcm, rate or 24000)
        report.add("new utterance works after interruption", ttfa is not None and audible(stats, 0.1),
                   {"first_audio_s": None if ttfa is None else round(ttfa, 2), **stats})
    except Exception as exc:  # noqa: BLE001
        report.add("live websocket checks", False, f"{type(exc).__name__}: {exc}")
    finally:
        await worker.queue_frame(EndFrame())
        try:
            await asyncio.wait_for(run, TIMEOUT)
            leftover = getattr(service, "_websocket", None)
            report.add("clean shutdown (EndFrame ends the run, socket cleared)", leftover is None, f"websocket={leftover!r}")
        except Exception as exc:  # noqa: BLE001
            report.add("clean shutdown", False, f"{type(exc).__name__}: {exc}")


async def live_bad_key(report: Report, voice: str) -> None:
    """What the REAL service emits for an invalid key, and how Voice Labs classifies it."""

    from pipecat.frames.frames import EndFrame, ErrorFrame, TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.workers.runner import WorkerRunner

    from runtime.pipeline import PipelineConfig
    from runtime.provider_failures import classify_error
    from runtime.speech_providers import build_tts

    bogus = "invalid-key-used-only-to-test-rejection"
    config = PipelineConfig(
        livekit_url="ws://unused", livekit_token="unused", room_name="smoke",
        deepgram_api_key="unused", openai_api_key="unused", cartesia_api_key="", cartesia_voice_id="",
        system_instruction="", providers={"stt": "deepgram", "llm": "openai-compatible", "tts": "elevenlabs"},
        elevenlabs_api_key=bogus, elevenlabs_voice_id=voice,
    )

    class Catch(FrameProcessor):
        def __init__(self) -> None:
            super().__init__(name="Catch")
            self.errors: list[ErrorFrame] = []
            self.audio = 0

        async def process_frame(self, frame, direction) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, ErrorFrame):
                self.errors.append(frame)
            if type(frame).__name__ == "TTSAudioRawFrame":
                self.audio += 1
            await self.push_frame(frame, direction)

    catch = Catch()
    service = build_tts(config, "en")
    # ErrorFrames are pushed UPSTREAM by the service, so the catcher goes in front of it.
    worker = PipelineWorker(Pipeline([catch, service]), params=PipelineParams(enable_metrics=False), cancel_on_idle_timeout=False)
    runner = WorkerRunner()
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(*_):
        started.set()

    await runner.add_workers(worker)
    run = asyncio.create_task(runner.run())
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(started.wait(), TIMEOUT)
        await worker.queue_frame(TTSSpeakFrame("x", append_to_context=False))
        # Pipecat retries a closed socket a few times before giving up, so collect for a while.
        while time.monotonic() - t0 < 12:
            await asyncio.sleep(0.1)
            if any(getattr(f, "fatal", False) for f in catch.errors):
                break
        if catch.errors:
            seen = []
            for frame in catch.errors:
                exc = getattr(frame, "exception", None)
                seen.append({
                    "t": round(time.monotonic() - t0, 2),
                    "classified_as": classify_error(
                        exception=exc, text=str(frame.error), category=getattr(frame, "category", None)
                    ),
                    "fatal": getattr(frame, "fatal", None),
                    "exception": f"{type(exc).__name__}: {redact(exc)}"[:160] if exc is not None else None,
                    "text": redact(str(frame.error))[:160],
                })
            report.add(
                "live invalid key: at least one emitted error classifies as auth",
                any(row["classified_as"] == "auth" for row in seen),
                {"errors": seen, "audio_frames": catch.audio},
            )
        else:
            report.add("live invalid key: at least one emitted error classifies as auth", False,
                       {"outcome": "no ErrorFrame within 15 s (silent hang?)", "audio_frames": catch.audio})
    except Exception as exc:  # noqa: BLE001
        report.add("live invalid key", False, f"{type(exc).__name__}: {exc}")
    finally:
        await worker.queue_frame(EndFrame())
        try:
            await asyncio.wait_for(run, TIMEOUT)
        except Exception:  # noqa: BLE001
            pass


async def bad_key_checks(report: Report, voice: str) -> None:
    bogus = "invalid-key-used-only-to-test-rejection"
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{API}/v1/text-to-speech/{voice}",
                headers={"xi-api-key": bogus},
                json={"text": "x", "model_id": "eleven_flash_v2_5"},
            )
        report.add("REST rejects an invalid key quickly", response.status_code in (401, 403), {"status": response.status_code}, time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        report.add("REST rejects an invalid key quickly", False, f"{type(exc).__name__}: {exc}")
    t0 = time.monotonic()
    try:
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus

        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice}/multi-stream-input?model_id=eleven_flash_v2_5"
        try:
            async with connect(url, additional_headers={"xi-api-key": bogus}, open_timeout=TIMEOUT):
                report.add("websocket handshake with an invalid key (INFO: accepted means auth fails in-band)", True, "connection was accepted; the key is checked after connect")
        except InvalidStatus as rejected:
            code = rejected.response.status_code
            report.add("websocket handshake with an invalid key (INFO: rejected at handshake)", True, {"status": code}, time.monotonic() - t0)
    except Exception as exc:  # noqa: BLE001
        report.add("websocket rejects an invalid key at the handshake", False, f"{type(exc).__name__}: {exc}")


def pick_voices(report: Report, key: str, wanted: list[str]) -> list[str]:
    headers = {"xi-api-key": key}
    try:
        sub = httpx.get(f"{API}/v1/user/subscription", headers=headers, timeout=TIMEOUT)
        body = sub.json() if sub.status_code == 200 else {}
        report.add(
            "key works (subscription)",
            sub.status_code == 200,
            {"status": sub.status_code, "tier": body.get("tier"),
             "characters_used": body.get("character_count"), "character_limit": body.get("character_limit")},
        )
    except Exception as exc:  # noqa: BLE001
        report.add("key works (subscription)", False, f"{type(exc).__name__}: {exc}")
    if wanted:
        return wanted
    try:
        response = httpx.get(f"{API}/v1/voices", headers=headers, timeout=TIMEOUT)
        voices = response.json().get("voices", []) if response.status_code == 200 else []
        premade = [v for v in voices if v.get("category") == "premade"] or voices
        chosen = [v["voice_id"] for v in premade[:2]]
        report.add("voices available to the key", len(chosen) >= 1,
                   {"status": response.status_code, "total": len(voices),
                    "chosen": [(v.get("name"), v["voice_id"]) for v in premade[:2]]})
        return chosen
    except Exception as exc:  # noqa: BLE001
        report.add("voices available to the key", False, f"{type(exc).__name__}: {exc}")
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm-billable", action="store_true", help="required: this calls ElevenLabs")
    parser.add_argument("--key-file", help=f"file holding the API key (default {DEFAULT_KEY_FILE})")
    parser.add_argument("--voice-id", action="append", default=[], help="voice id (repeat for a second voice)")
    parser.add_argument("--out-dir", type=Path, help="save the audio samples here to listen to them")
    parser.add_argument("--report", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)
    if not args.confirm_billable:
        print("This calls the ElevenLabs API and is billable; pass --confirm-billable.", file=sys.stderr)
        return 2
    global _secret
    _secret = load_key(args.key_file)
    if not _secret:
        print("No ElevenLabs key: set ELEVENLABS_API_KEY or put it in the key file.", file=sys.stderr)
        return 2
    import re

    for voice in args.voice_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", voice):
            print("Unsafe --voice-id.", file=sys.stderr)
            return 2

    report = Report()
    voices = pick_voices(report, _secret, list(args.voice_id))
    if not voices:
        print("No usable voice; pass --voice-id.", file=sys.stderr)
        return 1
    voice_a, voice_b = voices[0], voices[1] if len(voices) > 1 else voices[0]
    rest_checks(report, _secret, voice_a, args.out_dir)
    asyncio.run(live_checks(report, _secret, voice_a, voice_b, args.out_dir))
    asyncio.run(bad_key_checks(report, voice_a))
    asyncio.run(live_bad_key(report, voice_a))

    failed = [r for r in report.rows if not r["ok"]]
    summary = {"passed": len(report.rows) - len(failed), "failed": len(failed), "rows": report.rows,
               "note": "Objective measurements only; audio quality was not judged."}
    if args.report:
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n{summary['passed']} passed, {summary['failed']} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

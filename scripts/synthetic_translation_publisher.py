#!/usr/bin/env python
"""A stand-in for the Runtime's per-listener output, for validating browser-side latency (no providers, no credits).

    LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... python scripts/synthetic_translation_publisher.py spec.json

``spec.json`` lists one entry per listener: the private listen room, the track name (``translation-for-<identity>``) and the
tones to play: ``[{"room": "...", "track": "translation-for-guest-1", "tones": [[start_s, duration_s], ...]}]``.
Like the Runtime's TTS output the track carries audio continuously: silence between the tones, never a gap in the frames.
It prints one JSON line per tone with ``time.time()`` (epoch seconds) of the moment the tone's first frame was handed to
LiveKit, which a browser on the SAME host can compare with ``performance.timeOrigin + at``. Run by ``browser_latency_check.cjs``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time

from livekit import api, rtc

RATE = 24000
FRAME_MS = 20


def token(room: str, identity: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=False))
        .to_jwt()
    )


def frame(samples: list[int]) -> rtc.AudioFrame:
    data = b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples)
    return rtc.AudioFrame(data, RATE, 1, len(samples))


async def play(entry: dict, start_at: float, report: list) -> None:
    room = rtc.Room()
    await room.connect(os.environ["LIVEKIT_URL"], token(entry["room"], "voice-labs-synthetic"))
    source = rtc.AudioSource(RATE, 1, queue_size_ms=1000)
    track = rtc.LocalAudioTrack.create_audio_track(entry["track"], source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, options)
    per_frame = RATE * FRAME_MS // 1000
    tones = sorted(tuple(t) for t in entry["tones"])
    end = max(s + d for s, d in tones) + 1.0
    phase = 0.0
    t = 0.0
    while time.monotonic() < start_at:
        await source.capture_frame(frame([0] * per_frame))  # silence until the shared start, like an idle TTS track
        await asyncio.sleep(FRAME_MS / 1000 * 0.9)
    base = time.monotonic()
    while t < end:
        inside = next(((s, d) for s, d in tones if s <= t < s + d), None)
        samples = []
        for i in range(per_frame):
            if inside is not None:
                phase += 2 * math.pi * 440 / RATE
                samples.append(int(9000 * math.sin(phase)))
            else:
                samples.append(0)
        if inside is not None and abs(t - inside[0]) < FRAME_MS / 1000 / 2:
            report.append({"room": entry["room"], "track": entry["track"], "tone_start_s": inside[0], "duration_s": inside[1], "handed_to_livekit_epoch_s": time.time()})
        await source.capture_frame(frame(samples))
        t += FRAME_MS / 1000
        delay = base + t - time.monotonic() - 0.2  # stay about 200 ms ahead of real time, like a TTS burst
        if delay > 0:
            await asyncio.sleep(delay)
    await source.wait_for_playout()
    await room.disconnect()


async def main(spec: list[dict]) -> None:
    report: list[dict] = []
    start_at = time.monotonic() + 3.0
    await asyncio.gather(*(play(entry, start_at, report) for entry in spec))
    for line in sorted(report, key=lambda r: r["handed_to_livekit_epoch_s"]):
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.environ.get(name):
            sys.exit(f"missing environment: {name}")
    asyncio.run(main(json.loads(open(sys.argv[1]).read())))

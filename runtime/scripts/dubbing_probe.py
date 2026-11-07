#!/usr/bin/env python
"""Latency the dubbing mixer adds to the translation, through LiveKit. Free: tones only, no providers.

    cd runtime && python -u scripts/dubbing_probe.py [--rounds 16] [--out bench/dubbing]

Two LiveKit clients join a throwaway room, as in ``transport_probe.py``. The sender publishes a
translation tone (loud) three ways, and the receiver times its first loud frame:

  direct            ``AudioSource.capture_frame`` in 40 ms chunks, as the Runtime does today
  mixer, idle       ``DubbingChannel``: only the translation, nothing else playing
  mixer, original   ``DubbingChannel`` with the speaker's voice (quiet tone) already playing

All three run on the same connection in rotation, so network drift hits them equally. The
added latency is the difference between the medians. Credentials come from the environment
(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET); values are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics as stats
import sys
import time
import uuid
from pathlib import Path

from livekit import rtc

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transport_probe import Receiver, token, tone  # noqa: E402

from runtime.dubbing import PROFILES, DubbingChannel  # noqa: E402

RATE = 24000


def quiet_tone(ms: int, amplitude: int = 2500, hz: int = 300) -> bytes:
    import math
    import struct

    n = RATE * ms // 1000
    return b"".join(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * hz * i / RATE))) for i in range(n))


async def main(args: argparse.Namespace) -> None:
    url = os.environ["LIVEKIT_URL"]
    room_name = f"dub-probe-{uuid.uuid4().hex[:8]}"
    a, b = rtc.Room(), rtc.Room()
    await a.connect(url, token("probe-a", room_name))
    await b.connect(url, token("probe-b", room_name))
    receiver = Receiver(b)
    source = rtc.AudioSource(RATE, 1, queue_size_ms=1000)
    track = rtc.LocalAudioTrack.create_audio_track("dub-probe", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    publication = await a.local_participant.publish_track(track, options)
    await asyncio.sleep(2.0)

    translation = tone(RATE, 400)  # loud tone: the receiver's "arrived" test
    step = RATE * 40 // 1000 * 2
    chunks = [translation[i : i + step] for i in range(0, len(translation), step)]

    def make_frame(pcm: bytes) -> rtc.AudioFrame:
        return rtc.AudioFrame(pcm, RATE, 1, len(pcm) // 2)

    results: dict[str, list[float]] = {"direct": [], "mixer, idle": [], "mixer, original": []}
    cpu: dict[str, float] = {k: 0.0 for k in results}
    for round_number in range(args.rounds):
        for name in results:
            await asyncio.sleep(1.5)
            channel = DubbingChannel(source, PROFILES["balanced"], make_frame=make_frame)
            feeder = None
            if name == "mixer, original":
                async def feed() -> None:  # the speaker's voice, in real time, as LiveKit delivers it
                    for _ in range(100):
                        channel.push_original(quiet_tone(20))
                        await asyncio.sleep(0.02)

                feeder = asyncio.get_running_loop().create_task(feed())
                await asyncio.sleep(0.6)  # past the original's delay: it is playing
            receiver.arm()
            cpu_before = time.process_time()
            started = time.monotonic()
            if name == "direct":
                for chunk in chunks:
                    await source.capture_frame(make_frame(chunk))
            else:
                channel.push_translation(translation)
            await asyncio.sleep(1.0)
            cpu[name] += time.process_time() - cpu_before
            if receiver.first_loud is not None:
                results[name].append((receiver.first_loud - started) * 1000)
            if feeder is not None:
                feeder.cancel()
                await asyncio.gather(feeder, return_exceptions=True)
            await channel.aclose()
            await source.wait_for_playout()
    await a.local_participant.unpublish_track(publication.sid)
    await source.aclose()
    await a.disconnect()
    await b.disconnect()

    print(f"{'path':18}{'n':>4}{'P50':>8}{'P90':>8}{'min':>8}{'max':>8}")
    for name, values in results.items():
        if values:
            ordered = sorted(values)
            print(f"{name:18}{len(values):>4}{stats.median(values):>8.1f}{ordered[int(0.9 * (len(ordered) - 1))]:>8.1f}{min(values):>8.1f}{max(values):>8.1f}")
        else:
            print(f"{name:18}: nothing heard")
    base = stats.median(results["direct"])
    for name in ("mixer, idle", "mixer, original"):
        if results[name]:
            print(f"added by {name}: {stats.median(results[name]) - base:+.1f} ms (median)")
    for name, seconds in cpu.items():
        print(f"{name:18} process CPU per round {seconds / args.rounds * 1000:.0f} ms (sender and receiver share this process)")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dubbing_probe.json").write_text(json.dumps({"latency_ms": results, "cpu_s_total": cpu}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--out", default=None)
    parsed = parser.parse_args()
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.environ.get(name):
            sys.exit(f"missing environment: {name}")
    asyncio.run(main(parsed))

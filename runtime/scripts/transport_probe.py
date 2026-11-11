#!/usr/bin/env python
"""Audio transport latency through LiveKit, without any speech provider (no ElevenLabs credits).

    cd runtime && python scripts/transport_probe.py [--rounds 12] [--out bench/transport]

Two LiveKit clients join a throwaway room. One publishes a short tone through an
``rtc.AudioSource`` set up like the Runtime's output (24 kHz, 40 ms chunks, 1000 ms queue,
the same ``TrackPublishOptions``), the other subscribes and times the first loud frame. Both use
``time.monotonic()``; they share one process, so there is one clock and no offset to correct.

Reported per configuration (milliseconds, the sender's first ``capture_frame`` call to the
receiver's first loud frame): this is what a person at the other end of the call adds to the
benchmark's ``uplink`` and ``downlink`` numbers, minus the Runtime's own VAD and pacing. It
varies the things the Runtime controls: chunk size, queue size, paced or burst writes, the
idle time before the first frame, DTX. It also prints the round-trip time LiveKit reports and
the server the client was connected to.

Credentials come from the environment (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET); values are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics as stats
import struct
import sys
import time
import uuid
from pathlib import Path

from livekit import api, rtc

LOUD = 2000  # peak amplitude that counts as "the tone has arrived"
TONE_MS = 400


def tone(sample_rate: int, ms: int) -> bytes:
    n = sample_rate * ms // 1000
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sample_rate))) for i in range(n))


def token(identity: str, room: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )


class Receiver:
    def __init__(self, room: rtc.Room) -> None:
        self.armed_at: float | None = None
        self.first_loud: float | None = None
        self.frames: list[tuple[float, bool]] = []
        room.on("track_subscribed", self._on_track)

    def _on_track(self, track: rtc.Track, *_: object) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.get_running_loop().create_task(self._read(track))

    async def _read(self, track: rtc.Track) -> None:
        import numpy as np

        async for event in rtc.AudioStream(track, sample_rate=16000, num_channels=1):
            now = time.monotonic()
            samples = np.frombuffer(event.frame.data, dtype=np.int16)
            loud = bool(samples.size and np.abs(samples).max() > LOUD)
            if self.armed_at is not None and loud and self.first_loud is None and now >= self.armed_at:
                self.first_loud = now

    def arm(self) -> None:
        self.first_loud = None
        self.armed_at = time.monotonic()


async def one_config(sender: rtc.Room, receiver: Receiver, name: str, *, rate: int, chunk_ms: int, queue_ms: int,
                     paced: bool, idle_s: float, dtx: bool | None, rounds: int) -> dict:
    source = rtc.AudioSource(rate, 1, queue_size_ms=queue_ms)
    track = rtc.LocalAudioTrack.create_audio_track(f"probe-{name}", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    if dtx is not None:
        options.dtx = dtx
    publication = await sender.local_participant.publish_track(track, options)
    await asyncio.sleep(2.0)  # let the subscriber attach
    pcm = tone(rate, TONE_MS)
    step = rate * chunk_ms // 1000 * 2
    chunks = [pcm[i:i + step] for i in range(0, len(pcm), step)]
    latencies, returned = [], []
    for _ in range(rounds):
        await asyncio.sleep(idle_s)
        receiver.arm()
        started = time.monotonic()
        for chunk in chunks:
            frame = rtc.AudioFrame(chunk.ljust(step, b"\0"), rate, 1, len(chunk.ljust(step, b"\0")) // 2)
            await source.capture_frame(frame)
            if paced:
                await asyncio.sleep(chunk_ms / 1000)
        returned.append((time.monotonic() - started) * 1000)
        await source.wait_for_playout()
        await asyncio.sleep(0.3)
        if receiver.first_loud is not None:
            latencies.append((receiver.first_loud - started) * 1000)
    await sender.local_participant.unpublish_track(publication.sid)
    await source.aclose()
    return {"config": name, "rate": rate, "chunk_ms": chunk_ms, "queue_ms": queue_ms, "paced": paced, "idle_s": idle_s,
            "dtx": dtx, "rounds": rounds, "heard": len(latencies), "latency_ms": [round(x, 1) for x in latencies],
            "capture_returns_ms": [round(x, 1) for x in returned]}


def rtt_ms(stats_result: object) -> float | None:
    try:  # the publisher's selected candidate pair
        for stat in getattr(stats_result, "publisher_stats", []):
            pair = getattr(stat, "candidate_pair", None)
            if pair is not None and getattr(pair, "current_round_trip_time", 0):
                return round(pair.current_round_trip_time * 1000, 1)
    except Exception:
        pass
    return None


async def main(args: argparse.Namespace) -> None:
    url = os.environ["LIVEKIT_URL"]
    room_name = f"probe-{uuid.uuid4().hex[:8]}"
    a, b = rtc.Room(), rtc.Room()
    await a.connect(url, token("probe-a", room_name))
    await b.connect(url, token("probe-b", room_name))
    results = []
    plan = [  # name, rate, chunk_ms, queue_ms, paced, idle_s, dtx
        ("runtime-like (24k, 40ms chunks, 1000ms queue, burst)", 24000, 40, 1000, False, 1.5, None),
        ("paced writes (sleep one chunk between frames)", 24000, 40, 1000, True, 1.5, None),
        ("10 ms chunks", 24000, 10, 1000, False, 1.5, None),
        ("200 ms queue", 24000, 40, 200, False, 1.5, None),
        ("dtx off", 24000, 40, 1000, False, 1.5, False),
        ("8 s idle before the tone", 24000, 40, 1000, False, 8.0, None),
    ]
    for direction, (sender, receiver_room) in (("A to B", (a, b)), ("B to A", (b, a))):
        receiver = Receiver(receiver_room)
        for name, rate, chunk_ms, queue_ms, paced, idle_s, dtx in plan:
            if args.only and not any(word in name for word in args.only.split(",")):
                continue
            rounds = max(4, args.rounds // 2) if idle_s > 5 else args.rounds
            result = await one_config(sender, receiver, name, rate=rate, chunk_ms=chunk_ms, queue_ms=queue_ms,
                                      paced=paced, idle_s=idle_s, dtx=dtx, rounds=rounds)
            result["direction"] = direction
            results.append(result)
            v = result["latency_ms"]
            print(f"{direction}  {name:52} n={len(v):2}  P50 {stats.median(v):6.1f}  min {min(v):6.1f}  max {max(v):6.1f}" if v
                  else f"{direction}  {name}: nothing heard")
        pass
    stats_a = await a.get_rtc_stats()
    summary = {"livekit_host": url.split("//")[-1].split("/")[0], "rtt_ms_reported": rtt_ms(stats_a), "results": results}
    await a.disconnect()
    await b.disconnect()
    print(f"\nLiveKit host {summary['livekit_host']}, round-trip time reported by LiveKit: {summary['rtt_ms_reported']} ms")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "transport_probe.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None, help="comma-separated words; run only configurations whose name contains one")
    parsed = parser.parse_args()
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.environ.get(name):
            sys.exit(f"missing environment: {name}")
    asyncio.run(main(parsed))

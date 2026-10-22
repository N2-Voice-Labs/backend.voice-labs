"""Live latency report (NS-01): p50 / p95 time to first translated audio, per engine and mode.

    python -m src.evals.live_latency --platform http://localhost:8300 --confirm-billable \\
        --engines realtime,cascaded --modes instant,balanced,accurate --pairs ko-en \\
        --utterances 30 --out report.md

For every engine x latency mode x direction x language pair ("cell") the runner creates an
interpreter agent through the Platform API, starts a session, joins the LiveKit room as a
simulated participant (a fake microphone that plays dataset wav files in real time, with a
pause between utterances), listens to the translated audio, ends the session and then reads
the per-utterance ``utterance_timeline`` statistics back from ``GET /analytics/summary``.

Two views of the same latency are reported, because they answer different questions:

* **Runtime** (server): ``value_ms`` = T4 - T0 of the Runtime's ``utterance_timeline`` events,
  aggregated by the Platform (count, p50, p95 per engine). T0 is the speaker's speech start.
* **Heard** (client): measured by this runner on the received translation track. The speech
  start and end are known exactly (we play a wav whose leading/trailing silence we trim), the
  first voiced translated audio is detected on the incoming track. ``from speech start`` is
  comparable to T4 - T0 plus network; ``after speech end`` is the cascaded target
  (heard - end of the utterance's audio) and can be negative for the realtime engine, which
  speaks while the person is still talking. A missing translation (no voiced audio within
  ``--wait`` seconds) counts in ``missing`` and in neither percentile.

The NS-02 (interruptions) and NS-03 (audio quality) columns are in the table but empty on
purpose. ``--record-dir`` saves every received translation track as a wav (plus the utterance
windows in the JSON) so NS-03 can analyse them later.

Billable: sessions call Deepgram, OpenAI and Cartesia through the Runtime. Without
``--confirm-billable`` the runner only prints the plan and a rough cost estimate and exits
with status 2; ``--dry-run`` prints the same plan and exits 0. Neither touches the network.
The servers hold the provider keys; this runner needs only the Platform URL (and
``--api-key`` / ``PLATFORM_API_KEY`` when the Platform has ``PLATFORM_API_KEYS`` set).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
import time
import wave
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .dataset import DEFAULT_MANIFEST, load_manifest, validate_manifest
from .generate_audio import DEFAULT_OUT as DEFAULT_AUDIO_DIR
from .generate_audio import load_audio_manifest
from .metrics import latency_summary
from .run import file_digest, machine_description

DEFAULT_OUT_DIR = Path.home() / ".cache" / "voice-labs-evals" / "live"
ENGINES = ("realtime", "cascaded")
MODES = ("instant", "balanced", "accurate")
DIRECTIONS = ("one_way", "auto")
SAMPLE_RATE = 16000
FRAME_S = 0.01  # LiveKit audio frames are 10 ms
FRAME_BYTES = int(SAMPLE_RATE * FRAME_S) * 2
VOICED_PEAK = 500  # PCM16 peak that counts as sound; the same threshold as Runtime's timeline
MIN_VOICED_RUN_S = 0.06  # an onset needs this much continuous voiced audio (ignores clicks)
# The translation is "finished" after this much silence following voiced audio. The realtime
# engine speaks in bursts with pauses of 2-3 s, so a shorter wait lets its tail spill into the
# next utterance's window.
SETTLE_S = {"realtime": 3.0, "cascaded": 1.2}
MIC_TRACK = "mic"  # participants' own microphone tracks; everything else is a translation
JOIN_WAIT_S = 4.0  # Runtime joins the room and connects providers before the first utterance
TARGETS = {"realtime": ("from_start", 1.2), "cascaded": ("after_end", 1.5)}  # p95, seconds

# Rough list prices for the cost estimate only (USD). Check provider pricing before relying
# on them; the realtime figure is the one quoted in runtime/realtime_translate.py.
PRICE_REALTIME_PER_MIN = 0.034
PRICE_STT_PER_MIN = 0.0077  # Deepgram streaming nova-3, billed for the whole connection
PRICE_LLM_PER_UTTERANCE = 0.0005
PRICE_TTS_PER_UTTERANCE = 0.003  # about 100 characters of Cartesia speech
EST_WAIT_S = 4.0  # assumed time from end of speech to end of the translation, for the estimate


# -- plan ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    engine: str
    mode: str
    direction: str
    pair: str  # "ko-en": ko is spoken, en is heard (first language is the interpreter's source)

    @property
    def languages(self) -> tuple[str, str]:
        source, target = self.pair.split("-")
        return source, target

    @property
    def label(self) -> str:
        return f"{self.engine}/{self.mode}/{self.direction}/{self.pair}"


def split_csv(value: str, allowed: tuple[str, ...] | None = None, what: str = "value") -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [part for part in parts if allowed is not None and part not in allowed]
    if unknown or not parts:
        raise ValueError(f"unknown {what} {unknown or value!r}; choose from {', '.join(allowed or ())}")
    return parts


def plan_cells(engines: list[str], modes: list[str], directions: list[str],
               pairs: list[str]) -> tuple[list[Cell], list[tuple[Cell, str]]]:
    """All requested cells, and those skipped up front with the reason.

    The Platform has the final word (it answers 422 for other bad combinations, which the
    runner records per cell); only the rule that costs a whole cell is checked here:
    the realtime engine translates one direction only.
    """

    cells, skipped = [], []
    for engine, mode, direction, pair in itertools.product(engines, modes, directions, pairs):
        cell = Cell(engine, mode, direction, pair)
        if engine == "realtime" and direction != "one_way":
            skipped.append((cell, "realtime engine supports one_way only"))
        else:
            cells.append(cell)
    return cells, skipped


def interpreter_config(cell: Cell) -> dict[str, Any]:
    """The agent's ``interpreter`` block; the engine is explicit so the mode cannot change it."""

    source, target = cell.languages
    return {
        "languages": [source, target],
        "direction": cell.direction,
        "engine": cell.engine,
        "latency_mode": cell.mode,
    }


def pick_utterances(items: list[dict[str, Any]], cell: Cell, count: int) -> list[dict[str, Any]]:
    """``count`` dataset items to play, cycling when the category has fewer.

    ``one_way`` plays the ``<source>-<target>`` category. ``auto`` alternates between it and
    the reverse category, so both languages are heard. Each result is ``{"item", "speaker"}``
    where speaker is 0 for the first language and 1 for the second.
    """

    source, target = cell.languages
    forward = [i for i in items if i["category"] == f"{source}-{target}"]
    if not forward:
        raise ValueError(f"dataset has no category {source}-{target}")
    streams = [itertools.cycle(forward)]
    if cell.direction == "auto":
        backward = [i for i in items if i["category"] == f"{target}-{source}"]
        if not backward:
            raise ValueError(f"auto direction needs category {target}-{source} in the dataset")
        streams.append(itertools.cycle(backward))
    return [{"item": next(streams[n % len(streams)]), "speaker": n % len(streams)}
            for n in range(count)]


def estimate_cell(cell: Cell, count: int, avg_clip_s: float, pause_s: float) -> dict[str, float]:
    """Rough wall time (minutes) and cost (USD) of one cell. An upper-bound style guess."""

    per_utterance = avg_clip_s + EST_WAIT_S + SETTLE_S.get(cell.engine, 1.2) + pause_s
    minutes = (JOIN_WAIT_S + count * per_utterance) / 60
    if cell.engine == "realtime":
        cost = minutes * PRICE_REALTIME_PER_MIN
    else:
        cost = minutes * PRICE_STT_PER_MIN + count * (PRICE_LLM_PER_UTTERANCE + PRICE_TTS_PER_UTTERANCE)
    return {"minutes": minutes, "usd": cost}


def build_plan(cells: list[Cell], skipped: list[tuple[Cell, str]], count: int,
               avg_clip_s: float, pause_s: float, two_person: bool) -> str:
    lines = [f"Live latency plan: {len(cells)} cell(s) x {count} utterance(s)"
             f"{' (two-person rooms)' if two_person else ''}"]
    total_min = total_usd = 0.0
    for cell in cells:
        est = estimate_cell(cell, count, avg_clip_s, pause_s)
        total_min += est["minutes"]
        total_usd += est["usd"]
        lines.append(f"  {cell.label:<38} ~{est['minutes']:4.1f} min  ~${est['usd']:.2f}")
    for cell, reason in skipped:
        lines.append(f"  {cell.label:<38} skipped: {reason}")
    lines.append(f"Estimated total: ~{total_min:.0f} min, ~${total_usd:.2f} "
                 "(rough list prices; Runtime calls Deepgram, OpenAI and Cartesia)")
    return "\n".join(lines)


# -- audio analysis (pure) ---------------------------------------------------------------


def read_wav(path: Path) -> bytes:
    """PCM16 mono 16 kHz samples of a wav file."""

    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit wav")
        return handle.readframes(handle.getnframes())


def frame_peaks(pcm: bytes) -> list[int]:
    """Peak absolute sample of each 10 ms frame."""

    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    step = int(SAMPLE_RATE * FRAME_S)
    return [max((abs(s) for s in samples[i:i + step]), default=0) for i in range(0, len(samples), step)]


def speech_bounds(pcm: bytes) -> tuple[float, float]:
    """(start, end) in seconds of the voiced part of a clip, trimming leading/trailing silence."""

    peaks = frame_peaks(pcm)
    voiced = [i for i, peak in enumerate(peaks) if peak >= VOICED_PEAK]
    if not voiced:
        return 0.0, len(peaks) * FRAME_S
    return voiced[0] * FRAME_S, (voiced[-1] + 1) * FRAME_S


class VoiceDetector:
    """Finds when translated audio becomes audible on a stream of (start, peak, duration) frames."""

    def __init__(self) -> None:
        self.onsets: list[float] = []  # start of every voiced run of at least MIN_VOICED_RUN_S
        self.voiced_end: float | None = None
        self._run_start: float | None = None
        self._run_s = 0.0
        self._counted = False

    def feed(self, start: float, peak: int, duration: float) -> None:
        if peak < VOICED_PEAK:
            self._run_start, self._run_s, self._counted = None, 0.0, False
            return
        if self._run_start is None:
            self._run_start = start
        self._run_s += duration
        self.voiced_end = start + duration
        if not self._counted and self._run_s >= MIN_VOICED_RUN_S:
            self._counted = True
            self.onsets.append(self._run_start)

    def first_onset_since(self, since: float) -> float | None:
        return next((t for t in self.onsets if t >= since), None)


def utterance_result(index: int, item: dict[str, Any], speaker: int, start: float, end: float,
                     detector: VoiceDetector, run_t0: float) -> dict[str, Any]:
    """Heard-latency record of one played utterance; times are seconds after the run started."""

    onset = detector.first_onset_since(start)
    heard = onset is not None
    return {
        "index": index,
        "item_id": item["id"],
        "speaker": speaker,
        "source_language": item["source_language"],
        "speech_start_s": round(start - run_t0, 3),
        "speech_end_s": round(end - run_t0, 3),
        "speech_s": round(end - start, 3),
        "heard_start_s": round(onset - run_t0, 3) if heard else None,
        "heard_end_s": round(detector.voiced_end - run_t0, 3) if heard and detector.voiced_end else None,
        "from_start_s": round(onset - start, 3) if heard else None,
        "after_end_s": round(onset - end, 3) if heard else None,
    }


# -- aggregation and report ----------------------------------------------------------------


def pick_timeline(timeline: dict[str, Any], engine: str) -> dict[str, Any] | None:
    """The Platform's per-engine timeline entry for ``engine`` (else the busiest one)."""

    if not timeline:
        return None
    if engine in timeline:
        return timeline[engine]
    return max(timeline.values(), key=lambda entry: entry.get("count", 0))


# Runtime timeline marks (ms after T0, see runtime/timeline.py) in pipeline order.
STAGE_MARKS = (("vad", "T1 VAD"), ("first_transcript", "T2 first transcript"),
               ("translation_start", "T3 translation start"), ("first_audio", "T4 first audio"),
               ("stable", "T5 stable"))


def stage_summary(server: dict[str, Any] | None) -> dict[str, Any]:
    """p50/p95 seconds after T0 of every timeline mark the Runtime reported."""

    marks = (server or {}).get("marks") or {}
    return {
        name: {"count": marks[name]["count"], "p50": marks[name]["p50"] / 1000,
               "p95": marks[name]["p95"] / 1000}
        for name, _ in STAGE_MARKS if name in marks
    }


def overlapping(results: list[dict[str, Any]]) -> int:
    """Utterances whose translation was first heard before they started or after the next
    one started: a sign that one translation's audio was counted for another."""

    count = 0
    for index, result in enumerate(results):
        heard, start = result.get("heard_start_s"), result.get("speech_start_s")
        if heard is None or start is None:
            continue
        nxt = results[index + 1].get("speech_start_s") if index + 1 < len(results) else None
        if heard < start or (nxt is not None and heard >= nxt):
            count += 1
    return count


def summarize_cell(cell: Cell, engine: str, results: list[dict[str, Any]],
                   timeline: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    """One report row; seconds everywhere. ``runtime`` is the Platform's T4 - T0 summary."""

    played = len(results)
    heard = [r for r in results if r["heard_start_s"] is not None]
    server = pick_timeline(timeline, engine)
    return {
        "engine": engine,
        "mode": cell.mode,
        "direction": cell.direction,
        "pair": cell.pair,
        "played": played,
        "heard": len(heard),
        "missing": played - len(heard),
        "from_start": latency_summary([r["from_start_s"] for r in heard]),
        "after_end": latency_summary([r["after_end_s"] for r in heard]),
        "runtime": {
            "count": server["count"],
            "p50": server["p50"] / 1000,
            "p95": server["p95"] / 1000,
        } if server else None,
        "speech": latency_summary([r["speech_s"] for r in results if r.get("speech_s") is not None]),
        "stages": stage_summary(server),
        "overlapping": overlapping(results),
        "error": error,
        # NS-02 and NS-03 fill these in; the table already has the columns.
        "interruption": None,
        "audio_quality": None,
        "utterances": results,
    }


def _sec(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _pair(summary: dict[str, Any] | None) -> str:
    if not summary or not summary.get("count"):
        return "-"
    return f"{_sec(summary['p50'])} / {_sec(summary['p95'])}"


def target_verdict(row: dict[str, Any]) -> str:
    """p95 against the north-star target: realtime from speech start, cascaded after speech end."""

    key, limit = TARGETS.get(row["engine"], (None, None))
    p95 = row[key]["p95"] if key else None
    if p95 is None:
        return "-"
    return f"{'ok' if p95 <= limit else 'over'} (<= {limit:g} s)"


def build_report(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    headers = ["Engine", "Mode", "Direction", "Pair", "Played", "Heard", "Missing",
               "Runtime T4-T0 s p50 / p95 (n)", "Heard from speech start s p50 / p95",
               "Heard after speech end s p50 / p95", "p95 target",
               "NS-02 barge-in stop ms", "NS-03 gaps>200ms / clipping / LUFS"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        runtime = row["runtime"]
        cells = [row["engine"], row["mode"], row["direction"], row["pair"], str(row["played"]),
                 str(row["heard"]), str(row["missing"]),
                 f"{_pair(runtime)} ({runtime['count']})" if runtime else "-",
                 _pair(row["from_start"]), _pair(row["after_end"]), target_verdict(row),
                 "not measured (NS-02)", "not measured (NS-03)"]
        if row.get("error"):
            cells[7] = row["error"] if row["error"].startswith("skipped") else f"error: {row['error']}"
        lines.append("| " + " | ".join(cells) + " |")
    stage_headers = ["Engine", "Mode", "Direction", "Pair", "Speech length",
                     *(label for _, label in STAGE_MARKS), "Overlapping"]
    stages = ["| " + " | ".join(stage_headers) + " |",
              "|" + "|".join("---" for _ in stage_headers) + "|"]
    for row in rows:
        if row.get("error"):
            continue
        marks = row.get("stages") or {}
        stages.append("| " + " | ".join([
            row["engine"], row["mode"], row["direction"], row["pair"], _pair(row.get("speech")),
            *(_pair(marks.get(name)) for name, _ in STAGE_MARKS), str(row.get("overlapping", 0)),
        ]) + " |")
    notes = "\n".join(f"- **{key}**: {value}" for key, value in meta.items())
    return (
        "# Live latency report\n\n" + "\n".join(lines) + "\n\n"
        "## Where the time goes\n\n" + "\n".join(stages) + "\n\n"
        "Seconds after the speaker started (T0), p50 / p95 over the cell's utterances, from the "
        "Runtime's `utterance_timeline` marks. Compare T3 with the speech length: for the "
        "cascaded engine, T3 minus speech length is the end-of-turn wait plus the final "
        "transcript; T4 minus T3 is translation plus first TTS audio. The realtime engine does "
        "not report T2/T3. *Overlapping* counts utterances whose heard translation started "
        "before the utterance or after the next one began (their numbers are unreliable).\n\n"
        "Seconds. *Runtime* is the Platform's p50/p95 of `utterance_timeline` T4 - T0 (nearest "
        "rank). *Heard* is measured on the received translation track (linear interpolation), "
        "so it includes network and playout; after speech end can be negative when the "
        "translation starts before the speaker stops. Missing translations are in neither "
        "percentile. Targets: realtime p95 <= 1.2 s from speech start, cascaded p95 <= 1.5 s "
        "after speech end.\n\n" + notes + "\n"
    )


# -- Platform API ----------------------------------------------------------------------------


class Platform:
    """The few Platform calls the runner needs."""

    def __init__(self, base_url: str, api_key: str = "", client: httpx.AsyncClient | None = None):
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=30.0)

    async def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        response = await self.client.request(method, path, json=body)
        if response.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {response.status_code} {response.text[:200]}")
        return response.json() if response.content else None

    async def health(self) -> None:
        await self._call("GET", "/health")

    async def create_agent(self, name: str, interpreter: dict[str, Any]) -> dict[str, Any]:
        return await self._call("POST", "/agents", {"name": name, "mode": "interpreter", "interpreter": interpreter})

    async def disable_agent(self, agent_id: str) -> None:
        await self._call("DELETE", f"/agents/{agent_id}")

    async def create_session(self, agent_id: str, language: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {"agent_id": agent_id}
        if language:
            body["participant"] = {"language": language}
        return await self._call("POST", "/voice/sessions", body)

    async def add_participant(self, session_id: str, language: str) -> dict[str, Any]:
        return await self._call("POST", f"/voice/sessions/{session_id}/participants", {"language": language})

    async def end_session(self, session_id: str) -> None:
        await self._call("POST", f"/voice/sessions/{session_id}/end")

    async def timeline(self, agent_id: str) -> dict[str, Any]:
        """Per-engine ``{count, p50, p95, marks}`` (milliseconds) of this agent's conversations."""

        summary = await self._call("GET", f"/analytics/summary?agent_id={agent_id}")
        return summary.get("timeline") or {}

    async def wait_for_timeline(self, agent_id: str, expected: int, timeout: float,
                                interval: float = 2.0) -> dict[str, Any]:
        """Poll until ``expected`` timeline events arrived (the Runtime posts them asynchronously)."""

        deadline = time.monotonic() + timeout
        while True:
            timeline = await self.timeline(agent_id)
            seen = sum(entry.get("count", 0) for entry in timeline.values())
            if seen >= expected or time.monotonic() >= deadline:
                return timeline
            await asyncio.sleep(interval)


# -- live part (needs the livekit package, imported lazily) ---------------------------------------


class Mic:
    """A fake microphone: silence, or queued wav audio, sent in real time in 10 ms frames."""

    def __init__(self, rtc: Any):
        self.rtc = rtc
        self.source = rtc.AudioSource(SAMPLE_RATE, 1, queue_size_ms=200)
        self.buffer = bytearray()
        self.sent = 0
        self.base = 0.0
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.base = time.monotonic()
        self.task = asyncio.ensure_future(self._run())

    def time_of(self, frame_index: int) -> float:
        return self.base + frame_index * FRAME_S

    def say(self, pcm: bytes) -> int:
        """Queue a clip; returns the frame index at which it starts."""

        start = self.sent + len(self.buffer) // FRAME_BYTES
        self.buffer += pcm + b"\0" * (-len(pcm) % FRAME_BYTES)
        return start

    async def _run(self) -> None:
        while True:
            delay = self.time_of(self.sent) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            chunk = bytes(self.buffer[:FRAME_BYTES]).ljust(FRAME_BYTES, b"\0")
            del self.buffer[:FRAME_BYTES]
            self.sent += 1
            await self.source.capture_frame(self.rtc.AudioFrame(chunk, SAMPLE_RATE, 1, FRAME_BYTES // 2))


class Person:
    """A simulated participant: publishes ``mic``, listens to every non-mic remote audio track."""

    def __init__(self, rtc: Any, label: str, connection: dict[str, str], record: bool):
        self.rtc, self.label, self.connection, self.record = rtc, label, connection, record
        self.room = rtc.Room()
        self.mic = Mic(rtc)
        self.detector = VoiceDetector()
        self.tracks: dict[str, bytearray] = {}  # track name -> received pcm
        self.track_start: dict[str, float] = {}
        self.room.on("track_subscribed", self._subscribed)

    def _subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        if track.kind != self.rtc.TrackKind.KIND_AUDIO or publication.name == MIC_TRACK:
            return
        name = publication.name or participant.identity
        stream = self.rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        asyncio.ensure_future(self._listen(name, stream))

    async def _listen(self, name: str, stream: Any) -> None:
        async for event in stream:
            data = bytes(event.frame.data)
            now = time.monotonic()
            samples = array("h")
            samples.frombytes(data)
            duration = len(samples) / SAMPLE_RATE
            self.detector.feed(now - duration, max((abs(s) for s in samples), default=0), duration)
            if self.record:
                self.track_start.setdefault(name, now - duration)
                self.tracks.setdefault(name, bytearray()).extend(data)

    async def join(self) -> None:
        await self.room.connect(self.connection["url"], self.connection["token"],
                                options=self.rtc.RoomOptions(auto_subscribe=True))
        track = self.rtc.LocalAudioTrack.create_audio_track(MIC_TRACK, self.mic.source)
        await self.room.local_participant.publish_track(
            track, self.rtc.TrackPublishOptions(source=self.rtc.TrackSource.SOURCE_MICROPHONE))
        self.mic.start()

    async def leave(self) -> None:
        if self.mic.task:
            self.mic.task.cancel()
        await self.room.disconnect()


async def wait_for_translation(
    listener: Person, start: float, end: float, wait_s: float, settle_s: float
) -> None:
    """Until the translation of the utterance that started at ``start`` has finished playing."""

    detector = listener.detector
    while True:
        now = time.monotonic()
        onset = detector.first_onset_since(start)
        if onset is None and now - end >= wait_s:
            return  # nothing arrived: missing
        if onset is not None and detector.voiced_end is not None and now - detector.voiced_end >= settle_s:
            return
        if onset is not None and now - end >= wait_s + 30:
            return  # a translation that never stops; do not hang the run
        await asyncio.sleep(0.05)


def save_recordings(directory: Path, people: list[Person], run_t0: float) -> list[dict[str, Any]]:
    """Write every received translation track as 16 kHz mono wav; returns their index entries."""

    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for person in people:
        for name, pcm in person.tracks.items():
            path = directory / f"{person.label}_{name}.wav".replace("/", "_")
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(SAMPLE_RATE)
                out.writeframes(bytes(pcm))
            entries.append({"file": str(path), "listener": person.label, "track": name,
                            "starts_at_s": round(person.track_start[name] - run_t0, 3)})
    return entries


async def run_cell(args: argparse.Namespace, platform: Platform, cell: Cell, items: list[dict[str, Any]],
                   audio_dir: Path, run_id: str) -> dict[str, Any]:
    """Run one cell against live servers and return its report row (never raises)."""

    from livekit import rtc  # noqa: PLC0415 - only needed for live runs

    engine, error, results = cell.engine, None, []
    agent_id = None
    people: list[Person] = []
    session_id = None
    recordings: list[dict[str, Any]] = []
    try:
        agent = await platform.create_agent(f"live-latency {cell.label} {run_id}", interpreter_config(cell))
        agent_id = agent["id"]
        engine = (agent.get("interpreter") or {}).get("engine", engine)
        source, target = cell.languages
        session = await platform.create_session(agent_id, source if args.two_person else None)
        session_id = session["session_id"]
        print(f"[{cell.label}] session {session_id} runtime={session.get('runtime_status')}", flush=True)
        people = [Person(rtc, f"A-{source}", session["connection"], bool(args.record_dir))]
        if args.two_person:
            guest = await platform.add_participant(session_id, target)
            people.append(Person(rtc, f"B-{target}", guest["connection"], bool(args.record_dir)))
        for person in people:
            await person.join()
        run_t0 = time.monotonic()
        await asyncio.sleep(JOIN_WAIT_S)
        for index, entry in enumerate(pick_utterances(items, cell, args.utterances)):
            item = entry["item"]
            speaker = people[entry["speaker"]] if args.two_person else people[0]
            listener = people[1 - people.index(speaker)] if args.two_person else people[0]
            pcm = read_wav(audio_dir / f"{item['id']}.wav")
            lead, trail = speech_bounds(pcm)
            first_frame = speaker.mic.say(pcm)
            start = speaker.mic.time_of(first_frame + round(lead / FRAME_S))
            end = speaker.mic.time_of(first_frame + round(trail / FRAME_S))
            await asyncio.sleep(max(0.0, end - time.monotonic()))
            settle = args.settle if args.settle is not None else SETTLE_S.get(cell.engine, 1.2)
            await wait_for_translation(listener, start, end, args.wait, settle)
            results.append(utterance_result(index, item, entry["speaker"], start, end, listener.detector, run_t0))
            print(f"[{cell.label}] {index + 1}/{args.utterances} {item['id']} "
                  f"heard after {results[-1]['from_start_s']} s", flush=True)
            await asyncio.sleep(args.pause)
        if args.record_dir:
            recordings = save_recordings(args.record_dir / run_id / cell.label.replace("/", "_"), people, run_t0)
    except Exception as exc:  # noqa: BLE001 - one broken cell must not lose the others
        error = f"{type(exc).__name__}: {exc}"
        print(f"[{cell.label}] failed: {error}", file=sys.stderr, flush=True)
    finally:
        for person in people:
            try:
                await person.leave()
            except Exception:  # noqa: BLE001
                pass
        if session_id:
            try:
                await platform.end_session(session_id)
            except Exception as exc:  # noqa: BLE001
                error = error or f"end session: {exc}"
    timeline: dict[str, Any] = {}
    if agent_id:
        try:
            expected = sum(r["heard_start_s"] is not None for r in results)
            timeline = await platform.wait_for_timeline(agent_id, expected, args.event_wait)
            await platform.disable_agent(agent_id)
        except Exception as exc:  # noqa: BLE001
            error = error or f"analytics: {exc}"
    row = summarize_cell(cell, engine, results, timeline, error)
    row["recordings"] = recordings
    row["agent_id"] = agent_id
    row["session_id"] = session_id
    return row


# -- command line ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--platform", default=os.environ.get("VL_API", "http://localhost:8000"),
                        help="Platform base URL (default $VL_API or http://localhost:8000)")
    parser.add_argument("--api-key", default=os.environ.get("PLATFORM_API_KEY", ""),
                        help="Platform admin key when PLATFORM_API_KEYS is set (default $PLATFORM_API_KEY)")
    parser.add_argument("--engines", default="realtime,cascaded")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--directions", default="one_way",
                        help="one_way, auto or both; auto needs both languages in Deepgram's "
                             "code-switching set (not ko) and the cascaded engine")
    parser.add_argument("--pairs", default="ko-en",
                        help="source-target pairs, e.g. ko-en,ru-en (dataset categories)")
    parser.add_argument("--utterances", type=int, default=30, help="utterances per cell (default 30)")
    parser.add_argument("--two-person", action="store_true",
                        help="cascaded only: a second simulated participant listens "
                             "(translation-for-<identity>) instead of the speaker hearing it")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds of silence between utterances")
    parser.add_argument("--wait", type=float, default=15.0,
                        help="seconds after the end of speech before a translation counts as missing")
    parser.add_argument("--settle", type=float,
                        help="seconds of silence that end a translation (default: realtime 3.0, "
                             "cascaded 1.2)")
    parser.add_argument("--event-wait", type=float, default=20.0,
                        help="seconds to wait after the call for the Runtime's timeline events")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out", type=Path, help="report.md path (a .json with the data is written next to it)")
    parser.add_argument("--record-dir", type=Path,
                        help="save received translation tracks as wav here (for NS-03 audio analysis)")
    parser.add_argument("--confirm-billable", action="store_true",
                        help="required: sessions call paid Deepgram / OpenAI / Cartesia / realtime APIs")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and cost estimate, then exit")
    return parser


async def run_all(args: argparse.Namespace, cells: list[Cell], skipped: list[tuple[Cell, str]],
                  items: list[dict[str, Any]], audio_dir: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    platform = Platform(args.platform, args.api_key)
    try:
        await platform.health()
        rows = [await run_cell(args, platform, cell, items, audio_dir, meta["run_id"]) for cell in cells]
    finally:
        await platform.client.aclose()
    rows += [summarize_cell(cell, cell.engine, [], {}, f"skipped: {reason}") for cell, reason in skipped]
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engines = split_csv(args.engines, ENGINES, "engine")
        modes = split_csv(args.modes, MODES, "mode")
        directions = split_csv(args.directions, DIRECTIONS, "direction")
        pairs = split_csv(args.pairs)
        if any(len(pair.split("-")) != 2 for pair in pairs):
            raise ValueError("pairs look like ko-en")
        if args.utterances < 1:
            raise ValueError("--utterances must be at least 1")
        if args.two_person and "realtime" in engines:
            raise ValueError("--two-person needs the cascaded engine; the realtime engine serves one speaker")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    items = load_manifest(args.manifest)
    problems = validate_manifest(items)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    cells, skipped = plan_cells(engines, modes, directions, pairs)
    audio_dir = args.audio_dir.expanduser()
    audio_index = load_audio_manifest(audio_dir)
    try:
        chosen = {entry["item"]["id"] for cell in cells for entry in pick_utterances(items, cell, args.utterances)}
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    durations = [audio_index[i]["duration_s"] for i in chosen if i in audio_index]
    avg_clip_s = sum(durations) / len(durations) if durations else 5.0
    print(build_plan(cells, skipped, args.utterances, avg_clip_s, args.pause, args.two_person))
    missing = sorted(i for i in chosen if not (audio_dir / f"{i}.wav").exists())
    if missing:
        print(f"note: missing audio for {len(missing)} item(s) (e.g. {missing[0]}); run: "
              f"python -m src.evals.generate_audio --out {audio_dir}", file=sys.stderr)
    if args.dry_run:
        return 0
    if not args.confirm_billable:
        print("Refusing to run without --confirm-billable (sessions call paid APIs). "
              "Use --dry-run to only see the plan.", file=sys.stderr)
        return 2
    if missing or not cells:
        return 2
    try:
        import livekit.rtc  # noqa: F401, PLC0415
    except ImportError:
        print("error: the livekit package is required for live runs (use ~/.venvs/voice-labs/bin/python)",
              file=sys.stderr)
        return 2

    started = datetime.now(timezone.utc)
    meta = {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ") + "-live-latency",
        "date": started.isoformat(timespec="seconds"),
        "platform": args.platform,
        "dataset": f"{args.manifest.parent.name} (manifest {file_digest(args.manifest)[:12]}), "
                   "synthetic voices, items cycled when fewer than --utterances",
        "utterances_per_cell": args.utterances,
        "topology": "two-person room (listener hears translation-for-<identity>)" if args.two_person
        else "one participant speaks and hears the translation",
        "pause_s": args.pause,
        "machine": machine_description(),
    }
    try:
        rows = asyncio.run(run_all(args, cells, skipped, items, audio_dir, meta))
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"error: Platform unreachable or refused the request: {exc}", file=sys.stderr)
        return 2
    report = build_report(rows, meta)
    out = (args.out or DEFAULT_OUT_DIR / meta["run_id"] / "report.md").expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "cells": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report)
    print(f"wrote {out} and {out.with_suffix('.json')}")
    failed = [r for r in rows if r.get("error") and not r["error"].startswith("skipped")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

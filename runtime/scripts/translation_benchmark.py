#!/usr/bin/env python
"""Repeatable latency benchmark of the live ko<->en translation call (real providers, real LiveKit).

BILLABLE: every utterance is translated and spoken by ElevenLabs (about 40 characters of quota
each) and translated by Google (or the Gemini backup). The recorded source speech is made once
with the free ``edge-tts`` voices, so ElevenLabs is only spent inside the call.

Three steps, all offline except ``run``::

    python scripts/translation_benchmark.py prepare --out bench/samples [--count 200]
    python scripts/translation_benchmark.py run --samples bench/samples --profile balanced \\
        --out bench/balanced --confirm-billable
    python scripts/translation_benchmark.py report bench/balanced bench/instant

``prepare`` writes a deterministic corpus (``--seed``) and one 16 kHz PCM file per sentence, so
every profile and every fault run replays the SAME audio. ``run`` needs a Platform and a
Runtime already running against LiveKit (see ``docs/smoke-test.md``) and provider records for
STT, LLM and TTS; the LLM record must be ``google-translate`` to benchmark Google. It plays the
sentences, alternating speaker A (ko) and speaker B (en), into one session and writes
``raw.jsonl``: one line per utterance with every timestamp (monotonic clock, seconds from the
start of the run), the server-side stage timings, the recognised text and the translation.
``report`` reads those files and prints percentiles and rates; it judges nothing about
translation quality.

Environment: ``LIVEKIT_*`` is not needed (the Platform hands out tokens). Keys are never
printed. ``--env-file`` loads KEY=VALUE lines (default ``../.env``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
VOICES = {"ko": "ko-KR-SunHiNeural", "en": "en-US-JennyNeural"}
STT_CODE = {"ko": "kor", "en": "eng"}

# ---------------------------------------------------------------- corpus ----

KO_PLAIN = [
    "죄송하지만 그 계약서의 마지막 조항을 다시 설명해 주시겠어요?",
    "회의 자료는 오늘 오후까지 보내 드리겠습니다.",
    "이 제품은 다음 주부터 매장에서 판매될 예정입니다.",
    "혹시 다른 방법이 있는지 한번 알아봐 주시겠어요?",
    "저희 팀은 이번 프로젝트에 대해 매우 만족하고 있습니다.",
    "그 문제는 담당자와 먼저 상의해 보셔야 할 것 같아요.",
    "내일 아침에 다시 연락드려도 괜찮을까요?",
    "공항까지 가는 가장 빠른 길이 어디인가요?",
    "알레르기가 있어서 견과류가 들어간 음식은 먹을 수 없어요.",
    "계약서에 서명하기 전에 변호사에게 확인을 받고 싶습니다.",
    "배송이 늦어지고 있는데 어떻게 된 건지 알려 주세요.",
    "이번 달 보고서는 거의 다 완성되었습니다.",
    "죄송합니다만 말씀하신 내용을 잘 듣지 못했어요.",
    "환불 절차가 어떻게 되는지 설명해 주실 수 있나요?",
    "다음 단계로 넘어가기 전에 질문이 하나 있습니다.",
    "그 일정이라면 저희도 충분히 맞출 수 있을 것 같아요.",
    "호텔에서 공항까지 셔틀버스가 운행되나요?",
    "이 자료를 영어로 번역해 주시면 정말 감사하겠습니다.",
    "오늘은 몸이 좋지 않아서 일찍 퇴근하려고 합니다.",
    "가격을 조금만 조정해 주실 수 있을까요?",
    "저는 다음 달에 서울로 출장을 갈 예정입니다.",
    "그 부분은 제가 다시 확인하고 알려 드릴게요.",
    "창가 쪽 자리로 부탁드립니다.",
    "이 약은 하루에 두 번 식사 후에 드시면 됩니다.",
    "회의가 길어져서 약속에 조금 늦을 것 같습니다.",
    "견적서를 이메일로 보내 주시면 검토해 보겠습니다.",
    "정말 도움이 많이 되었습니다. 감사합니다.",
    "그건 제가 알고 있던 것과는 조금 다르네요.",
    "비밀번호를 잊어버려서 다시 설정하고 싶습니다.",
    "여기에서 가까운 약국이 어디에 있나요?",
]
KO_LEAD = ["", "저기요, ", "그리고 ", "혹시 ", "그런데 "]
KO_NUM = [
    "내일 {t}에 예약이 가능한가요?",
    "제 전화번호는 {phone}입니다.",
    "{n}명이 갈 건데 자리가 있을까요?",
    "회의는 {date}에 {room}호에서 열립니다.",
    "총 금액은 {price}원입니다.",
    "예약 번호는 {code}입니다.",
]
EN_PLAIN = [
    "Could you please explain the last clause of that contract again?",
    "I will send you the meeting materials this afternoon.",
    "This product is scheduled to go on sale in stores next week.",
    "Would you mind checking whether there is another option?",
    "Our team is very happy with how this project turned out.",
    "I think you should talk to the person in charge first.",
    "Is it all right if I call you again tomorrow morning?",
    "What is the fastest way to get to the airport from here?",
    "I am allergic to nuts, so I cannot eat anything that contains them.",
    "I would like a lawyer to look at the contract before I sign it.",
    "My delivery is late, could you tell me what is going on?",
    "This month's report is almost finished.",
    "I am sorry, I did not quite catch what you just said.",
    "Can you explain how the refund process works?",
    "I have one question before we move on to the next step.",
    "I think we can meet that schedule without any trouble.",
    "Does the hotel run a shuttle bus to the airport?",
    "I would really appreciate it if you could translate this document into Korean.",
    "I am not feeling well today, so I plan to leave work early.",
    "Could you lower the price just a little bit?",
    "I am going to Seoul on a business trip next month.",
    "Let me double check that part and get back to you.",
    "A table by the window, please.",
    "You should take this medicine twice a day after meals.",
    "The meeting is running long, so I may be a little late.",
    "If you send the quote by email, I will review it.",
    "That was really helpful, thank you very much.",
    "That is a little different from what I understood.",
    "I forgot my password and I would like to reset it.",
    "Where is the nearest pharmacy from here?",
]
EN_LEAD = ["", "Excuse me, ", "Also, ", "By the way, ", "Actually, "]
EN_NUM = [
    "Is a reservation available tomorrow at {t}?",
    "My phone number is {phone}.",
    "There will be {n} of us, do you have a table?",
    "The meeting is on {date} in room {room}.",
    "The total comes to {price} won.",
    "Your booking number is {code}.",
]


def _numbers(rng: random.Random, lang: str) -> dict[str, str]:
    hour = rng.randint(1, 11)
    return {
        "t": f"{'오후' if rng.random() < 0.6 else '오전'} {hour}시" if lang == "ko" else f"{hour}:{rng.choice(['00', '30'])} {rng.choice(['AM', 'PM'])}",
        "phone": f"010-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}",
        "n": str(rng.randint(2, 9)),
        "date": f"{rng.randint(1, 12)}월 {rng.randint(1, 28)}일" if lang == "ko" else f"November {rng.randint(1, 28)}",
        "room": str(rng.randint(101, 912)),
        "price": f"{rng.randint(10, 99)},{rng.randint(100, 999)}",
        "code": f"{rng.randint(10000, 99999)}",
    }


def build_corpus(count: int, seed: int) -> list[dict]:
    """``count`` sentences, alternating ko (speaker A) and en (speaker B); about 30% carry numbers."""

    rng = random.Random(seed)
    corpus: list[dict] = []
    seen: set[str] = set()
    plain = {"ko": (KO_PLAIN, KO_LEAD, KO_NUM), "en": (EN_PLAIN, EN_LEAD, EN_NUM)}
    while len(corpus) < count:
        lang = "ko" if len(corpus) % 2 == 0 else "en"
        sentences, leads, numbered = plain[lang]
        if rng.random() < 0.3:
            slots = _numbers(rng, lang)
            text = rng.choice(numbered).format(**slots)
        else:
            text = rng.choice(leads) + rng.choice(sentences)
        if text in seen:
            continue
        seen.add(text)
        corpus.append({
            "id": f"u{len(corpus):04d}", "speaker": "A" if lang == "ko" else "B", "lang": lang,
            "text": text, "digits": re.sub(r"\D", "", text),
        })
    return corpus


def cmd_prepare(args) -> None:
    import edge_tts

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(args.count, args.seed)

    async def make(item: dict) -> None:
        pcm = out / f"{item['id']}.pcm"
        if pcm.exists():
            return
        mp3 = out / f"{item['id']}.mp3"
        await edge_tts.Communicate(item["text"], VOICES[item["lang"]]).save(str(mp3))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-f", "s16le", "-ar", str(SR), "-ac", "1", str(pcm)],
            check=True,
        )
        mp3.unlink()

    async def go() -> None:
        sem = asyncio.Semaphore(4)

        async def one(item):
            async with sem:
                await make(item)

        await asyncio.gather(*(one(i) for i in corpus))

    asyncio.run(go())
    for item in corpus:
        item["audio_s"] = round((out / f"{item['id']}.pcm").stat().st_size / (SR * 2), 2)
    (out / "corpus.json").write_text(json.dumps({"seed": args.seed, "items": corpus}, ensure_ascii=False, indent=1))
    chars = sum(len(i["text"]) for i in corpus)
    print(f"{len(corpus)} sentences, {chars} characters, {sum(i['audio_s'] for i in corpus):.0f}s of audio in {out}")


# ------------------------------------------------------------------- run ----


def load_env(path: Path) -> None:
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                name, _, value = line.removeprefix("export ").partition("=")
                os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def el_characters() -> int | None:
    try:
        import httpx

        r = httpx.get("https://api.elevenlabs.io/v1/user/subscription", headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}, timeout=15)
        return r.json().get("character_count") if r.status_code == 200 else None
    except Exception:
        return None


class Person:
    """One participant: publishes recorded speech, records what it hears on its private listen room."""

    def __init__(self, name: str, connection: dict):
        from livekit import rtc

        self.rtc, self.name, self.conn = rtc, name, connection
        self.session, self.listen = rtc.Room(), rtc.Room()
        self.source = rtc.AudioSource(SR, 1)
        self.activity: list[tuple[float, bool]] = []  # (monotonic, loud?) per 10 ms frame

    async def join(self) -> None:
        rtc = self.rtc
        await self.session.connect(self.conn["url"], self.conn["token"])
        track = rtc.LocalAudioTrack.create_audio_track("mic", self.source)
        await self.session.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        listen = self.conn["listen"]

        @self.listen.on("track_subscribed")
        def _on(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self._drain(track))

        await self.listen.connect(listen["url"], listen["token"])

    async def _drain(self, track) -> None:
        import numpy as np

        async for event in self.rtc.AudioStream(track, sample_rate=SR, num_channels=1):
            samples = np.frombuffer(bytes(event.frame.data), dtype=np.int16)
            self.activity.append((time.monotonic(), bool(samples.size and np.abs(samples).max() > 400)))

    async def speak(self, pcm: bytes) -> None:
        rtc = self.rtc
        for offset in range(0, len(pcm), 320):
            chunk = pcm[offset:offset + 320].ljust(320, b"\0")
            await self.source.capture_frame(rtc.AudioFrame(chunk, SR, 1, 160))
        await self.source.wait_for_playout()  # the real end of speech, not the end of queuing

    async def close(self) -> None:
        await self.session.disconnect()
        await self.listen.disconnect()


def active_between(person: Person, start: float, end: float) -> list[float]:
    return [t for t, loud in person.activity if loud and start <= t <= end]


async def wait_for_translation(listener: Person, speaker: Person, started: float, ended: float, max_wait: float):
    """Wait until the listener's audio started and then went quiet for 1 s (or ``max_wait`` passed)."""

    deadline = ended + max_wait
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        loud = active_between(listener, started + 0.2, time.monotonic())
        if loud and time.monotonic() - loud[-1] > 1.0:
            return
    return


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


async def run_session(args) -> None:
    import httpx
    import numpy as np

    corpus = json.loads((Path(args.samples) / "corpus.json").read_text())["items"]
    if args.ids:  # an explicit selection, in the order given
        by_id = {item["id"]: item for item in corpus}
        corpus = [by_id[i] for i in args.ids.split(",")]
    else:
        corpus = corpus[: args.utterances]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = args.platform.rstrip("/")
    log_path = Path(args.runtime_log) if args.runtime_log else None
    log_offset = log_path.stat().st_size if log_path and log_path.exists() else 0
    chars_before = el_characters()

    async with httpx.AsyncClient(timeout=60) as client:
        if args.fake_google:
            await client.post(f"{args.fake_google.rstrip('/')}/mode/{args.google_mode}")
        ids = {}
        for kind, name in (("stt", "elevenlabs"), ("llm", args.llm), ("tts", "elevenlabs")):
            found = [p for p in (await client.get(f"{api}/providers")).json() if p["kind"] == kind and p["provider"] == name]
            ids[kind] = found[0]["id"] if found else (await client.post(f"{api}/providers", json={"kind": kind, "provider": name})).json()["id"]
        body = {
            "languages": ["ko", "en"], "latency_mode": args.profile,
            "stt_provider_id": ids["stt"], "llm_provider_id": ids["llm"], "tts_provider_id": ids["tts"],
            "creator": {"language": "ko", "display_name": "A"},
        }
        created = (await client.post(f"{api}/translations", json=body))
        created.raise_for_status()
        a = created.json()
        invite = await client.post(
            f"{api}/translations/{a['session_id']}/invite",
            json={"language": "en", "display_name": "B"},
            headers={"Authorization": f"Bearer {a['session_token']}", "Idempotency-Key": uuid.uuid4().hex},
        )
        invite.raise_for_status()
        people = {"A": Person("A", a["connection"]), "B": Person("B", invite.json()["connection"])}
        await asyncio.gather(*(p.join() for p in people.values()))
        await asyncio.sleep(3)
        t0 = time.monotonic()
        rows = []
        for item in corpus:
            speaker = people[item["speaker"]]
            listener = people["B" if item["speaker"] == "A" else "A"]
            pcm = (Path(args.samples) / f"{item['id']}.pcm").read_bytes()
            samples = np.frombuffer(pcm, dtype=np.int16)
            voiced = np.flatnonzero(np.abs(samples) > 400)
            trailing = (len(samples) - 1 - int(voiced[-1])) / SR if voiced.size else 0.0
            wall = datetime.now(timezone.utc)
            started = time.monotonic()
            await speaker.speak(pcm)
            file_end = time.monotonic()
            ended = file_end - trailing  # the last spoken sound, not the end of the silence after it
            await wait_for_translation(listener, speaker, started, file_end, args.max_wait)
            loud = active_between(listener, started + 0.2, time.monotonic())
            own = active_between(speaker, ended, time.monotonic())
            first = loud[0] if loud else None
            rows.append({
                **item, "profile": args.profile, "llm": args.llm, "google_mode": args.google_mode if args.fake_google else "real",
                "timeout_s": os.getenv("GOOGLE_TRANSLATE_TIMEOUT_S") or "default",
                "wall_start": wall.isoformat(),
                # absolute monotonic stamps (shared by all processes of this host) so the Runtime's
                # own monotonic marks can be lined up with what the client did
                "mono_start": round(started, 6), "mono_end_speech": round(ended, 6),
                "mono_first_audio": None if first is None else round(first, 6),
                "t_start": round(started - t0, 3), "t_file_end": round(file_end - t0, 3),
                "trailing_silence_s": round(trailing, 3), "t_end_speech": round(ended - t0, 3),
                "t_first_audio": None if first is None else round(first - t0, 3),
                "e2e_s": None if first is None else round(first - ended, 3),
                "heard_s": round(len(loud) * 0.01, 2), "speaker_heard_s": round(len(own) * 0.01, 2),
            })
            await asyncio.sleep(args.gap)
            print(f"\r{len(rows)}/{len(corpus)}", end="", flush=True)
            if args.char_budget and chars_before is not None:
                used = el_characters()
                if used is not None and used - chars_before >= args.char_budget:
                    print(f"\nstopped: {used - chars_before} ElevenLabs characters used (budget {args.char_budget})")
                    break
        print()
        await asyncio.sleep(2)
        for p in people.values():
            await p.close()
        cid = (await client.get(f"{api}/translations/{a['session_id']}")).json().get("conversation_id") or a["conversation_id"]
        await client.post(f"{api}/voice/sessions/{a['session_id']}/end")
        await asyncio.sleep(2)
        messages = (await client.get(f"{api}/conversations/{cid}/messages")).json()
        debug = (await client.get(f"{api}/conversations/{cid}/debug")).json()
        # an engine other than the one asked for would invalidate the comparison: record it
        config = debug.get("config", {})
    merge_server_side(rows, messages, debug.get("turns", []))
    log_lines = []
    if log_path and log_path.exists():
        with log_path.open("rb") as handle:
            handle.seek(log_offset)
            log_lines = handle.read().decode(errors="replace").splitlines()
    attach_fine_timeline(rows, log_lines)
    attach_probes(rows, log_lines)
    summary = {
        "profile": args.profile, "llm": args.llm, "utterances": len(rows), "session_id": a["session_id"], "conversation_id": cid,
        "engine": (config.get("interpreter") or {}).get("engine"), "latency_mode": (config.get("interpreter") or {}).get("latency_mode"),
        "providers": config.get("providers"),
        "primary_failures_logged": sum("Primary translator" in l and "failed" in l for l in log_lines),
        "primary_empty_logged": sum("Primary translator" in l and "returned nothing" in l for l in log_lines),
        "fallback_failures_logged": sum("Fallback translator" in l and "failed" in l for l in log_lines),
        "translation_failures_logged": sum("Interpreter translation failed" in l for l in log_lines),
        "elevenlabs_characters_used": None if chars_before is None or (chars_after := el_characters()) is None else chars_after - chars_before,
        "elevenlabs_characters_before": chars_before,
        "google_timeout_s": os.getenv("GOOGLE_TRANSLATE_TIMEOUT_S") or "default",
        "google_mode": args.google_mode if args.fake_google else "real",
    }
    with (out / "raw.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / "server_turns.json").write_text(json.dumps({"messages": messages, "turns": debug.get("turns", [])}, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


# ``speaker= listener= destination=`` were added to the line for two-person calls; older logs lack them.
FINE_LOG = re.compile(
    r"utterance_timeline turn=\S+ engine=\S+(?: speaker=\S+ listener=\S+ destination=\S+)? t0_monotonic_s=([\d.]+) marks_ms=(\{.*\})"
)
# Steps between two points of the fine timeline, in the order they happen. A point is a client
# time (``client_*``, from the benchmark) or a Runtime mark. Network-dependent steps are tagged.
FINE_STEPS = (
    ("uplink+detect", "client_end_speech", "speech_end", "network: client, LiveKit, jitter buffer, VAD frames"),
    ("vad_silence", "speech_end", "vad_stop", "fixed: VAD stop_secs"),
    ("stt", "vad_stop", "final_transcript", "network: ElevenLabs STT"),
    ("pause_wait", "final_transcript", "turn_end", "overlaps the STT wait: 0 when STT is slower than the pause"),
    ("hop", "turn_end", "translation_start", "in process"),
    ("translation", "translation_start", "translation_first_chunk", "network: Google Translation"),
    ("tts", "tts_request", "first_audio", "network: ElevenLabs TTS"),
    ("downlink", "first_audio", "client_first_audio", "network: LiveKit, jitter buffer, client"),
)


PROBE_LOG = re.compile(r"^partial_probe (\{.*\})\s*$")


def normalise(text: str | None) -> str:
    """Conservative: case, punctuation and spacing only. No words are changed or dropped."""

    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").casefold()).split())


def attach_probes(rows: list[dict], log_lines: list[str]) -> None:
    """Compare the partial that was current at the VAD stop with the first final transcript.

    A probe belongs to the last row that had started speaking by the VAD stop (monotonic clock,
    same host). ``probes`` lists one entry per STT segment of the utterance; a split sentence has several.
    """

    for line in log_lines:
        match = PROBE_LOG.match(line.strip())
        if not match:
            continue
        probe = json.loads(match.group(1))
        owners = [r for r in rows if r.get("mono_start") is not None and r["mono_start"] <= probe["t_vad_stop"]]
        if not owners:
            continue
        partial, final = probe["partial"], probe["final"]
        owners[-1].setdefault("probes", []).append({
            "partial": partial, "final": final,
            "partial_age_ms": None if probe["t_partial"] is None else round((probe["t_vad_stop"] - probe["t_partial"]) * 1000, 1),
            "final_after_vad_stop_ms": round((probe["t_final"] - probe["t_vad_stop"]) * 1000, 1),
            "exact": partial == final,
            "normalised": bool(partial) and normalise(partial) == normalise(final),
            "prefix": bool(partial) and normalise(final).startswith(normalise(partial)) and normalise(partial) != normalise(final),
        })


# Sub-steps of the two transport steps, available when the Runtime ran with RUNTIME_TIMELINE_DIAG=1.
TRANSPORT_STEPS = (
    ("up: client -> Runtime input", "client_end_speech", "audio_in_last_voiced", "network + client send queue + LiveKit + Runtime input buffering"),
    ("up: input -> VAD estimate", "audio_in_last_voiced", "speech_end", "VAD window/threshold bias (the speech_end error)"),
    ("down: TTS request -> first TTS frame", "tts_request", "tts_first_frame", "ElevenLabs"),
    ("down: TTS frame -> submit to LiveKit", "tts_first_frame", "livekit_submit", "Runtime buffering: voiced test, 40 ms chunking, playout clock"),
    ("down: capture_frame call", "livekit_submit", "livekit_submitted", "LiveKit audio source queue (back-pressure)"),
    ("down: LiveKit -> client hears it", "livekit_submitted", "client_first_audio", "encode + network + SFU + jitter buffer + client"),
)


def attach_fine_timeline(rows: list[dict], log_lines: list[str]) -> None:
    """Line the Runtime's monotonic marks (from its log) up with the client's own timestamps.

    Valid only when the client and the Runtime run on one host (CLOCK_MONOTONIC is shared).
    Each row gets ``fine``: the points in seconds on that clock, and ``fine_steps``: ms per step.
    A sentence the STT split in two has several timelines; the last one is the one the client's
    end-to-end time refers to.
    """

    entries = []
    for line in log_lines:
        match = FINE_LOG.search(line)
        if match:
            entries.append((float(match.group(1)), json.loads(match.group(2).replace("'", '"'))))
    for t0, marks in entries:
        owners = [r for r in rows if r.get("mono_start") is not None and r["mono_start"] <= t0 + 0.05]
        if not owners:
            continue
        row = owners[-1]
        if row.get("fine_t0", -1) > t0:
            continue
        row["fine_t0"] = t0
        points = {name: t0 + value / 1000 for name, value in marks.items()}
        points["client_end_speech"] = row.get("mono_end_speech")
        points["client_first_audio"] = row.get("mono_first_audio")
        row["fine_steps"] = {
            name: round((points[b] - points[a]) * 1000, 1)
            for name, a, b, _ in (*FINE_STEPS, *TRANSPORT_STEPS)
            if points.get(a) is not None and points.get(b) is not None
        }


# (from mark, to mark) of the pipeline's utterance timeline. The marks are measured from the
# start of the utterance, so only differences between two marks are stages.
DERIVED = {
    "d_translation": ("translation_start", "translation_first_chunk"),
    "d_tts": ("tts_request", "first_audio"),
}
STAGES = ("stt_ttfb", "translation_first_chunk_ms", "translation_ms", "tts_ttfb", "turn_to_first_audio", "utterance_timeline")


def merge_server_side(rows: list[dict], messages: list[dict], turns: list[dict]) -> None:
    """Attach recognised text, translation and per-stage timings to each utterance by time window."""

    def when(text: str) -> datetime:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc) if "+" not in text and not text.endswith("Z") else datetime.fromisoformat(text.replace("Z", "+00:00"))

    starts = [datetime.fromisoformat(r["wall_start"]) for r in rows]

    def slot(moment: datetime) -> int | None:
        index = None
        for i, start in enumerate(starts):
            if moment >= start - timedelta(seconds=0.5):
                index = i
        return index

    for row in rows:
        row.update({"stt_text": [], "translation": [], "turns": 0, **{s: None for s in STAGES}})
    for message in messages:
        i = slot(when(message["created_at"]))
        if i is not None:
            rows[i]["stt_text" if message["role"] == "user" else "translation"].append(message["content"])
    for turn in turns:
        i = slot(when(turn["started_at"]))
        if i is None or not turn.get("latency"):
            continue
        rows[i]["turns"] += 1
        for stage in STAGES:
            value = turn["latency"].get(stage)
            if isinstance(value, dict):  # the utterance timeline carries its marks
                rows[i].setdefault("marks_ms", value.get("marks_ms"))
                value = value.get("value_ms")
            if rows[i][stage] is None and value is not None:
                rows[i][stage] = value
    for row in rows:
        marks = {k: v for k, v in (row.get("marks_ms") or {}).items() if isinstance(v, (int, float))}
        # stages derived from the pipeline's own marks (one clock, so overlaps cannot mislead)
        for name, (a, b) in DERIVED.items():
            row[name] = round(marks[b] - marks[a], 1) if a in marks and b in marks else None
        # What is left of the end-to-end time once translation and TTS are taken out: end of
        # speech detection, the final transcript, the turn pause and the audio transport.
        row["d_other"] = (
            round(row["e2e_s"] * 1000 - row["d_translation"] - row["d_tts"], 1)
            if row.get("e2e_s") is not None and row["d_translation"] is not None and row["d_tts"] is not None
            else None
        )
        row["split"] = len(row["stt_text"]) > 1
        row["digits_stt"] = re.sub(r"\D", "", " ".join(row["stt_text"]))
        row["digits_translation"] = re.sub(r"\D", "", " ".join(row["translation"]))


# ------------------------------------------------- translation only ----


async def run_translate(args) -> None:
    """Direct Google Translation requests (no STT, no TTS): ko->en and en->ko interleaved."""

    import httpx

    items = json.loads((Path(args.samples) / "corpus.json").read_text())["items"]
    by_lang = {lang: [i for i in items if i["lang"] == lang] for lang in ("ko", "en")}
    pairs = {"ko": "en", "en": "ko"}  # source -> target
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    url = (os.getenv("GOOGLE_TRANSLATE_BASE_URL") or "https://translation.googleapis.com/language/translate/v2").rstrip("/")
    headers = {"x-goog-api-key": os.environ["GOOGLE_TRANSLATE_API_KEY"]}
    rows = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for _ in range(3):  # connection set-up is not what a sentence pays for
            await client.get(url + "/languages", params={"target": "en"}, headers=headers)
        t0 = time.monotonic()
        for n in range(args.requests):
            source = "ko" if n % 2 == 0 else "en"
            item = by_lang[source][(n // 2) % len(by_lang[source])]
            body = {"q": item["text"], "source": source, "target": pairs[source], "format": "text"}
            started = time.monotonic()
            row = {"n": n, "id": item["id"], "direction": f"{source}->{pairs[source]}", "chars": len(item["text"]),
                   "t_start": round(started - t0, 3), "wall": datetime.now(timezone.utc).isoformat()}
            try:
                response = await client.post(url, headers=headers, json=body)
                row.update(latency_ms=round((time.monotonic() - started) * 1000, 1), status=response.status_code,
                           bytes=len(response.content), error=None if response.status_code == 200 else response.text[:120])
            except Exception as error:
                row.update(latency_ms=round((time.monotonic() - started) * 1000, 1), status=None, bytes=0, error=type(error).__name__)
            rows.append(row)
            if n % 50 == 0:
                print(f"\r{n}/{args.requests}", end="", flush=True)
            await asyncio.sleep(args.pause)
    print()
    with (out / "requests.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"kind": "translate", "requests": len(rows), "endpoint": "real" if "googleapis.com" in url else url,
                                                  "pause_s": args.pause, "timeout_s": args.timeout}, indent=1))
    print(f"{len(rows)} requests, {sum(r['status'] != 200 for r in rows)} not OK, saved to {out}")


def report_translate(path: Path) -> None:
    rows = [json.loads(l) for l in (path / "requests.jsonl").read_text().splitlines()]
    print(f"\n=== {path}  Google Translation only, {len(rows)} requests, endpoint={json.loads((path / 'summary.json').read_text())['endpoint']}")
    for label, subset in (("all", rows), ("ko->en", [r for r in rows if r["direction"] == "ko->en"]), ("en->ko", [r for r in rows if r["direction"] == "en->ko"])):
        ok = [r["latency_ms"] for r in subset if r["status"] == 200]
        print(f"  {label:7} {describe(ok)}  {unreliable(len(ok))}")
        over = ", ".join(f">{ms}ms {100 * sum(v > ms for v in ok) / max(len(ok), 1):.1f}%" for ms in (500, 750, 1000, 2000))
        errors = [r for r in subset if r["status"] != 200]
        print(f"          {over}   errors {len(errors)}/{len(subset)}   response bytes median {statistics.median([r['bytes'] for r in subset]):.0f} max {max(r['bytes'] for r in subset)}")
    codes = {}
    for r in rows:
        codes[r["status"] or r["error"]] = codes.get(r["status"] or r["error"], 0) + 1
    print("  status codes:", codes)


def unreliable(n: int) -> str:
    return "" if n >= 100 else f"[P99 unreliable: n={n} < 100]"


# ---------------------------------------------------------------- report ----


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    return "n={:<3} p50={:>6.0f} p90={:>6.0f} p95={:>6.0f} p99={:>6.0f} max={:>6.0f}".format(
        len(values), pct(values, .5), pct(values, .9), pct(values, .95), pct(values, .99), max(values))


def side_by_side(folders: list[str]) -> None:
    """Balanced vs Instant (or any runs) in one table."""

    runs = []
    for folder in folders:
        path = Path(folder)
        rows = [json.loads(l) for l in (path / "raw.jsonl").read_text().splitlines()]
        runs.append((path, json.loads((path / "summary.json").read_text()), rows))

    def column(rows, key, scale=1.0):
        return [r[key] * scale for r in rows if r.get(key) is not None]

    metrics = [
        ("last sound -> first audio (ms)", lambda rows: column(rows, "e2e_s", 1000)),
        ("  Korean speaker", lambda rows: [r["e2e_s"] * 1000 for r in rows if r["e2e_s"] is not None and r["lang"] == "ko"]),
        ("  English speaker", lambda rows: [r["e2e_s"] * 1000 for r in rows if r["e2e_s"] is not None and r["lang"] == "en"]),
        ("STT first result (ms)", lambda rows: column(rows, "stt_ttfb")),
        ("Translation (ms)", lambda rows: column(rows, "d_translation")),
        ("TTS first audio (ms)", lambda rows: column(rows, "d_tts")),
    ]
    width = 26
    print("\n=== side by side (P99 is unreliable below 100 samples and is flagged *)")
    print(" " * 32 + "".join(f"{Path(p).name + ' ' + s['profile']:>{width}}" for p, s, _ in runs))
    for name, getter in metrics:
        for q, label in ((.5, "P50"), (.95, "P95"), (.99, "P99")):
            line = f"{name if q == .5 else '':32}"
            for _, _, rows in runs:
                values = getter(rows)
                value = pct(values, q)
                flag = "*" if q == .99 and len(values) < 100 else ""
                cell = "-" if value is None else f"{label} {value:6.0f}{flag} (n={len(values)})"
                line += f"{cell:>{width}}"
            print(line)
    print(f"{'premature splits':32}" + "".join(f"{sum(r['split'] for r in rows):>{width - 8}}/{len(rows):<3}     ({100 * sum(r['split'] for r in rows) / len(rows):.1f}%)"[:width].rjust(width) for _, _, rows in runs))
    print(f"{'no audio / no translation':32}" + "".join(f"{sum(r['e2e_s'] is None for r in rows)} / {sum(not r['translation'] for r in rows)}".rjust(width) for _, _, rows in runs))
    print(f"{'engine / fallbacks':32}" + "".join(f"{s['engine']} / {s['primary_failures_logged'] + s['primary_empty_logged']}".rjust(width) for _, s, _ in runs))
    print(f"{'ElevenLabs characters used':32}" + "".join(str(s['elevenlabs_characters_used']).rjust(width) for _, s, _ in runs))
    total = sum(s['elevenlabs_characters_used'] or 0 for _, s, _ in runs)
    print(f"{'total ElevenLabs characters':32}{total}")


def cmd_report(args) -> None:
    for folder in args.dirs:
        path = Path(folder)
        if (path / "requests.jsonl").exists():
            report_translate(path)
            continue
        rows = [json.loads(l) for l in (path / "raw.jsonl").read_text().splitlines()]
        summary = json.loads((path / "summary.json").read_text())
        print(f"\n=== {path}  profile={summary['profile']} llm={summary['llm']} engine={summary['engine']} "
              f"google={summary['google_mode']} timeout={summary['google_timeout_s']}  utterances={len(rows)}")
        for r in rows:  # older files: compute the residual here
            if r.get("d_other") is None and r.get("e2e_s") is not None and r.get("d_translation") is not None and r.get("d_tts") is not None:
                r["d_other"] = round(r["e2e_s"] * 1000 - r["d_translation"] - r["d_tts"], 1)
        heard = [r for r in rows if r["e2e_s"] is not None]
        print("last spoken sound -> first translated audio (ms):")
        print("  all      ", describe([r["e2e_s"] * 1000 for r in heard]))
        for lang in ("ko", "en"):
            print(f"  {lang} speaker", describe([r["e2e_s"] * 1000 for r in heard if r["lang"] == lang]))
        print("server-side stages (ms):")
        for stage in STAGES:
            print(f"  {stage:28}", describe([r[stage] for r in rows if r.get(stage) is not None]))
        print("from the utterance timeline (ms); d_other = end-to-end minus translation and TTS "
              "(end-of-speech detection, final transcript, turn pause, transport):")
        for name in (*DERIVED, "d_other"):
            print(f"  {name:28}", describe([r[name] for r in rows if r.get(name) is not None]))
        fine = [r for r in rows if r.get("fine_steps") and not r["split"]]
        if fine:
            print(f"fine timeline, unsplit sentences with Runtime marks (n={len(fine)}), ms:")
            for name, _, _, note in (*FINE_STEPS, *TRANSPORT_STEPS):
                values = [r["fine_steps"][name] for r in fine if name in r["fine_steps"]]
                if values:
                    print(f"  {name:14}", describe(values), f" [{note}]")
        probes = [p for r in rows for p in r.get("probes", [])]
        if probes:
            n = len(probes)
            print(f"partial STT text at the VAD stop vs the first final transcript (n={n} segments, exploratory):")
            print(f"  no partial yet: {sum(not p['partial'] for p in probes)}   exact: {sum(p['exact'] for p in probes)}   "
                  f"equal after case/punctuation/space normalisation: {sum(p['normalised'] for p in probes)}   "
                  f"partial is only the start of the final: {sum(p['prefix'] for p in probes)}")
            print("  age of that partial before the VAD stop (ms):", describe([p["partial_age_ms"] for p in probes if p["partial_age_ms"] is not None]))
            print("  final transcript after the VAD stop (ms)   :", describe([p["final_after_vad_stop_ms"] for p in probes]))
        translation = [r["translation_ms"] for r in rows if r.get("translation_ms") is not None]
        if translation:
            print("  translation over a deadline:", ", ".join(
                f"{ms}ms {100 * sum(v > ms for v in translation) / len(translation):.1f}%" for ms in (500, 750, 1000, 2000)))
        splits = [r for r in rows if r["split"]]
        missing = [r for r in rows if not r["translation"]]
        silent = [r for r in rows if r["e2e_s"] is None]
        print(f"premature splits : {len(splits)}/{len(rows)} ({100 * len(splits) / len(rows):.1f}%)")
        print(f"no translation   : {len(missing)}   no audio heard: {len(silent)}   "
              f"speaker got a notice: {sum(r['speaker_heard_s'] > 1.0 for r in rows)}")
        def norm(text: str) -> str:
            text = re.sub(r"(\d+):00", r"\1", " ".join(text) if isinstance(text, list) else text)
            return re.sub(r"(?<=\d),(?=\d)", "", text)

        def digits_of(text) -> str:
            return re.sub(r"\D", "", norm(text))

        def subseq(a: str, b: str) -> bool:  # "12" inside "1112": a month written as a number
            it = iter(b)
            return all(ch in it for ch in a)

        for r in rows:
            r["digits_stt"], r["digits_translation"] = digits_of(r["stt_text"]), digits_of(r["translation"])
        numbered = [r for r in rows if r["digits"]]
        recognised = [r for r in numbered if r["digits_stt"]]
        kept = [r for r in recognised if r["digits_stt"] == r["digits_translation"]]
        lenient = [r for r in recognised if subseq(r["digits_stt"], r["digits_translation"]) or subseq(r["digits_translation"], r["digits_stt"])]
        print(f"numbers          : {len(numbered)} sentences with digits; STT wrote digits in {len(recognised)}; "
              f"STT digits == translation digits in {len(kept)}/{len(recognised)} "
              f"(lenient {len(lenient)}/{len(recognised)}); "
              f"STT digits == spoken digits in {sum(r['digits_stt'] == r['digits'] for r in recognised)}/{len(recognised)}")
        for r in recognised:
            if r not in lenient:
                print(f"   !! {r['id']} stt={r['stt_text']} -> {r['translation']}")
        print("fallback         : primary failures {primary_failures_logged}, empty {primary_empty_logged}, "
              "backup failures {fallback_failures_logged}, dropped {translation_failures_logged}".format(**summary))
        print(f"elevenlabs chars : {summary['elevenlabs_characters_used']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / ".env")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out", required=True); p.add_argument("--count", type=int, default=200); p.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("run")
    r.add_argument("--samples", required=True); r.add_argument("--out", required=True)
    r.add_argument("--profile", choices=("instant", "balanced", "accurate"), default="balanced")
    r.add_argument("--utterances", type=int, default=200)
    r.add_argument("--ids", help="comma-separated corpus ids to replay, in this order (overrides --utterances)")
    r.add_argument("--llm", default="google-translate", help="llm provider record to use; google-translate is the benchmark default")
    r.add_argument("--platform", default="http://localhost:8000")
    r.add_argument("--runtime-log", help="Runtime log file, to count fallback activations during the run")
    r.add_argument("--fake-google", help="fault-injection server URL (see scratchpad fake_google.py); NOT for normal benchmarks")
    r.add_argument("--google-mode", default="ok", help="fake Google mode: ok, slow500, slow1000, 500 (HTTP), hang ...")
    r.add_argument("--char-budget", type=int, help="stop when this many ElevenLabs characters were used (checked after each utterance)")
    r.add_argument("--gap", type=float, default=0.6, help="silence between utterances")
    r.add_argument("--max-wait", type=float, default=15.0)
    r.add_argument("--confirm-billable", action="store_true")
    t = sub.add_parser("translate", help="Google Translation only: no STT, no TTS, no ElevenLabs quota")
    t.add_argument("--samples", required=True); t.add_argument("--out", required=True)
    t.add_argument("--requests", type=int, default=1000); t.add_argument("--pause", type=float, default=0.05)
    t.add_argument("--timeout", type=float, default=10.0); t.add_argument("--confirm-billable", action="store_true")
    q = sub.add_parser("report"); q.add_argument("dirs", nargs="+")
    q.add_argument("--side-by-side", action="store_true", help="one comparison table of the end-to-end runs")
    args = parser.parse_args()
    load_env(args.env_file)
    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "translate":
        if not args.confirm_billable:
            sys.exit("refusing to run without --confirm-billable")
        asyncio.run(run_translate(args))
    elif args.cmd == "run":
        if not args.confirm_billable:
            sys.exit("refusing to run without --confirm-billable")
        asyncio.run(run_session(args))
    else:
        cmd_report(args)
        runs = [d for d in args.dirs if (Path(d) / "raw.jsonl").exists()]
        if args.side_by_side and len(runs) > 1:
            side_by_side(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

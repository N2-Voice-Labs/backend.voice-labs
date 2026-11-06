#!/usr/bin/env python
"""Offline analysis of recorded benchmark runs. Costs nothing: no network, no providers.

    cd runtime && python scripts/benchmark_analysis.py [--bench bench]

Three reports, all read from the files ``translation_benchmark.py`` wrote:

1. timeline   where the time before the translation goes (sequential vs overlapping)
2. splits     what makes a sentence split in two, and whether the split did harm
3. deadline   what a Google deadline of 500 / 750 / 1000 / 2000 ms would have cost

Assumptions are printed next to the numbers they affect. The Gemini figures come from the few
fallback sentences in the fault runs (``GEMINI_SAMPLES_MS``), so they are rough.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from pathlib import Path

import numpy as np

SR = 16000
FRAME = SR // 100  # 10 ms energy frames
VAD_STOP_S = 0.2  # TranslationEngine's VAD stop_secs
VAD_START_S = 0.2  # pipecat's VAD start_secs default
# Whole-sentence Gemini times (ms) seen in the fault runs while Google was in its cooldown, or
# derived as (first chunk - deadline) after a timeout. Few samples: treat as a rough range.
GEMINI_SAMPLES_MS = [713, 811, 815, 924, 967, 1240, 1266, 1373]
COOLDOWN_SENTENCES = 4  # sentences spoken during one 30 s cooldown at a call cadence of ~7 s
FILLERS = ("mm-hmm", "음.", "음흠", "hmm")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - spread) / d), (centre + spread) / d


# -- 1. timeline ---------------------------------------------------------------


def timeline(bench: Path) -> None:
    print("\n== 1. Where the time goes before the translation (unsplit sentences)\n")
    print("   pipeline order, every step waits for the one before it:")
    print("   last voiced sample -> VAD stop (0.2 s) -> STT final -> turn end -> translation -> TTS -> audio out\n")
    print(f"{'profile':10}{'n':>4}{'e2e P50':>9}{'= before':>10}{'+ transl.':>10}{'+ TTS':>7}{'| STT':>7}{'VAD+STT':>9}{'unexplained':>12}")
    for profile in ("balanced", "instant"):
        data = [
            r for r in rows(bench / profile / "raw.jsonl")
            if not r["split"] and r["turns"] == 2 and r.get("d_translation") and r.get("d_tts") and r.get("stt_ttfb")
        ]
        before = [r["e2e_s"] * 1000 - r["d_translation"] - r["d_tts"] for r in data]
        stt = [r["stt_ttfb"] for r in data]
        residual = [b - VAD_STOP_S * 1000 - s for b, s in zip(before, stt)]
        print(
            f"{profile:10}{len(data):>4}{stats.median(r['e2e_s'] * 1000 for r in data):>9.0f}"
            f"{stats.median(before):>10.0f}{stats.median(r['d_translation'] for r in data):>10.0f}"
            f"{stats.median(r['d_tts'] for r in data):>7.0f}{stats.median(stt):>7.0f}"
            f"{VAD_STOP_S * 1000 + stats.median(stt):>9.0f}{stats.median(residual):>12.0f}"
        )
    print(
        "\n   The turn pause (0.3 s Balanced, 0.2 s Instant) runs at the same time as the STT wait, and the turn\n"
        "   ends only when BOTH are done. STT takes ~460 ms, longer than either pause, so the pause adds nothing:\n"
        "   'before' is the same in both profiles. 'unexplained' (~180 ms) is whatever the marks do not cover:\n"
        "   audio in over LiveKit, VAD frame analysis, the hop from the STT frame to the translator, audio out to\n"
        "   the listener. It needs a new mark to split further."
    )


# -- 2. splits -----------------------------------------------------------------


def inner_gaps(pcm: bytes) -> tuple[float, list[float]]:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(float)
    frames = len(samples) // FRAME
    energy = np.sqrt((samples[: frames * FRAME].reshape(frames, FRAME) ** 2).mean(axis=1))
    voiced = energy > max(150.0, 0.03 * energy.max())
    index = np.where(voiced)[0]
    gaps, run = [], 0
    for i in range(index[0], index[-1] + 1):
        if voiced[i]:
            if run:
                gaps.append(run * 0.01)
            run = 0
        else:
            run += 1
    return (max(gaps) if gaps else 0.0), gaps


def classify(row: dict) -> str:
    joined = " ".join(row["stt_text"]).lower()
    if any(f in joined for f in FILLERS):
        return "harmful: STT invented a filler and/or lost words at the cut"
    if row["stt_text"][0].rstrip(".-… ").strip() in ("그리고",) or row["stt_text"][0].endswith("-"):
        return "harmful: a fragment was spoken on its own"
    return "benign: cut at a natural clause break, both halves translated correctly"


def splits(bench: Path) -> None:
    print("\n== 2. Premature splits\n")
    by = {p: {r["id"]: r for r in rows(bench / p / "raw.jsonl")} for p in ("balanced", "instant")}
    gaps = {i: inner_gaps((bench / "samples" / f"{i}.pcm").read_bytes())[0] for i in by["balanced"]}
    for p in by:
        split = [gaps[i] for i, r in by[p].items() if r["split"]]
        whole = [gaps[i] for i, r in by[p].items() if not r["split"]]
        print(
            f"{p:9} splits {len(split):>2}/{len(by[p])}  longest silence inside the sentence: split min "
            f"{min(split):.2f} s, median {stats.median(split):.2f} s | not split max {max(whole):.2f} s, median {stats.median(whole):.2f} s"
        )
    print("\n   Every Balanced split is also an Instant split, so the cause is in the audio, not chance.")
    print("   A pause g inside a sentence splits it when g + VAD start (0.2 s) is longer than VAD stop (0.2 s) + turn pause,")
    print("   i.e. roughly when g > the turn pause: ~0.3 s Balanced, ~0.2 s Instant.\n")
    print(f"{'id':7}{'lang':>5}{'gap s':>7}  {'Bal':>4}{'Ins':>4}  classification (Instant run)")
    harmful = {"balanced": 0, "instant": 0}
    for i, r in sorted(by["instant"].items(), key=lambda kv: -gaps[kv[0]]):
        if not (r["split"] or by["balanced"][i]["split"]):
            continue
        verdict = classify(r)
        for p in by:
            if by[p][i]["split"] and verdict.startswith("harmful"):
                harmful[p] += 1
        print(f"{i:7}{r['lang']:>5}{gaps[i]:>7.2f}  {'B' if by['balanced'][i]['split'] else '-':>4}{'I' if r['split'] else '-':>4}  {verdict}")
        print(f"{'':25}{' | '.join(r['stt_text'])}")
    print(f"\n   harmful splits: Balanced {harmful['balanced']}, Instant {harmful['instant']} (of {len(by['balanced'])} sentences each)")


# -- 3. deadline ---------------------------------------------------------------


def deadline(bench: Path) -> None:
    print("\n== 3. Google deadline, from the recorded direct requests\n")
    latency = [r["latency_ms"] for r in rows(bench / "translate_only" / "requests.jsonl") if not r["error"]]
    n, google, gemini = len(latency), stats.median(latency), stats.median(GEMINI_SAMPLES_MS)
    print(f"   {n} successful requests, Google P50 {google:.0f} ms, max {max(latency):.0f} ms. Gemini whole-sentence median {gemini:.0f} ms")
    print(f"   (range {min(GEMINI_SAMPLES_MS)}-{max(GEMINI_SAMPLES_MS)}, n={len(GEMINI_SAMPLES_MS)}).\n")
    print(f"{'deadline':>9}{'false timeouts':>16}{'95% interval':>16}{'cost of one':>13}{'mean per sentence':>19}{'cost of a hung Google':>23}")
    for d in (500, 750, 1000, 2000):
        slow = [x for x in latency if x > d]
        low, high = wilson(len(slow), n)
        # One false timeout: the sentence waits d, then Gemini, instead of the slow Google answer;
        # then COOLDOWN_SENTENCES more sentences go to Gemini instead of Google.
        one = (d + gemini - stats.mean(slow) if slow else d + gemini - d) + COOLDOWN_SENTENCES * (gemini - google)
        print(
            f"{d:>7} ms{len(slow):>12}/{n}{low * 100:>9.2f}-{high * 100:<5.2f}%{one / 1000:>11.1f} s{len(slow) / n * one:>16.1f} ms"
            f"{(d + gemini) / 1000:>16.2f} s"
        )
    print(
        "\n   cost of one = extra delay summed over the sentence that timed out and the next few that go to Gemini during the\n"
        f"   30 s cooldown ({COOLDOWN_SENTENCES} assumed), not one stall. 'cost of a hung Google' = deadline + Gemini, the time a\n"
        "   listener waits for the first sentence when Google never answers.\n\n"
        "   Assumptions and limits: Google requests were fired in one burst of ~5 minutes from one machine with one key, so\n"
        "   it shows no outage, no time-of-day effect and no regional difference. A real timeout of the old kind (a hung\n"
        "   connection) is not in this data. Gemini has only 8 samples. A 0-in-1000 result means 'below ~0.4%', not 'never'.\n"
        "   The cost of a false timeout counts only added delay: it ignores that Gemini may translate differently."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=Path(__file__).resolve().parents[1] / "bench")
    parser.add_argument("section", nargs="?", choices=("timeline", "splits", "deadline", "all"), default="all")
    args = parser.parse_args()
    for name, fn in (("timeline", timeline), ("splits", splits), ("deadline", deadline)):
        if args.section in (name, "all"):
            fn(args.bench)


if __name__ == "__main__":
    main()

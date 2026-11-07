#!/usr/bin/env python
"""Offline experiments for cinematic dubbing: no network, no providers, no credits.

    cd runtime && python scripts/dubbing_lab.py [--samples bench/samples] [--wav-dir DIR]

Runs ``DubMixer`` on a virtual clock over recorded utterances (the benchmark samples) and prints,
for several profiles and delay/duck settings, numbers that stand in for the listening criteria:

  presence      share of the speaker's voiced time that stays audible in the mix (> -40 dBFS)
  TOR dB        translation-to-original ratio over 100 ms windows while both play (median / 10th
                percentile): higher is clearer; the low end is the moment the duck is still closing
  duck s        time for the original to reach 90% of its ducking depth after the translation starts
  step dB       largest volume change of the original in one 20 ms block: smaller is smoother
  clip          samples at full scale after the mix (the limiter should keep this at 0)
  peak          highest sample, fraction of full scale
  us/blk, KiB   mixing time per 20 ms block and the most audio held in the mixer's buffers

These are proxies, not listening results. ``--wav-dir`` writes each mix as a WAV for a listening
study (keep those files out of the repository).
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys
import time
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from runtime.dubbing import BLOCK, BLOCK_MS, FULL_SCALE, PROFILES, RATE, DubMixer, DubbingProfile  # noqa: E402

PIPELINE_S = 1.26  # measured median: end of speech to first translated audio


def load(path: Path, seconds: float | None = None) -> np.ndarray:
    """A 16 kHz recording as 24 kHz int16 (linear interpolation is enough for a lab)."""

    data = np.frombuffer(path.read_bytes(), dtype=np.int16).astype(np.float32)
    if seconds is not None:
        data = data[: int(seconds * 16000)]
    n = int(len(data) * RATE / 16000)
    return np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.int16)


def rms_db(x: np.ndarray) -> float:
    return 20 * np.log10(max(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) / FULL_SCALE, 1e-9))


class Scene:
    """What arrives when: ``original`` chunks (10 ms, as LiveKit delivers) and ``translation`` bursts."""

    def __init__(self) -> None:
        self.original: list[tuple[float, np.ndarray]] = []
        self.translation: list[tuple[float, np.ndarray]] = []
        self.clear_at: float | None = None

    def speak(self, at: float, audio: np.ndarray, network_s: float = 0.16) -> float:
        """The speaker talks from ``at``; the Runtime hears it ``network_s`` later. Returns the end."""

        step = RATE // 100
        for i in range(0, len(audio), step):
            self.original.append((at + network_s + i / RATE, audio[i : i + step]))
        return at + len(audio) / RATE

    def translate(self, at: float, audio: np.ndarray) -> None:
        self.translation.append((at, audio))


def scenes(samples: Path) -> dict[str, Scene]:
    a, b, c = (load(samples / f"{name}.pcm") for name in ("u0000", "u0001", "u0002"))
    out: dict[str, Scene] = {}
    s = Scene()  # one sentence, translation after it
    end = s.speak(0.0, a)
    s.translate(end + PIPELINE_S, c)
    out["sentence"] = s
    s = Scene()  # the speaker goes on talking while the first sentence is being translated
    end = s.speak(0.0, a)
    s.translate(end + PIPELINE_S, c)
    s.speak(end + 0.6, b)
    out["continuous"] = s
    s = Scene()  # a long monologue under a short translation: how the original comes back afterwards
    long = np.concatenate([a, b, a])
    s.speak(0.0, long)
    s.translate(4.0, load(samples / "u0003.pcm", seconds=1.5))
    out["recovery"] = s
    s = Scene()  # a very short utterance
    short = load(samples / "u0000.pcm", seconds=0.6)
    end = s.speak(0.0, short)
    s.translate(end + PIPELINE_S, load(samples / "u0003.pcm", seconds=0.8))
    out["short"] = s
    s = Scene()  # the translation is cut off one second in (an interruption)
    end = s.speak(0.0, a)
    s.translate(end + PIPELINE_S, c)
    s.clear_at = end + PIPELINE_S + 1.0
    s.speak(end + PIPELINE_S + 1.0, b)
    out["interrupted"] = s
    return out


def run(profile: DubbingProfile | None, scene: Scene) -> dict:
    """Render a scene on a virtual clock. ``profile=None`` is the translation-only baseline."""

    with_orig = DubMixer(profile) if profile else None
    alone = DubMixer(profile or PROFILES["balanced"])  # the same translation without the original
    events = sorted(
        [(t, 0, "o", x) for t, x in scene.original] + [(t, 1, "t", x) for t, x in scene.translation]
        + ([(scene.clear_at, 2, "c", None)] if scene.clear_at is not None else []),
        key=lambda e: (e[0], e[1]),
    )
    end_t = max([t + len(x) / RATE for t, x in scene.translation] + [t + len(x) / RATE for t, x in scene.original]) + 3.0
    mix, tr, orig_contrib, gains = [], [], [], []
    cost, peak_bytes, i, t = [], 0, 0, 0.0
    block_s = BLOCK_MS / 1000
    while t < end_t:
        while i < len(events) and events[i][0] <= t:
            _, _, kind, x = events[i]
            i += 1
            if kind == "o":
                if with_orig:
                    with_orig.push_original(x.tobytes(), events[i - 1][0])
            elif kind == "t":
                alone.push_translation(x.tobytes())
                if with_orig:
                    with_orig.push_translation(x.tobytes())
            else:
                alone.clear_translation()
                if with_orig:
                    with_orig.clear_translation()
        started = time.perf_counter()
        m = np.frombuffer((with_orig or alone).render(t), dtype=np.int16)
        cost.append((time.perf_counter() - started) * 1e6)
        tr_only = np.frombuffer(alone.render(t), dtype=np.int16) if with_orig else m
        if with_orig:
            peak_bytes = max(peak_bytes, with_orig.buffered_bytes)
            gains.append(with_orig.gain)
        mix.append(m)
        tr.append(tr_only)
        orig_contrib.append(m.astype(np.float32) - tr_only.astype(np.float32))
        t += block_s
    mixed = np.concatenate(mix)
    translation = np.concatenate(tr)
    original = np.concatenate(orig_contrib)
    ratios = []
    for w in range(0, len(tr) - 4, 5):  # 100 ms windows
        tw, ow = np.concatenate(tr[w : w + 5]), np.concatenate(orig_contrib[w : w + 5])
        if rms_db(tw) > -40 and rms_db(ow) > -70:
            ratios.append(rms_db(tw) - rms_db(ow))
    # presence: how much of the speaker's voiced time can be heard in the mix
    heard = voiced = 0
    speaker = np.zeros(len(mixed), dtype=np.float32)
    for at, chunk in scene.original:
        start = int((at + (profile.delay_ms / 1000 if profile else 0)) * RATE)
        speaker[start : start + len(chunk)] = np.abs(chunk)[: max(0, len(speaker) - start)]
    frame = BLOCK
    blocks = len(mixed) // frame
    for b in range(blocks):
        if rms_db(speaker[b * frame : (b + 1) * frame]) > -45:
            voiced += 1
            heard += rms_db(original[b * frame : (b + 1) * frame]) > -40
    duck_s = None
    if with_orig and scene.translation:
        start_block = int((scene.translation[0][0]) / block_s)
        g0 = gains[max(0, start_block - 1)] if start_block - 1 < len(gains) else with_orig.gain
        target = PROFILES["balanced"] and (with_orig.profile.duck_gain_db)
        depth = g0 - 10 ** (target / 20)
        for k in range(start_block, min(len(gains), start_block + 100)):
            if depth > 0 and g0 - gains[k] >= 0.9 * depth:
                duck_s = (k - start_block) * block_s
                break
    steps = np.abs(np.diff(20 * np.log10(np.maximum(gains, 1e-6)))) if gains else np.array([0.0])
    return {
        "mix": mixed,
        "presence": heard / voiced if voiced and with_orig else (0.0 if voiced else float("nan")),
        "tor_med": stats.median(ratios) if ratios else float("nan"),
        "tor_min": float(np.percentile(ratios, 10)) if ratios else float("nan"),
        "duck_s": duck_s,
        "step_db": float(steps.max()),
        "clip": int((np.abs(mixed.astype(np.int32)) >= 32767).sum()),
        "peak": float(np.abs(mixed).max() / FULL_SCALE),
        "us_block": stats.mean(cost),
        "kib": peak_bytes / 1024,
    }


def table(title: str, rows: list[tuple[str, DubbingProfile | None]], scene: Scene, wav_dir: Path | None, tag: str) -> None:
    print(f"\n{title}")
    print(f"{'profile':26}{'presence':>9}{'TOR med':>9}{'TOR p10':>9}{'duck s':>8}{'step dB':>9}{'clip':>6}{'peak':>6}{'us/blk':>8}{'KiB':>6}")
    for label, profile in rows:
        r = run(profile, scene)
        duck = f"{r['duck_s']:.2f}" if r["duck_s"] is not None else "-"
        print(f"{label:26}{r['presence']:>9.0%}{r['tor_med']:>9.1f}{r['tor_min']:>9.1f}{duck:>8}{r['step_db']:>9.2f}{r['clip']:>6}{r['peak']:>6.2f}{r['us_block']:>8.0f}{r['kib']:>6.0f}")
        if wav_dir is not None:
            wav_dir.mkdir(parents=True, exist_ok=True)
            with wave.open(str(wav_dir / f"{tag}_{label.replace(' ', '_').replace('=', '')}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(r["mix"].tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=Path(__file__).resolve().parents[1] / "bench" / "samples")
    parser.add_argument("--wav-dir", type=Path, default=None)
    args = parser.parse_args()
    all_scenes = scenes(args.samples)
    presets = [("translation only", None), *[(name, p) for name, p in PROFILES.items()]]
    for name, scene in all_scenes.items():
        table(f"scenario: {name}", presets, scene, args.wav_dir, name)
    base = PROFILES["balanced"]
    cont = all_scenes["continuous"]
    table("delay sweep (balanced, continuous speech)", [(f"delay={d}", replace(base, delay_ms=d)) for d in (0, 150, 300, 600, 1200)], cont, args.wav_dir, "delay")
    table("duck depth sweep (balanced)", [(f"duck={d}", replace(base, duck_gain_db=d)) for d in (-16, -20, -24, -30, -36)], cont, args.wav_dir, "duck")
    table("base level sweep (balanced)", [(f"base={b}", replace(base, base_gain_db=b)) for b in (-4, -8, -12, -16, -20)], cont, args.wav_dir, "base")
    table("attack sweep (balanced)", [(f"attack={a}", replace(base, attack_ms=a)) for a in (20, 40, 80, 150, 250)], cont, args.wav_dir, "attack")
    table("release sweep (balanced, long speech under a short translation)", [(f"release={r}", replace(base, release_ms=r)) for r in (150, 300, 500, 900, 1500)], all_scenes["recovery"], args.wav_dir, "release")


if __name__ == "__main__":
    main()

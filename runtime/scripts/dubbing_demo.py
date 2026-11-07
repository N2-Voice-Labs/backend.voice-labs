#!/usr/bin/env python
"""Listening demonstrations for cinematic dubbing: no network, no providers, no credits.

    cd runtime && python scripts/dubbing_demo.py --out bench/dubbing/listening [--blind]

Renders the SAME conversation (the same Korean speech and the same English translation clips,
from ``bench/samples``) in five presentations, so a listener compares the audio presentation and
nothing else:

  ai_only         translation only, as today
  live_balanced   the shipped prototype: the live original under the translation (profile balanced)
  replay_brief    exploratory: the first ~2 s of the original replayed under the start of the translation
  replay_full     exploratory: the whole original replayed under the translation
  live_plus_brief exploratory: live original (subtle) and a brief replay at the translation's start

Scenarios: short, sentence, continuous (three sentences, the speaker talks on while the first is
translated), interrupted (the translation is cut off one second in).

The two replay variants are NOT production code: they need an utterance paired with its translation,
which the live path does not have. They exist only to be heard. The translation is never moved
or shortened; every presentation uses the identical translation signal.

Outputs, per scenario and presentation: a WAV (24 kHz mono) and an MP3, plus ``index.html`` (labelled
player), ``metrics.json`` and, with ``--blind``, ``blind/`` with shuffled file names and ``key.json``.
The numbers it prints are proxies for the listening criteria, not results.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import dubbing_lab as lab  # noqa: E402
from runtime.dubbing import FULL_SCALE, PROFILES, RATE, _soft_limit  # noqa: E402

PIPELINE_S = lab.PIPELINE_S
TARGET_DB = -26.0  # voiced RMS of every clip, so the comparison is about presentation, not recording level

# (Korean id, English id): the same sentence in both languages (checked against corpus.json).
PAIRS = {
    "short": ("u0158", "u0135"),      # 6명이 갈 건데 자리가 있을까요? / There will be 6 of us, do you have a table?
    "airport": ("u0098", "u0151"),    # 저기요, 공항까지... / Excuse me, What is the fastest way to get to the airport...
    "seoul": ("u0018", "u0043"),      # 그리고 저는 다음 달에 서울로... / Also, I am going to Seoul on a business trip...
    "lawyer": ("u0086", "u0019"),     # 그런데 계약서에 서명하기 전에... / By the way, I would like a lawyer...
    "allergy": ("u0110", "u0131"),    # 그런데 알레르기가... / By the way, I am allergic to nuts...
}

PRESENTATIONS = ("ai_only", "live_balanced", "replay_brief", "replay_full", "live_plus_brief")
LABELS = {
    "ai_only": "AI only (today)",
    "live_balanced": "Live original, ducked (prototype)",
    "replay_brief": "Replay, brief (exploratory)",
    "replay_full": "Replay, whole utterance (exploratory)",
    "live_plus_brief": "Live (subtle) + brief replay (exploratory)",
}
REPLAY = {"start_db": -12.0, "hold_db": -22.0, "ramp_s": 0.15, "fade_s": 0.8, "brief_s": 2.0}


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


def rms_db(x: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) / FULL_SCALE) if x.size else -120.0


def normalise(pcm: np.ndarray) -> np.ndarray:
    """Scale so the voiced part sits at TARGET_DB."""

    x = pcm.astype(np.float32)
    frames = x[: len(x) // 480 * 480].reshape(-1, 480)
    levels = np.array([rms_db(f) for f in frames])
    voiced = frames[levels > levels.max() - 30]
    gain = 10 ** ((TARGET_DB - rms_db(voiced)) / 20)
    return np.clip(x * gain, -FULL_SCALE, FULL_SCALE - 1).astype(np.int16)


class Clips:
    def __init__(self, samples: Path) -> None:
        self.samples = samples
        self.text = {i["id"]: i["text"] for i in json.loads((samples / "corpus.json").read_text())["items"]}

    def get(self, uid: str) -> np.ndarray:
        return normalise(lab.load(self.samples / f"{uid}.pcm"))


class Conversation:
    """Speech and translations on one timeline, in the same units as ``dubbing_lab.Scene``."""

    def __init__(self, clips: Clips, plan: list[tuple[str, float]], clear_after: tuple[int, float] | None = None) -> None:
        self.scene = lab.Scene()
        self.speech: list[tuple[float, float, np.ndarray]] = []  # start, end, audio
        self.translations: list[tuple[float, float, np.ndarray]] = []  # play start, play end, audio
        self.texts: list[tuple[str, str]] = []
        t = 0.0
        prev_end = 0.0
        for index, (pair, pause_before) in enumerate(plan):
            ko, en = PAIRS[pair]
            speech, translation = clips.get(ko), clips.get(en)
            t += pause_before
            end = self.scene.speak(t, speech)
            self.speech.append((t, end, speech))
            arrival = end + PIPELINE_S
            self.scene.translate(arrival, translation)
            start = max(arrival, prev_end)
            prev_end = start + len(translation) / RATE
            self.translations.append((start, prev_end, translation))
            self.texts.append((clips.text[ko], clips.text[en]))
            t = end
        self.clear_at: float | None = None
        if clear_after is not None:
            k, offset = clear_after
            self.clear_at = self.translations[k][0] + offset
            self.scene.clear_at = self.clear_at
            # the translation is cut short, so it ends then; later translations are not part of this scene
            start, _, audio = self.translations[k]
            self.translations[k] = (start, self.clear_at, audio)


def replay_stem(conv: Conversation, length: int, whole: bool) -> np.ndarray:
    """The original voice replayed under each translation, from the translation's first sample."""

    out = np.zeros(length, dtype=np.float32)
    for (s_start, s_end, speech), (t_start, t_end, _) in zip(conv.speech, conv.translations):
        play_s = min((len(speech) / RATE) if whole else REPLAY["brief_s"], len(speech) / RATE)
        if whole:
            play_s = min(play_s, max(t_end - t_start, 0.0))  # never outlasts its translation
        if play_s <= 0:
            continue
        x = speech[: int(play_s * RATE)].astype(np.float32)
        t = np.arange(len(x)) / RATE
        g_db = np.where(t < REPLAY["ramp_s"], REPLAY["start_db"] + (REPLAY["hold_db"] - REPLAY["start_db"]) * t / REPLAY["ramp_s"], REPLAY["hold_db"])
        fade = np.clip((play_s - t) / REPLAY["fade_s"], 0, 1)
        fade = np.sin(fade * np.pi / 2) ** 2
        if conv.clear_at is not None and t_start <= conv.clear_at < t_start + play_s:
            fade *= np.clip((conv.clear_at - t_start + 0.08 - t) / 0.08, 0, 1)  # cut with a short fade
        y = x * 10 ** (g_db / 20) * fade
        a = int(t_start * RATE)
        out[a : a + len(y)] += y[: max(0, length - a)]
    return out


def render(conv: Conversation) -> dict[str, dict[str, np.ndarray]]:
    """Every presentation as {mix, translation, original} (int16-range float arrays, 24 kHz)."""

    ai = lab.run(None, conv.scene)["mix"].astype(np.float32)
    n = len(ai)
    out = {"ai_only": {"mix": ai, "translation": ai, "original": np.zeros(n, np.float32)}}

    def live(profile) -> np.ndarray:
        mixed = lab.run(profile, conv.scene)["mix"].astype(np.float32)
        return mixed[:n] if len(mixed) >= n else np.pad(mixed, (0, n - len(mixed)))

    balanced = live(PROFILES["balanced"])
    out["live_balanced"] = {"mix": balanced, "translation": ai, "original": balanced - ai}
    for name, whole in (("replay_brief", False), ("replay_full", True)):
        stem = replay_stem(conv, n, whole)
        out[name] = {"mix": _soft_limit(ai + stem).astype(np.float32), "translation": ai, "original": stem}
    subtle = live(PROFILES["subtle"])
    stem = subtle - ai + replay_stem(conv, n, False)
    out["live_plus_brief"] = {"mix": _soft_limit(ai + stem).astype(np.float32), "translation": ai, "original": stem}
    return out


def window_levels(x: np.ndarray, win: int = RATE // 10) -> np.ndarray:
    n = len(x) // win
    return np.array([rms_db(x[i * win : (i + 1) * win]) for i in range(n)])


CHAINS = {  # name: (encode arguments, decode input arguments); "ref" chains carry no codec error
    "mulaw": (["-c:a", "pcm_mulaw", "-f", "mulaw", "-ar", "8000"], ["-f", "mulaw", "-ar", "8000", "-ac", "1"]),
    "mulaw_ref": (["-c:a", "pcm_s16le", "-f", "s16le", "-ar", "8000"], ["-f", "s16le", "-ar", "8000", "-ac", "1"]),
    "opus12": (["-c:a", "libopus", "-b:a", "12k", "-application", "voip", "-ar", "16000", "-f", "ogg"], []),
    "opus24": (["-c:a", "libopus", "-b:a", "24k", "-application", "voip", "-ar", "16000", "-f", "ogg"], []),
    "opus24_ref": (["-c:a", "pcm_s16le", "-f", "s16le", "-ar", "16000"], ["-f", "s16le", "-ar", "16000", "-ac", "1"]),
    "opus12_ref": (["-c:a", "pcm_s16le", "-f", "s16le", "-ar", "16000"], ["-f", "s16le", "-ar", "16000", "-ac", "1"]),
}


def chain(x: np.ndarray, kind: str) -> np.ndarray:
    """``x`` through a telephone-style chain (8 kHz mu-law, or 16 kHz Opus voice at 24 or 12 kbit/s) back to 24 kHz.

    The ``_ref`` chains do the same resampling without the codec, so the difference between a chain
    and its reference is the codec's own error.
    """

    pcm = np.clip(x, -FULL_SCALE, FULL_SCALE - 1).astype(np.int16).tobytes()
    encode, decode = CHAINS[kind]
    enc = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "-", *encode, "-"],
                         input=pcm, capture_output=True, check=True).stdout
    out = subprocess.run(["ffmpeg", "-v", "error", *decode, "-i", "-", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-"],
                         input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32)


def codec_error(mix: np.ndarray, original: np.ndarray, kind: str) -> float | None:
    """Share of the windows where the original is audible in which it stands >= 6 dB above the codec's error."""

    decoded, ref = chain(mix, kind), chain(mix, f"{kind}_ref")
    ref_orig = chain(original, f"{kind}_ref")
    k = min(len(decoded), len(ref), len(ref_orig))
    lag = max(range(-1440, 1441, 48), key=lambda g: float(np.dot(decoded[max(g, 0):k + min(g, 0)], ref[max(-g, 0):k - max(g, 0)])))  # Opus delay
    d, r = decoded[max(lag, 0):k + min(lag, 0)], ref[max(-lag, 0):k - max(lag, 0)]
    o = ref_orig[max(-lag, 0):k - max(lag, 0)]
    err, level = window_levels(d - r), window_levels(o)
    n = min(len(err), len(level))
    heard = level[:n] > -55
    return round(float(((level[:n] - err[:n]) > 6)[heard].mean()), 2) if heard.any() else None


def metrics(conv: Conversation, stems: dict[str, np.ndarray]) -> dict:
    mix, tr, orig = stems["mix"], stems["translation"], stems["original"]
    L = min(len(mix), len(tr), len(orig))
    mix, tr, orig = mix[:L], tr[:L], orig[:L]
    m, t, o = window_levels(mix), window_levels(tr), window_levels(orig)
    audible = m > -50
    t_end = max(end for _, end, _ in conv.translations)
    span = int(t_end * 10)
    gaps, run = [], 0
    for a in audible[:span]:
        run = 0 if a else run + 1
        gaps.append(run)
    silence = float((~audible[:span]).sum() / 10)
    longest = max(gaps) / 10 if gaps else 0.0
    first = next((i for i, a in enumerate(audible) if a), None)
    both = (t > -40) & (o > -55)
    ratios = (t - o)[both]
    energy_o = float(np.sum(orig.astype(np.float64) ** 2))
    energy_t = float(np.sum(tr.astype(np.float64) ** 2))
    o_only = (o > -55) & ~(t > -40)
    result = {
        "first_sound_s": None if first is None else round(first / 10, 2),
        "silence_s": round(silence, 2),
        "longest_silence_s": round(longest, 2),
        "original_audible_s": round(float((o > -55).sum() / 10), 2),
        "original_alone_s": round(float(o_only.sum() / 10), 2),
        "co_presence_s": round(float(both.sum() / 10), 2),
        "tor_median_db": round(float(np.median(ratios)), 1) if ratios.size else None,
        "tor_p10_db": round(float(np.percentile(ratios, 10)), 1) if ratios.size else None,
        "original_energy_share": round(energy_o / (energy_o + energy_t), 3) if energy_o + energy_t else 0.0,
        "peak": round(float(np.abs(mix).max() / FULL_SCALE), 2),
        "clipped": int((np.abs(mix) >= FULL_SCALE - 1).sum()),
    }
    # telephone: is the original still above the codec's own error where it is meant to be audible?
    if (o > -55).any():
        for kind in ("mulaw", "opus24", "opus12"):
            result[f"phone_{kind}_original_above_codec_error"] = codec_error(mix, orig, kind)
    return result


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = np.clip(x, -FULL_SCALE, FULL_SCALE - 1).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def mp3(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:a", "libmp3lame", "-b:a", "64k", "-ar", "24000", str(dst)], check=True)


SCENARIOS = {
    "short": ("One short utterance.", [("short", 0.3)], None),
    "sentence": ("One sentence; the original has ended before the translation begins.", [("airport", 0.3)], None),
    "continuous": ("Three sentences with short pauses: the speaker talks on while the first is translated.",
                   [("seoul", 0.3), ("lawyer", 0.5), ("allergy", 0.5)], None),
    "interrupted": ("The translation of the first sentence is cut off one second in; the speaker goes on.",
                    [("lawyer", 0.3), ("seoul", 0.3)], (0, 1.0)),
}


def page(rows: list[dict], note: str) -> str:
    cards = []
    for sc in rows:
        items = "".join(
            f'<tr><td>{LABELS[p]}</td><td><audio controls preload="none" src="{sc["name"]}_{p}.mp3"></audio></td></tr>'
            for p in PRESENTATIONS)
        cards.append(f'<section><h2>{sc["name"]}</h2><p>{sc["blurb"]}</p><p class="t">{sc["text"]}</p><table>{items}</table></section>')
    return f"""<!doctype html><meta charset="utf-8"><title>Dubbing listening demos</title>
<style>body{{font:16px system-ui;max-width:880px;margin:2rem auto;padding:0 1rem}}td{{padding:.3rem .8rem .3rem 0}}.t{{color:#555;font-size:.9em}}
section{{margin-bottom:2rem}}audio{{width:380px;max-width:100%}}</style>
<h1>Dubbing listening demos</h1><p>{note}</p>{''.join(cards)}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=HERE.parent / "bench" / "samples")
    parser.add_argument("--out", type=Path, default=HERE.parent / "bench" / "dubbing" / "listening")
    parser.add_argument("--blind", action="store_true", help="also write blind/ with shuffled names and key.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--keep-wav", action="store_true", help="keep the WAV files next to the MP3s")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required for the MP3s and the telephone codec simulation")
    clips = Clips(args.samples)
    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}
    rows = []
    for name, (blurb, plan, clear) in SCENARIOS.items():
        conv = Conversation(clips, plan, clear)
        stems = render(conv)
        all_metrics[name] = {}
        for p in PRESENTATIONS:
            wav = args.out / f"{name}_{p}.wav"
            write_wav(wav, stems[p]["mix"])
            mp3(wav, args.out / f"{name}_{p}.mp3")
            if not args.keep_wav:
                wav.unlink()
            all_metrics[name][p] = metrics(conv, stems[p])
        rows.append({"name": name, "blurb": blurb, "text": " / ".join(f"{ko} → {en}" for ko, en in conv.texts)})
        print(f"\n{name}: {blurb}")
        keys = ["first_sound_s", "silence_s", "longest_silence_s", "original_audible_s", "original_alone_s", "co_presence_s",
                "tor_median_db", "tor_p10_db", "original_energy_share", "clipped", "phone_mulaw_original_above_codec_error",
                "phone_opus24_original_above_codec_error", "phone_opus12_original_above_codec_error"]
        print(f"{'':18}" + "".join(f"{k[:11]:>12}" for k in keys))
        for p in PRESENTATIONS:
            print(f"{p:18}" + "".join(f"{str(all_metrics[name][p].get(k)):>12}" for k in keys))
    note = ("Same Korean speech and the same English translation in every row; only the audio presentation changes. "
            "Both voices are synthetic recordings, so the 'human' quality of the original is only approximated here. Use headphones.")
    (args.out / "index.html").write_text(page(rows, note))
    (args.out / "metrics.json").write_text(json.dumps({"pairs": PAIRS, "replay": REPLAY, "metrics": all_metrics}, indent=1))
    if args.blind:
        rng = random.Random(args.seed)
        blind = args.out / "blind"
        blind.mkdir(exist_ok=True)
        key = {}
        for name in SCENARIOS:
            order = list(PRESENTATIONS)
            rng.shuffle(order)
            for i, p in enumerate(order):
                clip = f"{name}_{'abcde'[i]}.mp3"
                shutil.copy(args.out / f"{name}_{p}.mp3", blind / clip)
                key[clip] = p
        (blind / "key.json").write_text(json.dumps(key, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

import json
import stat
import sys
import wave
from array import array
from pathlib import Path

import pytest

from src.evals import generate_audio as ga
from src.evals.dataset import load_manifest, DEFAULT_MANIFEST

FAKE_SAY = '''#!{python}
import os, struct, sys, wave, math
args = sys.argv[1:]
voice = args[args.index("-v") + 1]
out = args[args.index("-o") + 1]
text = args[args.index("--") + 1]
with open(os.environ["FAKE_SAY_LOG"], "a", encoding="utf-8") as log:
    log.write(voice + "\\t" + text + "\\n")
rate = 16000
frames = int(rate * 0.05 * max(len(text), 1))
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 8))) for i in range(frames)))
'''


@pytest.fixture
def fake_say(tmp_path, monkeypatch):
    script = tmp_path / "say"
    script.write_text(FAKE_SAY.format(python=sys.executable))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "say.log"
    monkeypatch.setenv("FAKE_SAY_LOG", str(log))
    return script, log


def _calls(log: Path) -> list[tuple[str, str]]:
    return [tuple(line.split("\t", 1)) for line in log.read_text(encoding="utf-8").splitlines()]


def test_split_by_script_for_code_switched_text() -> None:
    assert ga.split_by_script("Мне нужно book an appointment на завтра.") == [
        ("ru", "Мне нужно"),
        ("en", "book an appointment"),
        ("ru", "на завтра."),
    ]
    assert ga.split_by_script("Hello, мир") == [("en", "Hello,"), ("ru", "мир")]
    assert ga.segments_for("ko", "안녕하세요") == [("ko", "안녕하세요")]


def test_phone_and_noise_degradation_properties() -> None:
    import math

    rate = ga.SAMPLE_RATE
    tone_low = array("h", (int(8000 * math.sin(2 * math.pi * 1000 * i / rate)) for i in range(rate)))
    tone_high = array("h", (int(8000 * math.sin(2 * math.pi * 6000 * i / rate)) for i in range(rate)))
    phone_low = ga.phone_quality(tone_low)
    phone_high = ga.phone_quality(tone_high)
    assert len(phone_low) == len(tone_low)
    assert ga.rms(phone_low) > 0.5 * ga.rms(tone_low)  # 1 kHz is inside the phone band
    assert ga.rms(phone_high) < 0.1 * ga.rms(tone_high)  # 6 kHz is cut

    noisy = ga.add_white_noise(tone_low, 10, seed=1)
    noise = array("h", (a - b for a, b in zip(noisy, tone_low)))
    snr = 20 * math.log10(ga.rms(tone_low) / ga.rms(noise))
    assert snr == pytest.approx(10, abs=0.7)
    assert ga.add_white_noise(tone_low, 10, seed=1) == noisy  # deterministic


def test_synthesize_with_fake_say(fake_say) -> None:
    script, log = fake_say
    samples = ga.synthesize_segment("hello", "Samantha", say_bin=str(script))
    assert len(samples) == int(16000 * 0.05 * 5)
    assert _calls(log) == [("Samantha", "hello")]


def test_generate_renders_wavs_manifest_and_voices(fake_say, tmp_path) -> None:
    script, log = fake_say
    items = [
        i for i in load_manifest(DEFAULT_MANIFEST)
        if i["id"] in {"en-ko-01", "mixed-01", "interruptions-04", "noisy-phone-01"}
    ]
    all_items = load_manifest(DEFAULT_MANIFEST)
    out = tmp_path / "audio"
    records = ga.generate(items, all_items, out, dataset_version="v1", say_bin=str(script), log=lambda _: None)

    assert {r["id"] for r in records} == {i["id"] for i in items}
    for record in records:
        with wave.open(str(out / record["file"]), "rb") as handle:
            assert handle.getframerate() == 16000 and handle.getnchannels() == 1
            assert handle.getnframes() / 16000 == pytest.approx(record["duration_s"], abs=0.001)
        assert record["synthetic"] is True

    rows = [json.loads(line) for line in (out / "audio_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in rows}
    assert by_id["noisy-phone-01"]["degradation"] == {"type": "noise", "snr_db": 10}
    assert by_id["mixed-01"]["voices"] == ["Milena", "Samantha"]
    assert by_id["interruptions-04"]["voices"] == ["Yuna"]
    assert by_id["en-ko-01"]["voices"] == ["Samantha"]
    # interruption = two utterances, noisy variant re-synthesizes its parent text
    calls = _calls(log)
    assert ("Yuna", "아니, 죄송해요. 부산이 아니라 대구로 바꿔 주세요.") in calls
    assert sum(1 for _, text in calls if text.startswith("Hello, I would like to make an appointment")) == 2
    # mixed item rendered per script with the right voice
    assert ("Milena", "Мне нужно") in calls and ("Samantha", "book an appointment") in calls


def test_generate_reuses_existing_files_unless_forced(fake_say, tmp_path) -> None:
    script, log = fake_say
    all_items = load_manifest(DEFAULT_MANIFEST)
    items = all_items[:1]
    out = tmp_path / "audio"
    ga.generate(items, all_items, out, dataset_version="v1", say_bin=str(script), log=lambda _: None)
    first_calls = len(_calls(log))
    ga.generate(items, all_items, out, dataset_version="v1", say_bin=str(script), log=lambda _: None)
    assert len(_calls(log)) == first_calls
    ga.generate(items, all_items, out, dataset_version="v1", say_bin=str(script), force=True, log=lambda _: None)
    assert len(_calls(log)) == first_calls * 2
    rows = (out / "audio_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_voice_overrides_and_cli(fake_say, tmp_path) -> None:
    script, log = fake_say
    out = tmp_path / "cli"
    code = ga.main(["--out", str(out), "--categories", "ko-en", "--limit", "1",
                    "--voice", "ko=Eddy", "--say-bin", str(script)])
    assert code == 0
    assert _calls(log)[0][0] == "Eddy"
    assert (out / "ko-en-01.wav").exists()
    with pytest.raises(ValueError):
        ga.parse_voice_overrides(["de=Anna"])

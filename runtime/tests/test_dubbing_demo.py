"""The listening demos must differ only in presentation: same speech, same translation, no clipping."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "bench" / "samples"

pytestmark = pytest.mark.skipif(not (SAMPLES / "u0158.pcm").exists(), reason="benchmark samples not present")


@pytest.fixture(scope="module")
def demo():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("dubbing_demo", ROOT / "scripts" / "dubbing_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dubbing_demo"] = module
    spec.loader.exec_module(module)
    return module


def conversation(demo, plan, clear=None):
    return demo.Conversation(demo.Clips(SAMPLES), plan, clear)


def test_every_presentation_carries_the_identical_translation(demo):
    stems = demo.render(conversation(demo, [("airport", 0.3), ("seoul", 0.5)]))
    assert set(stems) == set(demo.PRESENTATIONS)
    for name, parts in stems.items():
        assert np.array_equal(parts["translation"], stems["ai_only"]["translation"]), name
    assert not stems["ai_only"]["original"].any()
    assert np.array_equal(stems["ai_only"]["mix"], stems["ai_only"]["translation"])


def test_no_presentation_clips(demo):
    stems = demo.render(conversation(demo, [("seoul", 0.3), ("lawyer", 0.4), ("allergy", 0.4)]))
    for name, parts in stems.items():
        assert np.abs(parts["mix"]).max() < demo.FULL_SCALE - 1, name


def test_the_translation_is_never_moved_for_the_original(demo):
    conv = conversation(demo, [("airport", 0.3)])
    start, _, _ = conv.translations[0]
    end_of_speech = conv.speech[0][1]
    assert start == pytest.approx(end_of_speech + demo.PIPELINE_S)  # same arrival as translation-only


def test_replay_starts_with_the_translation_and_ends_with_it(demo):
    conv = conversation(demo, [("airport", 0.3)])
    n = int((conv.translations[0][1] + 3) * demo.RATE)
    stem = demo.replay_stem(conv, n, whole=True)
    t0, t1, _ = conv.translations[0]
    active = np.flatnonzero(np.abs(stem) > 1)
    assert active[0] / demo.RATE >= t0 - 1e-3
    assert active[-1] / demo.RATE <= t1 + 1e-3
    assert not demo.replay_stem(conv, n, whole=False)[: int(t0 * demo.RATE)].any()


def test_an_interrupted_translation_takes_its_replay_with_it(demo):
    conv = conversation(demo, [("lawyer", 0.3), ("seoul", 0.3)], clear=(0, 1.0))
    n = int((conv.translations[1][1] + 3) * demo.RATE)
    stem = demo.replay_stem(conv, n, whole=True)
    cut = conv.clear_at
    first_end = int((cut + 0.15) * demo.RATE)
    second_start = int(conv.translations[1][0] * demo.RATE)
    assert not stem[first_end:second_start].any()


def test_brief_replay_is_brief(demo):
    conv = conversation(demo, [("lawyer", 0.3)])
    n = int((conv.translations[0][1] + 3) * demo.RATE)
    stem = demo.replay_stem(conv, n, whole=False)
    active = np.flatnonzero(np.abs(stem) > 1)
    assert (active[-1] - active[0]) / demo.RATE <= demo.REPLAY["brief_s"] + 1e-3

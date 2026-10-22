import json
import wave
from array import array
from pathlib import Path

import httpx
import pytest

from src.evals import live_latency as live
from src.evals.dataset import DEFAULT_MANIFEST, load_manifest


def pcm(*segments: tuple[int, float]) -> bytes:
    """PCM16 16 kHz from (amplitude, seconds) segments."""

    out = array("h")
    for amplitude, seconds in segments:
        out.extend([amplitude] * int(16000 * seconds))
    return out.tobytes()


def test_plan_skips_realtime_auto_and_keeps_the_rest():
    cells, skipped = live.plan_cells(["realtime", "cascaded"], ["instant", "balanced"],
                                     ["one_way", "auto"], ["ru-en"])
    assert len(cells) == 6 and len(skipped) == 2
    assert all(c.engine == "realtime" and c.direction == "auto" for c, _ in skipped)
    assert live.interpreter_config(cells[0]) == {
        "languages": ["ru", "en"], "direction": "one_way", "engine": "realtime", "latency_mode": "instant"}


def test_split_csv_rejects_unknown_values():
    assert live.split_csv("instant, balanced", live.MODES, "mode") == ["instant", "balanced"]
    with pytest.raises(ValueError, match="unknown mode"):
        live.split_csv("fast", live.MODES, "mode")


def test_pick_utterances_cycles_and_alternates_auto():
    items = load_manifest(DEFAULT_MANIFEST)
    one_way = live.pick_utterances(items, live.Cell("cascaded", "balanced", "one_way", "ko-en"), 25)
    assert len(one_way) == 25 and {e["item"]["category"] for e in one_way} == {"ko-en"}
    assert one_way[0]["item"]["id"] == one_way[20]["item"]["id"]  # 20 items, then it cycles
    auto = live.pick_utterances(items, live.Cell("cascaded", "balanced", "auto", "en-ru"), 4)
    assert [e["item"]["category"] for e in auto] == ["en-ru", "ru-en", "en-ru", "ru-en"]
    assert [e["speaker"] for e in auto] == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        live.pick_utterances(items, live.Cell("cascaded", "balanced", "one_way", "en-ja"), 1)


def test_speech_bounds_trims_silence():
    clip = pcm((0, 0.5), (3000, 1.0), (0, 0.8))
    start, end = live.speech_bounds(clip)
    assert (start, end) == pytest.approx((0.5, 1.5), abs=0.011)
    assert live.speech_bounds(pcm((0, 1.0))) == (0.0, 1.0)


def test_read_wav_requires_16k_mono(tmp_path: Path):
    good, bad = tmp_path / "good.wav", tmp_path / "bad.wav"
    for path, rate in ((good, 16000), (bad, 8000)):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1), handle.setsampwidth(2), handle.setframerate(rate)
            handle.writeframes(pcm((1000, 0.1)))
    assert len(live.read_wav(good)) == 3200
    with pytest.raises(ValueError):
        live.read_wav(bad)


def feed(detector: live.VoiceDetector, start: float, seconds: float, peak: int) -> None:
    for i in range(round(seconds / 0.01)):
        detector.feed(start + i * 0.01, peak, 0.01)


def test_detector_ignores_clicks_and_finds_onsets():
    detector = live.VoiceDetector()
    feed(detector, 0.0, 1.0, 0)
    feed(detector, 1.0, 0.02, 4000)  # a click shorter than MIN_VOICED_RUN_S
    feed(detector, 1.02, 0.5, 0)
    feed(detector, 1.52, 0.5, 4000)  # speech
    feed(detector, 2.02, 0.5, 0)
    feed(detector, 2.52, 0.3, 4000)  # second burst
    assert detector.onsets == pytest.approx([1.52, 2.52])
    assert detector.first_onset_since(0) == pytest.approx(1.52)
    assert detector.first_onset_since(2.0) == pytest.approx(2.52)
    assert detector.first_onset_since(3.0) is None
    assert detector.voiced_end == pytest.approx(2.82)


def test_utterance_result_from_start_and_after_end():
    detector = live.VoiceDetector()
    feed(detector, 12.4, 1.0, 5000)
    item = {"id": "ko-en-01", "source_language": "ko"}
    heard = live.utterance_result(0, item, 0, start=10.0, end=12.0, detector=detector, run_t0=5.0)
    assert heard["from_start_s"] == pytest.approx(2.4) and heard["after_end_s"] == pytest.approx(0.4)
    assert heard["speech_start_s"] == 5.0 and heard["heard_start_s"] == pytest.approx(7.4)
    # speech that started after the translation: nothing heard for it
    missing = live.utterance_result(1, item, 0, start=20.0, end=22.0, detector=detector, run_t0=5.0)
    assert missing["heard_start_s"] is None and missing["after_end_s"] is None


def result(from_start, after_end):
    heard = from_start is not None
    return {"heard_start_s": 1.0 if heard else None, "from_start_s": from_start, "after_end_s": after_end}


def test_summarize_cell_percentiles_missing_and_server_numbers():
    cell = live.Cell("cascaded", "balanced", "one_way", "ko-en")
    results = [result(2.0 + i * 0.1, 0.5 + i * 0.1) for i in range(10)] + [result(None, None)] * 2
    timeline = {"cascaded": {"count": 10, "p50": 2400.0, "p95": 2900.0, "marks": {}}}
    row = live.summarize_cell(cell, "cascaded", results, timeline)
    assert (row["played"], row["heard"], row["missing"]) == (12, 10, 2)
    assert row["from_start"]["count"] == 10 and row["from_start"]["p50"] == pytest.approx(2.45)
    assert row["after_end"]["p95"] == pytest.approx(1.355)
    assert row["runtime"] == {"count": 10, "p50": 2.4, "p95": 2.9}
    empty = live.summarize_cell(cell, "cascaded", [], {})
    assert empty["runtime"] is None and empty["from_start"]["p50"] is None


def test_pick_timeline_prefers_the_engine_then_the_busiest():
    timeline = {"cascaded": {"count": 2}, "realtime": {"count": 5}}
    assert live.pick_timeline(timeline, "cascaded")["count"] == 2
    assert live.pick_timeline(timeline, "agent_translation")["count"] == 5
    assert live.pick_timeline({}, "cascaded") is None


def sample_rows():
    items = [result(0.9 + i * 0.02, -0.5 + i * 0.02) for i in range(29)] + [result(None, None)]
    realtime = live.summarize_cell(live.Cell("realtime", "instant", "one_way", "ko-en"), "realtime", items,
                                   {"realtime": {"count": 29, "p50": 1000.0, "p95": 1300.0, "marks": {}}})
    items = [result(2.0 + i * 0.03, 0.4 + i * 0.03) for i in range(30)]
    cascaded = live.summarize_cell(live.Cell("cascaded", "balanced", "one_way", "ko-en"), "cascaded", items,
                                   {"cascaded": {"count": 30, "p50": 2900.0, "p95": 3500.0, "marks": {}}})
    skipped = live.summarize_cell(live.Cell("realtime", "instant", "auto", "ko-en"), "realtime", [], {},
                                  "skipped: realtime engine supports one_way only")
    return [realtime, cascaded, skipped]


def test_report_table_has_targets_and_empty_ns_columns():
    report = live.build_report(sample_rows(), {"run_id": "test"})
    summary = report.split("## Where the time goes")[0]
    rows = [line for line in summary.splitlines() if line.startswith("| ") and "Engine" not in line]
    assert len(rows) == 3
    realtime, cascaded, skipped = rows
    assert "| 30 | 29 | 1 |" in realtime and "1.00 / 1.30 (29)" in realtime
    assert "over (<= 1.2 s)" in realtime  # heard p95 from start is 1.46 s
    assert "ok (<= 1.5 s)" in cascaded  # heard p95 after end is about 1.2 s
    assert "not measured (NS-02)" in realtime and "not measured (NS-03)" in realtime
    assert "skipped: realtime engine supports one_way only" in skipped and "error:" not in skipped
    assert "**run_id**: test" in report


def test_plan_and_estimate_are_rough_but_sane():
    cells, skipped = live.plan_cells(["realtime", "cascaded"], ["instant", "balanced", "accurate"],
                                     ["one_way"], ["ko-en"])
    plan = live.build_plan(cells, skipped, 30, 4.4, 2.0, False)
    assert "6 cell(s) x 30 utterance(s)" in plan and "Estimated total" in plan
    realtime = live.estimate_cell(cells[0], 30, 4.4, 2.0)
    assert 4 < realtime["minutes"] < 8 and 0.1 < realtime["usd"] < 0.4


def test_main_dry_run_and_refusal_do_not_need_network(capsys, monkeypatch):
    def no_network(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("network used")

    monkeypatch.setattr(live.httpx.AsyncClient, "request", no_network)
    assert live.main(["--dry-run", "--utterances", "3"]) == 0
    assert "Live latency plan: 6 cell(s) x 3 utterance(s)" in capsys.readouterr().out
    assert live.main(["--utterances", "3"]) == 2
    assert "--confirm-billable" in capsys.readouterr().err
    assert live.main(["--engines", "gpt"]) == 2
    assert live.main(["--two-person", "--engines", "realtime,cascaded", "--dry-run"]) == 2


def test_platform_client_calls_and_polling():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.url.path == "/agents":
            return httpx.Response(201, json={"id": "a1", "interpreter": json.loads(request.content)["interpreter"]})
        if request.url.path == "/analytics/summary":
            count = sum(1 for _, path, _ in seen if path == "/analytics/summary")
            return httpx.Response(200, json={"timeline": {"cascaded": {"count": count, "p50": 1.0, "p95": 2.0}}})
        return httpx.Response(404, text="nope")

    import asyncio

    async def scenario():
        client = httpx.AsyncClient(base_url="http://platform", headers={"authorization": "Bearer k"},
                                   transport=httpx.MockTransport(handler))
        platform = live.Platform("http://platform", client=client)
        agent = await platform.create_agent("n", {"engine": "cascaded"})
        timeline = await platform.wait_for_timeline("a1", expected=3, timeout=30, interval=0)
        with pytest.raises(RuntimeError, match="404"):
            await platform.end_session("missing")
        await client.aclose()
        return agent, timeline

    agent, timeline = asyncio.run(scenario())
    assert agent["interpreter"] == {"engine": "cascaded"}
    assert timeline["cascaded"]["count"] == 3  # polled until three events had arrived
    assert seen[0][2] == "Bearer k"


def test_stage_breakdown_and_overlap_check() -> None:
    server = {"count": 2, "p50": 3000, "p95": 3500,
              "marks": {"vad": {"count": 2, "p50": 200, "p95": 250},
                        "translation_start": {"count": 2, "p50": 2600, "p95": 2900}}}
    stages = live.stage_summary(server)
    assert stages["vad"] == {"count": 2, "p50": 0.2, "p95": 0.25}
    assert "first_transcript" not in stages
    results = [
        {"speech_start_s": 1.0, "heard_start_s": 4.0},
        {"speech_start_s": 3.0, "heard_start_s": 3.5},   # heard before the next utterance? no
        {"speech_start_s": 9.0, "heard_start_s": 8.0},   # heard before it started
        {"speech_start_s": 12.0, "heard_start_s": None},
    ]
    # the first one's translation was heard after the second utterance began
    assert live.overlapping(results) == 2
    report = live.build_report([live.summarize_cell(
        live.Cell("cascaded", "balanced", "one_way", "ko-en"), "cascaded", [], {"cascaded": server})], {})
    assert "## Where the time goes" in report and "T3 translation start" in report

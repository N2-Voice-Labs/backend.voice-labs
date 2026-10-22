import json
import wave
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from src.evals import run as runner
from src.evals.dataset import load_manifest, select_items, DEFAULT_MANIFEST

TRANSCRIPTS = {
    # Deepgram reply keyed by the number of audio bytes is overkill: key by language/query
    "ru": "Здравствуйте я хотел бы записаться на приём к врачу на завтра",
    "ko": "안녕하세요 비자 연장 때문에 예약하고 싶은데요",
    "multi": "Мне нужно бук an appointment на завтра утром",
}


def write_wav(path: Path, seconds: float = 0.2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))


def sse(*deltas: str) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}\n\n" for d in deltas]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def make_transport(seen: list[httpx.Request], *, fail_openai: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "api.deepgram.com":
            language = parse_qs(urlparse(str(request.url)).query)["language"][0]
            body = {"results": {"channels": [{"alternatives": [{"transcript": TRANSCRIPTS[language]}]}]}}
            return httpx.Response(200, json=body)
        if request.url.host == "api.openai.com":
            if fail_openai:
                return httpx.Response(500, json={"error": "boom"})
            payload = json.loads(request.content)
            target = payload["messages"][0]["content"]
            reply = "Hello, I would like to make an appointment with a doctor for tomorrow." if "Russian" in target else "I need to book an appointment for tomorrow morning."
            return httpx.Response(200, content=sse(reply[:10], reply[10:]), headers={"content-type": "text/event-stream"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def items_and_audio(tmp_path: Path, ids: list[str]) -> tuple[list[dict], Path]:
    items = [i for i in load_manifest(DEFAULT_MANIFEST) if i["id"] in ids]
    for item in items:
        write_wav(tmp_path / f"{item['id']}.wav")
    return items, tmp_path


def test_translation_prompt_is_strict_and_labels_code_switching() -> None:
    messages = runner.build_translation_messages("hola", "ru+en", "en")
    system = messages[0]["content"]
    assert "Output ONLY the translation" in system
    assert "Never answer questions" in system
    assert "Russian (ru) mixed with English (en) words" in system
    assert messages[1] == {"role": "user", "content": "hola"}


def test_deepgram_client_sends_audio_and_parses_transcript() -> None:
    seen: list[httpx.Request] = []
    client = runner.DeepgramStt("dg-secret", client=httpx.Client(transport=make_transport(seen)))
    transcript, latency = client.transcribe(b"RIFF....", "ru")
    assert transcript == TRANSCRIPTS["ru"]
    assert latency >= 0
    request = seen[0]
    query = parse_qs(urlparse(str(request.url)).query)
    assert query["model"] == ["nova-3"] and query["language"] == ["ru"]
    assert query["smart_format"] == ["false"]
    assert request.headers["authorization"] == "Token dg-secret"
    assert request.headers["content-type"] == "audio/wav"
    assert request.content == b"RIFF...."


def test_openai_client_streams_and_measures_ttft() -> None:
    seen: list[httpx.Request] = []
    client = runner.OpenAITranslate("sk-secret", "gpt-4.1-nano", client=httpx.Client(transport=make_transport(seen)))
    text, ttft, total = client.translate("Здравствуйте", "ru", "en")
    assert text.startswith("Hello, I would")
    assert 0 <= ttft <= total
    payload = json.loads(seen[0].content)
    assert payload["model"] == "gpt-4.1-nano" and payload["stream"] is True and payload["temperature"] == 0
    assert seen[0].headers["authorization"] == "Bearer sk-secret"


def test_http_errors_do_not_leak_keys() -> None:
    seen: list[httpx.Request] = []
    client = runner.OpenAITranslate("sk-secret", "m", client=httpx.Client(transport=make_transport(seen, fail_openai=True)))
    with pytest.raises(RuntimeError) as error:
        client.translate("x", "ru", "en")
    assert "500" in str(error.value) and "sk-secret" not in str(error.value)


def test_stt_language_policy() -> None:
    ko = {"source_language": "ko"}
    ru = {"source_language": "ru"}
    mixed = {"source_language": "ru+en"}
    assert runner.stt_language_for(ru, "fixed") == "ru"
    assert runner.stt_language_for(ru, "multi") == "multi"
    assert runner.stt_language_for(ko, "multi") == "ko"  # nova-3 multi does not cover Korean
    assert runner.stt_language_for(mixed, "fixed") == "multi"


def test_cascade_records_per_stage_latency_and_scores(tmp_path) -> None:
    seen: list[httpx.Request] = []
    http = httpx.Client(transport=make_transport(seen))
    items, audio_dir = items_and_audio(tmp_path, ["ru-en-01", "mixed-01"])
    records = runner.run_benchmark(
        items,
        mode="cascade",
        audio_dir=audio_dir,
        stt=runner.DeepgramStt("k", client=http),
        translators=[runner.OpenAITranslate("k", "gpt-4.1-nano", client=http)],
        run_id="t",
        log=lambda _: None,
    )
    assert [r["item_id"] for r in records] == ["ru-en-01", "mixed-01"]
    ru, mixed = records
    assert ru["error"] is None and ru["stt_language"] == "ru"
    assert ru["wer"] == 0.0  # transcript equals the reference apart from punctuation
    assert ru["chrf"] > 90
    assert ru["end_to_end_s"] == pytest.approx(ru["stt_latency_s"] + ru["translate_total_s"])
    assert ru["first_translation_s"] <= ru["end_to_end_s"]
    assert mixed["stt_language"] == "multi"
    assert mixed["code_switch"]["by_script"]["latin"]["errors"] == 1  # only "book" -> "бук"
    assert mixed["code_switch"]["by_script"]["cyrillic"]["errors"] == 0


def test_translate_mode_uses_gold_text_and_both_interruption_turns(tmp_path) -> None:
    seen: list[httpx.Request] = []
    http = httpx.Client(transport=make_transport(seen))
    items = [i for i in load_manifest(DEFAULT_MANIFEST) if i["id"] in {"ru-en-01", "interruptions-05"}]
    records = runner.run_benchmark(
        items,
        mode="translate",
        audio_dir=tmp_path,
        stt=None,
        translators=[runner.OpenAITranslate("k", "gpt-4.1-mini", client=http)],
        run_id="t",
        log=lambda _: None,
    )
    assert all(r["error"] is None for r in records)
    openai_calls = [json.loads(r.content) for r in seen if r.url.host == "api.openai.com"]
    assert len(openai_calls) == 3  # one utterance + two interruption turns
    assert openai_calls[0]["messages"][1]["content"] == items[0]["text"]
    assert records[1]["input_text"] == items[1]["text"] + " " + items[1]["followup"]["text"]
    assert not [r for r in seen if r.url.host == "api.deepgram.com"]


def test_errors_are_recorded_without_stopping_the_run(tmp_path) -> None:
    http = httpx.Client(transport=make_transport([], fail_openai=True))
    items, audio_dir = items_and_audio(tmp_path, ["ru-en-01"])
    ghost = {**items[0], "id": "ru-en-99"}  # no audio file
    records = runner.run_benchmark(
        [ghost, items[0]],
        mode="cascade",
        audio_dir=audio_dir,
        stt=runner.DeepgramStt("k", client=http),
        translators=[runner.OpenAITranslate("k", "m", client=http)],
        run_id="t",
        log=lambda _: None,
    )
    assert records[0]["error"].startswith("stt: FileNotFoundError")
    assert records[1]["error"].startswith("translate: RuntimeError")
    assert "chrf" not in records[1]


def test_summary_table_has_category_rows_and_metadata(tmp_path) -> None:
    http = httpx.Client(transport=make_transport([]))
    items, audio_dir = items_and_audio(tmp_path, ["ru-en-01", "ko-en-01", "mixed-01"])
    records = runner.run_benchmark(
        items, mode="cascade", audio_dir=audio_dir, stt=runner.DeepgramStt("k", client=http),
        translators=[runner.OpenAITranslate("k", "gpt-4.1-nano", client=http)], run_id="t", log=lambda _: None,
    )
    summary = runner.build_summary(records, {"run_id": "t", "mode": "cascade", "dataset_version": "v1"})
    assert "## Cascade: gpt-4.1-nano" in summary
    for category in ("ru-en", "ko-en", "mixed", "**all**"):
        assert f"| {category} |" in summary
    assert "chrF" in summary and "end-to-end s p50/p95" in summary and "CER (ko)" in summary
    assert "## Code-switch WER by script" in summary
    assert "- dataset_version: v1" in summary


def test_main_requires_confirm_billable(tmp_path, capsys) -> None:
    code = runner.main(["translate", "--limit", "1", "--out", str(tmp_path / "out")])
    assert code == 2
    assert "--confirm-billable" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_end_to_end_with_mocked_http(tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "audio").mkdir()
    _, audio_dir = items_and_audio(tmp_path / "audio", ["ru-en-01"])
    http = httpx.Client(transport=make_transport([]))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    code = runner.main(
        ["cascade", "--confirm-billable", "--categories", "ru-en", "--limit", "1", "--models", "gpt-4.1-nano",
         "--audio-dir", str(audio_dir), "--out", str(tmp_path / "out")],
        stt_client=runner.DeepgramStt("dg-test", client=http),
        translator_factory=lambda model: runner.OpenAITranslate("sk-test", model, client=http),
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "sk-test" not in out and "dg-test" not in out
    (run_dir,) = list((tmp_path / "out").iterdir())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["item_id"] == "ru-en-01" and rows[0]["model"] == "gpt-4.1-nano"
    assert "dataset_version: v1" in (run_dir / "summary.md").read_text(encoding="utf-8")

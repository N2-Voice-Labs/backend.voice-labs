"""Offline tests for FAQ normalization, scoring, and dataset loading."""

from pathlib import Path

from src.evals.faq_eval import has_no_answer_phrase, normalize_text, score_item
from scripts.run_faq_eval import load_dataset


def fixture_item(**overrides):
    item = {
        "id": "faq-1",
        "question": "How much?",
        "expected_answer": "The fee is KRW 5,462,000.",
        "must_include": ["5,462,000", "2026–2027"],
        "source_file": "fees.pdf",
        "answerable": True,
    }
    return item | overrides


def test_normalize_numbers_currency_and_dashes():
    assert normalize_text("₩5,462,000 — 2026–2027") == "krw 5462000 - 2026-2027"
    assert normalize_text("KRW 5462000 won") == "krw 5462000 krw"


def test_answerable_requires_every_normalized_keyword():
    item = fixture_item()
    passed = score_item(item, "The fee is KRW 5462000; the year is 2026-2027.")
    assert passed["passed"]
    assert passed["missing_must_include"] == []
    failed = score_item(item, "The fee is KRW 5,462,000.")
    assert not failed["passed"]
    assert failed["needs_review"]
    assert failed["missing_must_include"] == ["2026–2027"]


def test_unanswerable_phrase_or_handoff_and_numeric_flags_only():
    item = fixture_item(answerable=False, must_include=[])
    unknown = score_item(item, "I can't find that policy; the fee may be 5000 won.")
    assert unknown["passed"]
    assert unknown["suspicious_numbers"]
    assert score_item(item, "Please contact our office.", True)["passed"]
    assert not score_item(item, "The fee is 5000 won.")["passed"]
    assert has_no_answer_phrase("I don't know that")


def test_load_dataset_supports_list_inside_top_level_key(tmp_path: Path):
    dataset = tmp_path / "faq.yaml"
    dataset.write_text(
        "questions:\n  - id: x\n    question: Q?\n    expected_answer: A\n"
        "    must_include: [answer]\n    source_file: source.pdf\n",
        encoding="utf-8",
    )
    loaded = load_dataset(dataset)
    assert len(loaded) == 1
    assert loaded[0]["answerable"] is True


def test_guideline_and_spoken_dates_match():
    from src.evals.faq_eval import score_item
    item = {"id": "d", "must_include": ["2026. 11. 05", "2027. 01. 12", "17:00"]}
    reply = "Applications close on November 5, 2026, at 17:00; payment ends January 12th 2027."
    assert score_item(item, reply)["passed"]
    assert not score_item(item, "Applications close on November 6, 2026, at 17:00.")["passed"]

import re
from collections import Counter

from src.evals.dataset import (
    CATEGORIES,
    DEFAULT_MANIFEST,
    expected_text,
    expected_translation,
    load_manifest,
    parse_categories,
    select_items,
    validate_item,
    validate_manifest,
)

EXPECTED_COUNTS = {
    "en-ko": 20,
    "ko-en": 20,
    "ru-en": 20,
    "en-ru": 20,
    "mixed": 20,
    "names-numbers": 10,
    "domain-terms": 10,
    "noisy-phone": 10,
    "interruptions": 10,
}


def test_v1_manifest_is_valid_and_complete() -> None:
    items = load_manifest(DEFAULT_MANIFEST)
    assert validate_manifest(items) == []
    assert 130 <= len(items) <= 170
    assert Counter(item["category"] for item in items) == EXPECTED_COUNTS
    assert set(EXPECTED_COUNTS) == set(CATEGORIES)
    assert len({item["id"] for item in items}) == len(items)


def test_v1_manifest_content_rules() -> None:
    items = load_manifest(DEFAULT_MANIFEST)
    by_id = {item["id"]: item for item in items}
    for item in items:
        assert item["id"].startswith(item["category"] + "-")
        # spoken text is digit-free so WER does not test number formatting
        assert not re.search(r"\d", item["text"]), item["id"]
        if item.get("followup"):
            assert not re.search(r"\d", item["followup"]["text"]), item["id"]
        assert item["key_terms"], item["id"]
        if item["source_language"] == "ru+en":
            assert re.search(r"[A-Za-z]", item["text"]) and re.search(r"[А-Яа-яЁё]", item["text"])
        if item["category"] == "noisy-phone":
            parent = by_id[item["derived_from"]]
            assert parent["category"] != "noisy-phone"
            assert item["text"] == parent["text"]
            assert item["reference_translation"] == parent["reference_translation"]
        if item["category"] == "interruptions":
            assert item["followup"]["text"] and item["followup"]["reference_translation"]


def test_every_direction_has_script_appropriate_text() -> None:
    for item in load_manifest(DEFAULT_MANIFEST):
        if item["source_language"] == "ko":
            assert re.search(r"[가-힣]", item["text"])
        if item["source_language"] == "ru":
            assert re.search(r"[А-Яа-я]", item["text"])
        if item["target_language"] == "ko":
            assert re.search(r"[가-힣]", item["reference_translation"])
        if item["target_language"] == "ru":
            assert re.search(r"[А-Яа-я]", item["reference_translation"])
        if item["target_language"] == "en":
            assert not re.search(r"[가-힣А-Яа-я]", item["reference_translation"])


def test_validate_item_reports_problems() -> None:
    good = load_manifest(DEFAULT_MANIFEST)[0]
    assert validate_item(good) == []
    assert any("missing field 'text'" in p for p in validate_item({k: v for k, v in good.items() if k != "text"}))
    assert validate_item({**good, "source_language": "de"})
    assert validate_item({**good, "target_language": good["source_language"]})
    assert validate_item({**good, "category": "other"})
    assert validate_item({**good, "key_terms": "visa"})
    assert validate_item({**good, "category": "interruptions"})  # needs followup


def test_validate_manifest_catches_duplicates_and_dangling_parents() -> None:
    items = load_manifest(DEFAULT_MANIFEST)
    dup = items + [items[0]]
    assert any("duplicate id" in p for p in validate_manifest(dup))
    noisy = next(i for i in items if i["category"] == "noisy-phone")
    broken = [i for i in items if i["id"] != noisy["derived_from"]]
    assert any("derived_from" in p for p in validate_manifest(broken))


def test_select_items_limits_per_category_and_filters() -> None:
    items = load_manifest(DEFAULT_MANIFEST)
    picked = select_items(items, ["en-ko", "mixed"], 3)
    assert Counter(i["category"] for i in picked) == {"en-ko": 3, "mixed": 3}
    assert len(select_items(items)) == len(items)
    assert parse_categories("en-ko, ko-en") == ["en-ko", "ko-en"]
    assert parse_categories(None) is None


def test_interruption_items_join_both_turns() -> None:
    item = next(i for i in load_manifest(DEFAULT_MANIFEST) if i["category"] == "interruptions")
    assert expected_text(item) == f"{item['text']} {item['followup']['text']}"
    assert item["followup"]["reference_translation"] in expected_translation(item)

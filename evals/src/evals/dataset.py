"""Evaluation dataset manifest: schema, loading, validation and selection.

The manifest is committed text only (``datasets/<version>/manifest.jsonl``); audio is
rendered locally by :mod:`evals.generate_audio` or supplied as consented recordings that
reuse the same ``id`` and fields. One JSON object per line.

Fields
------
``id``                    unique, stable; the audio file is ``<id>.wav``
``category``              one of :data:`CATEGORIES`
``source_language``       ``en`` | ``ko`` | ``ru`` | ``ru+en`` (code-switched)
``target_language``       ``en`` | ``ko`` | ``ru``
``text``                  exactly what is spoken (the STT reference); no digits, numbers
                          are written as spoken words so WER is not a formatting test
``reference_translation`` careful, natural translation into ``target_language``
``key_terms``             terms for terminology recall; scored on whichever side
                          (``text`` or ``reference_translation``) contains them;
                          ``a|b`` lists acceptable alternatives
``tts_voice``             voice hint for synthetic rendering
Optional: ``derived_from`` + ``degradation`` (noisy/phone variants), ``followup``
(``text`` + ``reference_translation`` of the second utterance in interruption items).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets"
DEFAULT_MANIFEST = DATASET_DIR / "v1" / "manifest.jsonl"

CATEGORIES = (
    "en-ko",
    "ko-en",
    "ru-en",
    "en-ru",
    "mixed",
    "names-numbers",
    "domain-terms",
    "noisy-phone",
    "interruptions",
)
TARGET_LANGUAGES = frozenset({"en", "ko", "ru"})
SOURCE_LANGUAGES = TARGET_LANGUAGES | {"ru+en"}
REQUIRED_FIELDS = (
    "id",
    "category",
    "source_language",
    "target_language",
    "text",
    "reference_translation",
    "key_terms",
    "tts_voice",
)
DEGRADATION_TYPES = frozenset({"noise", "phone"})


def validate_item(item: dict[str, Any]) -> list[str]:
    """Return a list of problems with one manifest item (empty when valid)."""

    problems: list[str] = []
    for name in REQUIRED_FIELDS:
        if name not in item:
            problems.append(f"missing field {name!r}")
    if problems:
        return problems
    if item["category"] not in CATEGORIES:
        problems.append(f"unknown category {item['category']!r}")
    if item["source_language"] not in SOURCE_LANGUAGES:
        problems.append(f"invalid source_language {item['source_language']!r}")
    if item["target_language"] not in TARGET_LANGUAGES:
        problems.append(f"invalid target_language {item['target_language']!r}")
    if item["source_language"] == item["target_language"]:
        problems.append("source and target language are the same")
    for name in ("id", "text", "reference_translation", "tts_voice"):
        if not isinstance(item[name], str) or not item[name].strip():
            problems.append(f"{name!r} must be a non-empty string")
    terms = item["key_terms"]
    if not isinstance(terms, list) or not all(isinstance(t, str) and t.strip() for t in terms):
        problems.append("'key_terms' must be a list of non-empty strings")
    if item["category"] == "interruptions":
        follow = item.get("followup")
        if not (
            isinstance(follow, dict)
            and follow.get("text")
            and follow.get("reference_translation")
        ):
            problems.append("interruption items need followup.text and followup.reference_translation")
    if item["category"] == "noisy-phone":
        if not item.get("derived_from"):
            problems.append("noisy-phone items need 'derived_from'")
        degradation = item.get("degradation")
        if not isinstance(degradation, dict) or degradation.get("type") not in DEGRADATION_TYPES:
            problems.append("noisy-phone items need degradation.type in noise|phone")
    return problems


def validate_manifest(items: list[dict[str, Any]]) -> list[str]:
    """Validate every item plus cross-item rules (unique ids, valid derived_from)."""

    problems: list[str] = []
    ids: set[str] = set()
    for index, item in enumerate(items, 1):
        label = item.get("id", f"line {index}")
        problems.extend(f"{label}: {p}" for p in validate_item(item))
        if label in ids:
            problems.append(f"{label}: duplicate id")
        ids.add(label)
    for item in items:
        parent = item.get("derived_from")
        if parent and parent not in ids:
            problems.append(f"{item.get('id')}: derived_from {parent!r} not found")
    return problems


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return items


def parse_categories(value: str | None) -> list[str] | None:
    if not value:
        return None
    categories = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"unknown categories {unknown}; choose from {', '.join(CATEGORIES)}")
    return categories


def select_items(
    items: list[dict[str, Any]],
    categories: list[str] | None = None,
    limit_per_category: int | None = None,
) -> list[dict[str, Any]]:
    """Filter by category and keep at most ``limit_per_category`` items of each."""

    taken: dict[str, int] = defaultdict(int)
    selected = []
    for item in items:
        if categories and item["category"] not in categories:
            continue
        if limit_per_category is not None and taken[item["category"]] >= limit_per_category:
            continue
        taken[item["category"]] += 1
        selected.append(item)
    return selected


def turns(item: dict[str, Any]) -> list[dict[str, str]]:
    """Utterances in the item: one, or two for interruption items."""

    result = [{"text": item["text"], "reference_translation": item["reference_translation"]}]
    follow = item.get("followup")
    if follow:
        result.append(
            {
                "text": follow["text"],
                "reference_translation": follow["reference_translation"],
            }
        )
    return result


def expected_text(item: dict[str, Any]) -> str:
    """Everything spoken in the item's audio, in order."""

    return " ".join(turn["text"] for turn in turns(item))


def expected_translation(item: dict[str, Any]) -> str:
    return " ".join(turn["reference_translation"] for turn in turns(item))

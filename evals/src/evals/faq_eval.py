"""Scoring helpers for the demo assistant's text FAQ evaluation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_CURRENCY = re.compile(r"₩|\bKRW\b|\bwon(?:s)?\b|원", re.IGNORECASE)
_NUMBER_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_WHITESPACE = re.compile(r"\s+")
_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), start=1)}
_DOTTED_DATE = re.compile(r"\b(\d{4})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{1,2})\b\.?")
_WORDED_DATE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_NO_ANSWER = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i can't find", "i cannot find", "couldn't find", "could not find",
    "not available", "information is unavailable", "i don't have that information",
    "i do not have that information", "unable to find", "no information",
)
_SUSPICIOUS_NUMBER = re.compile(
    r"(?<!\w)(?:₩\s*)?\d[\d,]*(?:\.\d+)?\s*(?:원|won|krw)?(?!\w)", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """Casefold and normalize common numeric, currency, dash, and spacing variants."""
    value = text.casefold().replace("–", "-").replace("—", "-").replace("−", "-")
    # "2026. 11. 05" (guideline style) and "November 5, 2026" (spoken style) both become 2026-11-05.
    value = _DOTTED_DATE.sub(lambda m: f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}", value)
    value = _WORDED_DATE.sub(lambda m: f"{m[3]}-{_MONTHS[m[1]]:02d}-{int(m[2]):02d}", value)
    value = re.sub(r"(\d)\s*(?:percent|per cent)\b", r"\1%", value)
    value = _CURRENCY.sub(" krw ", value)
    value = _NUMBER_COMMA.sub("", value)
    return _WHITESPACE.sub(" ", value).strip()


def has_no_answer_phrase(reply: str) -> bool:
    folded = normalize_text(reply)
    return any(phrase in folded for phrase in _NO_ANSWER)


def suspicious_numbers(reply: str) -> list[str]:
    """Return numeric claims for human review; these are flags, not score failures."""
    return _SUSPICIOUS_NUMBER.findall(reply)


def score_item(item: Mapping[str, Any], reply: str, handoff_suggested: bool = False) -> dict[str, Any]:
    answerable = item.get("answerable", True) is not False
    must_include = item.get("must_include", []) or []
    normalized_reply = normalize_text(reply)
    missing = [term for term in must_include if normalize_text(str(term)) not in normalized_reply]
    if answerable:
        passed = not missing
    else:
        passed = (has_no_answer_phrase(reply) or handoff_suggested)
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "expected_answer": item.get("expected_answer"),
        "must_include": must_include,
        "source_file": item.get("source_file"),
        "answerable": answerable,
        "reply": reply,
        "handoff_suggested": bool(handoff_suggested),
        "passed": passed,
        "missing_must_include": missing,
        "needs_review": bool(answerable and missing and reply.strip()),
        "suspicious_numbers": suspicious_numbers(reply) if not answerable else [],
    }

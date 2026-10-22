"""Shared request-validation helpers (importable without importing ``main``)."""

import re
from typing import Any

SECRET_KEY_PARTS = ("secret", "password", "token", "api_key", "apikey")
SECRET_CONFIGURATION_MESSAGE = (
    "provider secrets must use secret_reference, not configuration"
)
OUTCOMES = frozenset(
    {
        "resolved",
        "booked",
        "callback_requested",
        "no_answer",
        "declined",
        "escalated",
        "failed",
        "other",
    }
)
INTERPRETER_LANGUAGES = frozenset(
    {"en", "ru", "ko", "ja", "zh", "es", "fr", "de", "it", "pt", "nl", "hi", "tr", "pl"}
)
# Target (output) languages of OpenAI ``gpt-realtime-translate``; the model accepts 70+
# input languages. Verified 2026-10-18 against
# https://developers.openai.com/cookbook/examples/voice_solutions/realtime_translation_guide
# ("13 target output languages": Spanish, Portuguese, French, Japanese, Russian, Chinese,
# German, Korean, Hindi, Indonesian, Vietnamese, Italian, English).
REALTIME_TRANSLATE_OUTPUT_LANGUAGES = frozenset(
    {"es", "pt", "fr", "ja", "ru", "zh", "de", "ko", "hi", "id", "vi", "it", "en"}
)
# Interpreter latency modes (RM-011); mirrored by Runtime's ``latency_modes``.
LATENCY_MODES = ("instant", "balanced", "accurate")
DEFAULT_LATENCY_MODE = "balanced"
AUTO_DIRECTION_LANGUAGES = frozenset({"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"})
E164_PATTERN = r"^\+[1-9]\d{6,14}$"
_E164_RE = re.compile(E164_PATTERN)


def contains_secret_key(value: Any) -> bool:
    """Return True if any (nested) dict key looks like a secret name."""
    if isinstance(value, dict):
        return any(
            any(part in str(key).casefold() for part in SECRET_KEY_PARTS)
            or contains_secret_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(contains_secret_key(item) for item in value)
    return False


def reject_secret_configuration(value: dict[str, Any]) -> dict[str, Any]:
    if contains_secret_key(value):
        raise ValueError(SECRET_CONFIGURATION_MESSAGE)
    return value


def validate_e164(value: str) -> str:
    if not isinstance(value, str) or not _E164_RE.match(value):
        raise ValueError("phone number must be in E.164 format, e.g. +14155550123")
    return value


TRANSFER_TO_MAX_LENGTH = 255
_TRANSFER_URI_SCHEMES = ("tel:", "sip:", "sips:")


def validate_transfer_to(value: Any) -> str:
    """A human-transfer destination: ``+E.164`` or a ``tel:``/``sip:``/``sips:`` URI."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > TRANSFER_TO_MAX_LENGTH
    ):
        raise ValueError("invalid_transfer_to")
    lowered = value.lower()
    for scheme in _TRANSFER_URI_SCHEMES:
        if lowered.startswith(scheme):
            rest = value[len(scheme) :]
            if not rest or any(ch.isspace() for ch in rest):
                raise ValueError("invalid_transfer_to")
            return value
    try:
        return validate_e164(value)
    except ValueError:
        raise ValueError("invalid_transfer_to") from None


def validate_deployment_configuration(value: dict[str, Any]) -> dict[str, Any]:
    """Reject secrets and validate the optional ``transfer_to`` destination."""

    reject_secret_configuration(value)
    if "transfer_to" in value and value["transfer_to"] is not None:
        validate_transfer_to(value["transfer_to"])
    return value


# NS-08: per-agent glossary. ``[{"source": "residence permit", "translations": {"ru": "..."},
# "note": "..."}, {"source": "Voice Labs", "keep": true}]``. Mirrored by Runtime's
# ``glossary`` module, which reads the normalized form.
GLOSSARY_MAX_ENTRIES = 200
GLOSSARY_MAX_TERM_LENGTH = 100
GLOSSARY_MAX_NOTE_LENGTH = 200
_GLOSSARY_ENTRY_KEYS = frozenset({"source", "translations", "keep", "note"})


def _glossary_text(value: Any, label: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"glossary {label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"glossary {label} must not be empty")
    if len(text) > max_length:
        raise ValueError(f"glossary {label} must be at most {max_length} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"glossary {label} must not contain control characters")
    return text


def validate_glossary(value: Any) -> list[dict[str, Any]]:
    """Validate and normalize a glossary (NS-08); raises ``ValueError``.

    At most :data:`GLOSSARY_MAX_ENTRIES` entries. Each entry has a ``source`` term
    (at most :data:`GLOSSARY_MAX_TERM_LENGTH` characters, unique case-insensitively)
    and either ``translations`` (language code -> required rendering) or ``keep: true``
    (never translate: names, brands). ``note`` is an optional hint.
    """

    if not isinstance(value, list):
        raise ValueError("glossary must be a list")
    if len(value) > GLOSSARY_MAX_ENTRIES:
        raise ValueError(f"glossary must have at most {GLOSSARY_MAX_ENTRIES} entries")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("glossary entries must be objects")
        unknown = set(entry) - _GLOSSARY_ENTRY_KEYS
        if unknown:
            raise ValueError(f"unknown glossary entry field: {sorted(unknown)[0]}")
        source = _glossary_text(entry.get("source"), "source", GLOSSARY_MAX_TERM_LENGTH)
        if source.casefold() in seen:
            raise ValueError(f"duplicate glossary term: {source}")
        seen.add(source.casefold())
        keep = entry.get("keep", False)
        if not isinstance(keep, bool):
            raise ValueError("glossary keep must be a boolean")
        translations = entry.get("translations")
        result: dict[str, Any] = {"source": source}
        if keep:
            if translations:
                raise ValueError("a glossary entry cannot have both keep and translations")
            result["keep"] = True
        else:
            if not isinstance(translations, dict) or not translations:
                raise ValueError(
                    "a glossary entry needs translations or keep: true"
                )
            rendered: dict[str, str] = {}
            for code, term in translations.items():
                language = code.lower() if isinstance(code, str) else code
                if language not in INTERPRETER_LANGUAGES:
                    raise ValueError(f"unknown glossary language: {code}")
                if language in rendered:
                    raise ValueError(f"duplicate glossary language: {code}")
                rendered[language] = _glossary_text(
                    term, "translation", GLOSSARY_MAX_TERM_LENGTH
                )
            result["translations"] = rendered
        if entry.get("note") is not None:
            result["note"] = _glossary_text(
                entry["note"], "note", GLOSSARY_MAX_NOTE_LENGTH
            )
        normalized.append(result)
    return normalized

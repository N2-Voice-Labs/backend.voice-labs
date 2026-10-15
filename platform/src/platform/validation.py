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

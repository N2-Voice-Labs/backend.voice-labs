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

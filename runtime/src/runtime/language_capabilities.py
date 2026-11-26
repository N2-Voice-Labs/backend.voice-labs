"""Language capability table (ST-20): the single source of truth for language support.

One row per language code:

* ``stt_fixed``: Deepgram nova-3 transcribes the language when it is fixed up front.
* ``stt_multi``: the language is in Deepgram nova-3's ``multi`` (code-switching) set, so
  the caller may switch between such languages and still be transcribed. Verified against
  https://developers.deepgram.com/docs/models-languages-overview (nova-3 "Multilingual":
  English, Spanish, French, German, Hindi, Russian, Portuguese, Japanese, Italian, Dutch).
  With the ``elevenlabs`` STT provider (realtime Scribe, language detection on) every
  ``stt_fixed`` language can be detected instead, so mixed-language speech is
  transcribed for pairs Deepgram's set does not cover (e.g. Korean).
* ``tts``: Cartesia can speak the language.
* ``realtime_output``: OpenAI ``gpt-realtime-translate`` can speak the language as a
  target (13 output languages; verified 2026-10-18 against the OpenAI realtime
  translation guide).

Platform vendors this table as ``platform/src/platform/language_capabilities.json`` (a
drift test compares both). Regenerate the JSON after editing this file, from the
workspace root::

    python runtime/src/runtime/language_capabilities.py > platform/src/platform/language_capabilities.json

``id`` and ``vi`` are realtime output languages only; Voice Labs has not enabled them for
the cascaded STT/TTS stack, so they are not interpreter languages.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

DIRECTIONS = ("auto", "one_way")

# code -> (name, stt_fixed, stt_multi, tts, realtime_output)
_ROWS: dict[str, tuple[str, bool, bool, bool, bool]] = {
    "en": ("English", True, True, True, True),
    "ru": ("Russian", True, True, True, True),
    "ko": ("Korean", True, False, True, True),
    "ja": ("Japanese", True, True, True, True),
    "zh": ("Chinese", True, False, True, True),
    "es": ("Spanish", True, True, True, True),
    "fr": ("French", True, True, True, True),
    "de": ("German", True, True, True, True),
    "it": ("Italian", True, True, True, True),
    "pt": ("Portuguese", True, True, True, True),
    "nl": ("Dutch", True, True, True, False),
    "hi": ("Hindi", True, True, True, True),
    "tr": ("Turkish", True, False, True, False),
    "pl": ("Polish", True, False, True, False),
    "id": ("Indonesian", False, False, False, True),
    "vi": ("Vietnamese", False, False, False, True),
}

CAPABILITIES: dict[str, dict[str, object]] = {
    code: {
        "name": name,
        "stt_fixed": stt_fixed,
        "stt_multi": stt_multi,
        "tts": tts,
        "realtime_output": realtime_output,
    }
    for code, (name, stt_fixed, stt_multi, tts, realtime_output) in _ROWS.items()
}


def _codes(flag: str) -> frozenset[str]:
    return frozenset(code for code, row in CAPABILITIES.items() if row[flag])


# STT providers whose automatic language detection covers every ``stt_fixed`` language,
# not only Deepgram's ``multi`` set. ``multi`` is passed to them as "detect the language"
# (see ``speech_providers._elevenlabs_stt``).
_BROAD_DETECTION_STT_PROVIDERS = frozenset({"elevenlabs"})


def auto_detect_languages(provider: str = "deepgram") -> frozenset[str]:
    """Languages ``provider`` can transcribe with automatic (mixed-language) detection.

    Deepgram: its ``multi`` (code-switching) set. ElevenLabs: every ``stt_fixed`` language.
    """

    if provider in _BROAD_DETECTION_STT_PROVIDERS:
        return _codes("stt_fixed")
    return _codes("stt_multi")


def interpreter_languages() -> frozenset[str]:
    """Languages the cascaded stack can both transcribe (fixed) and speak."""

    return frozenset(
        code for code, row in CAPABILITIES.items() if row["stt_fixed"] and row["tts"]
    )


def realtime_output_languages() -> frozenset[str]:
    """Target languages of the realtime translation engine."""

    return _codes("realtime_output")


def can_auto_detect(langs: Sequence[str], provider: str = "deepgram") -> bool:
    """Whether ``provider`` can auto-detect every language in ``langs``."""

    detectable = auto_detect_languages(provider)
    return bool(langs) and all(code in detectable for code in langs)


def stt_language_for(
    langs: Sequence[str], direction: str | None, provider: str = "deepgram"
) -> str:
    """The STT language to request: ``"multi"`` (detect) or the fixed source language.

    ``multi`` is chosen only for ``direction == "auto"`` and when ``provider`` can detect
    every language in ``langs``; otherwise the first language (the source) is used.
    """

    if not langs:
        raise ValueError("at least one language is required")
    if direction == "auto" and can_auto_detect(langs, provider):
        return "multi"
    return langs[0]


def as_json() -> str:
    """The table as stable, indented JSON (the Platform's vendored copy)."""

    return json.dumps(CAPABILITIES, indent=2, sort_keys=False) + "\n"


if __name__ == "__main__":  # pragma: no cover - regeneration helper
    print(as_json(), end="")

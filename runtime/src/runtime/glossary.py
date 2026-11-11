"""Per-agent glossary (NS-08): terms the translation must respect.

The Platform validates and normalizes the glossary (``validate_glossary``) and passes it in
the ``interpreter`` or ``translation`` block of the session contract::

    [{"source": "residence permit", "translations": {"ru": "вид на жительство"}, "note": "..."},
     {"source": "Voice Labs", "keep": true}]

An entry is a set of equivalent renderings of one term. ``translations`` maps a language
to its rendering; ``source`` is the rendering in every language that has no entry (usually
the working language, e.g. English). So for ``residence permit`` above, English to Russian
uses ``вид на жительство`` and Russian to English uses ``residence permit``. A ``keep`` entry
is never translated (names, brands).

Only the entries relevant to an utterance go into the prompt (:func:`select_entries`), which
keeps the translation call short. :func:`glossary_keyterms` lists the terms for STT keyterm
boosting.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# At most this many entries are put in one translation prompt.
MAX_PROMPT_ENTRIES = 12
# A glossary of at most this many applicable entries is always sent whole; a larger one
# is filtered down to the entries whose term appears in the utterance.
SEND_ALL_THRESHOLD = 6
# Deepgram accepts limited keyterms per request (~500 tokens); stay well below.
MAX_KEYTERMS = 100
_STEM_MIN_WORD = 5
_STEM_DROP = 2


@dataclass(frozen=True)
class GlossaryEntry:
    source: str
    translations: tuple[tuple[str, str], ...] = ()
    keep: bool = False
    note: str | None = None

    def rendering(self, language: str) -> str | None:
        """The term as written in ``language`` (``source`` when it has no entry)."""

        if self.keep:
            return self.source
        for code, term in self.translations:
            if code == language:
                return term
        return self.source

    def has_translation(self, language: str) -> bool:
        return any(code == language for code, _ in self.translations)


Glossary = tuple[GlossaryEntry, ...]


def parse_glossary(raw: Any) -> Glossary:
    """Parse the contract's glossary list; raises ``RuntimeError`` when malformed.

    The Platform has already validated it, so this only guards the Runtime's own types.
    """

    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError("glossary must be a list")
    entries: list[GlossaryEntry] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("source"), str):
            raise RuntimeError("glossary entries need a source term")
        source = item["source"].strip()
        keep = item.get("keep", False)
        translations = item.get("translations") or {}
        note = item.get("note")
        if (
            not source
            or not isinstance(keep, bool)
            or not isinstance(translations, Mapping)
            or not all(
                isinstance(code, str) and isinstance(term, str) and term.strip()
                for code, term in translations.items()
            )
            or not (note is None or isinstance(note, str))
            or (not keep and not translations)
        ):
            raise RuntimeError("invalid glossary entry")
        entries.append(
            GlossaryEntry(
                source=source,
                translations=tuple(
                    (code.lower(), term.strip()) for code, term in translations.items()
                ),
                keep=keep,
                note=note.strip() if note and note.strip() else None,
            )
        )
    return tuple(entries)


def _appears(term: str, text: str) -> bool:
    """Whether ``term`` occurs in ``text``, tolerating inflected endings.

    Exact (case-insensitive) containment, or every word of the term found by its stem
    (long words lose their last two letters: ``жительство`` also matches ``жительства``).
    """

    haystack = text.casefold()
    needle = term.casefold()
    if needle in haystack:
        return True
    words = needle.split()
    if not words:
        return False
    return all(
        (word[:-_STEM_DROP] if len(word) >= _STEM_MIN_WORD else word) in haystack
        for word in words
    )


def select_entries(
    glossary: Sequence[GlossaryEntry],
    text: str,
    source_language: str,
    target_language: str,
) -> list[GlossaryEntry]:
    """The entries to put in the prompt for ``text`` (``source_language`` -> ``target_language``).

    An entry applies when it is ``keep`` or when its source-side and target-side renderings
    differ (otherwise there is nothing to enforce). A small glossary is sent whole;
    otherwise only entries whose source-side term appears in ``text``. At most
    :data:`MAX_PROMPT_ENTRIES`.
    """

    applicable = [
        entry
        for entry in glossary
        if entry.keep
        or entry.rendering(source_language) != entry.rendering(target_language)
    ]
    if len(applicable) > SEND_ALL_THRESHOLD:
        applicable = [
            entry
            for entry in applicable
            if _appears(entry.rendering(source_language) or entry.source, text)
        ]
    return applicable[:MAX_PROMPT_ENTRIES]


GLOSSARY_RULE = (
    "Glossary. These terms have a required translation: always use exactly "
    "this wording (adjust only grammar, such as endings), even where another "
    "wording would also be correct:\n"
)
KEEP_RULE = "Never translate these names or terms; keep them exactly as written "


def prompt_has_glossary(system_prompt: str) -> bool:
    """True when :func:`glossary_prompt` added a section to this translation prompt."""

    return GLOSSARY_RULE in system_prompt or KEEP_RULE in system_prompt


def glossary_prompt(
    glossary: Sequence[GlossaryEntry],
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """Prompt section for ``text``, or ``""`` when no entry is relevant."""

    entries = select_entries(glossary, text, source_language, target_language)
    if not entries:
        return ""
    renderings = [e for e in entries if not e.keep]
    kept = [e for e in entries if e.keep]
    parts: list[str] = []
    if renderings:
        lines = []
        for entry in renderings:
            line = (
                f"- {json.dumps(entry.rendering(source_language), ensure_ascii=False)}"
                f" => {json.dumps(entry.rendering(target_language), ensure_ascii=False)}"
            )
            if entry.note:
                line += f" ({entry.note})"
            lines.append(line)
        parts.append(
            GLOSSARY_RULE + "\n".join(lines)
        )
    if kept:
        names = ", ".join(json.dumps(e.source, ensure_ascii=False) for e in kept)
        parts.append(
            KEEP_RULE
            + f"(transliterate only if the target script requires it): {names}"
        )
    return "\n\n".join(parts)


def glossary_keyterms(
    glossary: "Iterable[GlossaryEntry] | Any", language: str | None = None
) -> list[str]:
    """Terms to boost in speech recognition (Deepgram ``keyterm``), de-duplicated.

    ``glossary`` is a parsed glossary or any settings object with a ``glossary``
    attribute (``InterpreterSettings`` / ``AgentTranslationSettings``). With ``language``,
    only the renderings spoken in that language (see :meth:`GlossaryEntry.rendering`);
    without, every rendering. Capped at :data:`MAX_KEYTERMS`.
    """

    entries: Iterable[GlossaryEntry] = getattr(glossary, "glossary", glossary)
    terms: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if language is not None:
            candidates: Iterable[str | None] = (entry.rendering(language),)
        else:
            candidates = (entry.source, *(term for _, term in entry.translations))
        for term in candidates:
            if term and term.casefold() not in seen:
                seen.add(term.casefold())
                terms.append(term)
            if len(terms) >= MAX_KEYTERMS:
                return terms
    return terms

"""Instant / Balanced / Accurate interpreter modes (RM-011).

``interpreter.latency_mode`` trades speed for care. A mode is a *profile* of the cascaded
engine plus a preference for the engine itself:

=========  ===================================  ============  =========  ===============
mode       engine when ``engine`` is not set    turn pause    chunking   translation model
=========  ===================================  ============  =========  ===============
instant    realtime if allowed, else cascaded   0.2 s         clause     gpt-4.1-nano
                                                              (>= 20)
balanced   cascaded (today's defaults)          0.3 s         sentence / gpt-4.1-mini
                                                              clause (>= 40)
accurate   cascaded                             0.7 s         sentence   gpt-4.1
                                                              only
=========  ===================================  ============  =========  ===============

*Allowed* for the realtime engine means ``direction == "one_way"`` and a target language the
realtime model can speak (it translates one stream into one fixed output language). An
explicit ``engine`` in the config always wins over the engine a mode would pick. The
profile also fixes the translation context: every mode passes the last
``context_pairs`` (4) utterance pairs to the translator.

``provider_settings.llm.model`` still overrides the profile's translation model. The
realtime engine ignores the profile (it has no turn pause, chunking or translator).

The Platform applies the same engine rule when it stores the config (``instant`` +
``one_way`` + supported target => ``engine: "realtime"``), so the contract it sends to
Runtime already names the engine; Runtime applies the rule itself for clients that omit it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .language_capabilities import realtime_output_languages

LATENCY_MODES = ("instant", "balanced", "accurate")
DEFAULT_LATENCY_MODE = "balanced"

# gpt-realtime-translate output languages come from the capability table (ST-20).
_REALTIME_OUTPUT_LANGUAGES = realtime_output_languages()


@dataclass(frozen=True)
class LatencyProfile:
    """Tuning of the cascaded engine for one latency mode."""

    mode: str
    turn_pause_secs: float
    # None: cut the translation at sentence boundaries only.
    clause_min_chars: int | None
    translation_model: str
    context_pairs: int = 4


PROFILES = {
    "instant": LatencyProfile("instant", 0.2, 20, "gpt-4.1-nano"),
    "balanced": LatencyProfile("balanced", 0.3, 40, "gpt-4.1-mini"),
    "accurate": LatencyProfile("accurate", 0.7, None, "gpt-4.1"),
}


def profile_for(mode: str | None) -> LatencyProfile:
    return PROFILES[mode or DEFAULT_LATENCY_MODE]


def realtime_allowed(direction: str | None, target: str | None) -> bool:
    """Whether the realtime engine can serve ``direction`` with output language ``target``."""

    return direction == "one_way" and target in _REALTIME_OUTPUT_LANGUAGES


def default_engine_for(
    mode: str | None,
    direction: str | None,
    target: str | None,
    *,
    has_glossary: bool = False,
) -> str:
    """The engine a mode picks when the config does not name one.

    A glossary (ST-41) keeps ``instant`` on the cascaded engine: the realtime model
    accepts no instructions or glossary, so it could not honour it.
    """

    if has_glossary:
        return "cascaded"
    if (mode or DEFAULT_LATENCY_MODE) == "instant" and realtime_allowed(direction, target):
        return "realtime"
    return "cascaded"

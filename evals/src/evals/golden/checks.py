"""Pure checks for the golden-path suite: an :class:`Observation` in, :class:`CheckResult` out.

Nothing here touches the network or the clock, so every check is unit tested.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..metrics import _contains_term, _fold, percentile, script_of
from .scenarios import Expect

SESSION_FAILED_EVENTS = frozenset({"SessionFailed", "RuntimeFailed"})
# ProviderFailed is the pre-ST-10 name, kept so older runtimes still match.
PROVIDER_ERROR_EVENTS = frozenset({"ProviderError", "ProviderFailed"})
SCRIPT_LANGUAGE = {"cyrillic": "ru", "hangul": "ko", "latin": "en"}


@dataclass
class Observation:
    """What a scenario run saw. Latencies are milliseconds from the end of the caller's speech
    to the first audible reply (negative: the reply started while the caller still spoke)."""

    played: int = 0  # utterances the caller played
    heard_latencies_ms: list[float] = field(default_factory=list)  # one per heard reply
    replies: list[str] = field(default_factory=list)  # assistant transcripts, in order
    events: list[dict[str, Any]] = field(default_factory=list)  # runtime-state events
    session_status: str | None = None
    session_error: str | None = None  # the Platform refused / failed to start the session

    @property
    def heard(self) -> int:
        return len(self.heard_latencies_ms)

    @property
    def event_types(self) -> list[str]:
        return [str(e.get("event_type")) for e in self.events]


@dataclass
class CheckResult:
    name: str
    passed: bool
    measured: Any = None
    threshold: Any = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_language(text: str) -> str | None:
    """Dominant script of ``text`` as ``ru`` (Cyrillic), ``ko`` (Hangul) or ``en`` (Latin).

    A script heuristic, not a language identifier: any Latin-script text counts as ``en``.
    ``None`` when the text has no letters in these scripts.
    """

    counts = {"cyrillic": 0, "hangul": 0, "latin": 0}
    for char in text:
        script = script_of(char)
        if script in counts:
            counts[script] += 1
    best = max(counts, key=lambda s: counts[s])
    return SCRIPT_LANGUAGE[best] if counts[best] else None


def check_reply_heard(obs: Observation) -> CheckResult:
    return CheckResult("reply_heard", obs.heard >= 1, obs.heard, ">= 1",
                       f"{obs.heard} of {obs.played} utterances got an audible reply")


def check_min_replies(obs: Observation, minimum: int) -> CheckResult:
    return CheckResult("min_replies", len(obs.replies) >= minimum, len(obs.replies), f">= {minimum}")


def check_reply_latency(obs: Observation, max_ms: float) -> CheckResult:
    if not obs.heard_latencies_ms:
        return CheckResult("reply_latency", False, None, f"<= {max_ms:g} ms", "no reply was heard")
    worst = max(obs.heard_latencies_ms)
    return CheckResult("reply_latency", worst <= max_ms, round(worst, 1), f"<= {max_ms:g} ms",
                       f"worst of {obs.heard} replies")


def check_p95(obs: Observation, max_ms: float) -> CheckResult:
    p95 = percentile(obs.heard_latencies_ms, 95)
    if p95 is None:
        return CheckResult("p95_latency", False, None, f"<= {max_ms:g} ms", "no reply was heard")
    return CheckResult("p95_latency", p95 <= max_ms, round(p95, 1), f"<= {max_ms:g} ms",
                       f"over {obs.heard} replies")


def check_reply_language(obs: Observation, language: str) -> CheckResult:
    detected = [detect_language(reply) for reply in obs.replies]
    if not detected:
        return CheckResult("reply_language", False, [], language, "no reply transcript")
    return CheckResult("reply_language", all(d == language for d in detected),
                       [d or "?" for d in detected], language)


def check_heard_ratio(obs: Observation, minimum: float) -> CheckResult:
    if obs.played == 0:
        return CheckResult("heard_ratio", False, None, f">= {minimum:g}", "no utterance was played")
    ratio = obs.heard / obs.played
    return CheckResult("heard_ratio", ratio >= minimum, round(ratio, 3), f">= {minimum:g}",
                       f"{obs.heard}/{obs.played}")


def check_glossary_terms(obs: Observation, terms: Sequence[str]) -> CheckResult:
    text = _fold(" ".join(obs.replies))
    missing = [t for t in terms if not _contains_term(text, _fold(t))]
    return CheckResult("glossary_terms", not missing, f"{len(terms) - len(missing)}/{len(terms)} present",
                       "all present", f"missing: {', '.join(missing)}" if missing else "")


def check_provider_error(obs: Observation) -> CheckResult:
    seen = sorted({t for t in obs.event_types if t in PROVIDER_ERROR_EVENTS})
    return CheckResult("provider_error", bool(seen), seen or "none observed", "ProviderError observed")


def check_session_failed(obs: Observation) -> CheckResult:
    seen = sorted({t for t in obs.event_types if t in SESSION_FAILED_EVENTS})
    refused = bool(obs.session_error and "runtime_unavailable" in obs.session_error)
    failed = bool(seen) or obs.session_status == "failed" or refused
    measured = seen or ([obs.session_status] if obs.session_status == "failed" else
                        ["runtime_unavailable"] if refused else "none observed")
    return CheckResult("session_failed", failed, measured, "session failed")


def check_tool_called(obs: Observation, tool: str) -> CheckResult:
    names = [str((e.get("payload") or {}).get("tool_name")) for e in obs.events
             if e.get("event_type") == "ToolCalled"]
    return CheckResult("tool_called", tool in names, names or "none observed", tool)


def run_checks(expect: Expect, obs: Observation) -> list[CheckResult]:
    """Checks for every threshold the scenario sets, plus ``reply_heard`` unless it expects a failure."""

    results: list[CheckResult] = []
    expects_failure = expect.expect_session_failed or expect.expect_provider_error
    if not expects_failure:
        results.append(check_reply_heard(obs))
    if expect.min_replies is not None:
        results.append(check_min_replies(obs, expect.min_replies))
    if expect.max_reply_latency_ms is not None:
        results.append(check_reply_latency(obs, expect.max_reply_latency_ms))
    if expect.max_p95_ms is not None:
        results.append(check_p95(obs, expect.max_p95_ms))
    if expect.min_heard_ratio is not None:
        results.append(check_heard_ratio(obs, expect.min_heard_ratio))
    if expect.reply_language is not None:
        results.append(check_reply_language(obs, expect.reply_language))
    if expect.glossary_terms:
        results.append(check_glossary_terms(obs, expect.glossary_terms))
    if expect.expect_provider_error:
        results.append(check_provider_error(obs))
    if expect.expect_session_failed:
        results.append(check_session_failed(obs))
    if expect.tool_called:
        results.append(check_tool_called(obs, expect.tool_called))
    return results

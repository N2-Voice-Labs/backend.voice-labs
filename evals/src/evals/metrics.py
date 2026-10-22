"""Small dependency-free evaluation metrics.

WER, code-switch aware WER (per script), CER, chrF, terminology recall and latency
summaries. Everything is deterministic and pure Python so it can run anywhere.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_HANGUL = "\u1100-\u11ff\u3130-\u318f\uac00-\ud7af"


def _words(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return re.findall(r"[\w']+", normalized)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        next_row = [i]
        for j, candidate in enumerate(hyp, 1):
            next_row.append(
                min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (token != candidate))
            )
        row = next_row
    return row[-1] / max(len(ref), 1)


def _fold(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().replace("\u0451", "\u0435")


def terminology_accuracy(reference: str, hypothesis: str, terms: list[str]) -> float:
    """Share of the terms present in ``reference`` that are also present in ``hypothesis``.

    A term may list alternatives separated by ``|`` (any one counts). Returns 1.0 when
    no term occurs in the reference.
    """

    normalized_terms = {_fold(term) for term in terms if term.strip()}
    reference = _fold(reference)
    hypothesis = _fold(hypothesis)
    expected = {term for term in normalized_terms if _contains_term(reference, term)}
    if not expected:
        return 1.0
    actual = {term for term in normalized_terms if _contains_term(hypothesis, term)}
    return len(expected & actual) / len(expected)


def _contains_term(text: str, term: str) -> bool:
    """Whole-word match; ``a|b`` accepts any alternative.

    Hangul is agglutinative (particles attach to nouns: 비자 -> 비자를), so a Hangul
    neighbour does not count as a word character for boundary purposes.
    """

    for alternative in term.split("|"):
        alternative = alternative.strip()
        if not alternative:
            continue
        pattern = (
            rf"(?<!(?![{_HANGUL}])\w){re.escape(alternative)}(?!(?![{_HANGUL}])\w)"
        )
        if re.search(pattern, text) is not None:
            return True
    return False


# --------------------------------------------------------------------------------------
# Edit distance helpers
# --------------------------------------------------------------------------------------


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    row = list(range(len(hyp) + 1))
    for i, token in enumerate(ref, 1):
        next_row = [i]
        for j, candidate in enumerate(hyp, 1):
            next_row.append(
                min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (token != candidate))
            )
        row = next_row
    return row[-1]


def _align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment as ``(op, ref_index, hyp_index)`` with op in ok/sub/del/ins."""

    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
        ):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


# --------------------------------------------------------------------------------------
# Code-switch aware WER
# --------------------------------------------------------------------------------------


def script_of(token: str) -> str:
    """Dominant script of a token: hangul, cyrillic, latin or other (digits, symbols)."""

    for char in token:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name.startswith("HANGUL"):
            return "hangul"
        if name.startswith("CYRILLIC"):
            return "cyrillic"
        if name.startswith("LATIN"):
            return "latin"
        return "other"
    return "other"


def _normalized_tokens(value: str) -> list[str]:
    """NFC, casefold, ё -> е so Russian spelling variants do not count as errors."""

    return [token.replace("\u0451", "\u0435") for token in _words(value)]


def code_switch_wer(reference: str, hypothesis: str) -> dict[str, object]:
    """WER with errors attributed to the script (language segment) they occur in.

    Each reference word is assigned to its script; a substitution or deletion is charged
    to the reference word's script, an insertion to the hypothesis word's script. So a
    Latin word the recognizer transliterated into Cyrillic ("book" -> "бук") is a *latin*
    error, which exposes code-switch failures that one blended WER hides.

    Returns ``{"wer", "ref_words", "errors", "by_script": {script: {"ref_words",
    "errors", "wer"}}}``; per-script ``wer`` is ``None`` when the reference has no words
    in that script.
    """

    ref, hyp = _normalized_tokens(reference), _normalized_tokens(hypothesis)
    ref_words: Counter[str] = Counter(script_of(t) for t in ref)
    errors: Counter[str] = Counter()
    for op, ref_index, hyp_index in _align(ref, hyp):
        if op in ("sub", "del") and ref_index is not None:
            errors[script_of(ref[ref_index])] += 1
        elif op == "ins" and hyp_index is not None:
            errors[script_of(hyp[hyp_index])] += 1
    by_script: dict[str, dict[str, object]] = {}
    for script in sorted(set(ref_words) | set(errors)):
        words = ref_words[script]
        by_script[script] = {
            "ref_words": words,
            "errors": errors[script],
            "wer": errors[script] / words if words else None,
        }
    total_errors = sum(errors.values())
    return {
        "wer": total_errors / max(len(ref), 1),
        "ref_words": len(ref),
        "errors": total_errors,
        "by_script": by_script,
    }


# --------------------------------------------------------------------------------------
# Character error rate (Korean and other unsegmented text)
# --------------------------------------------------------------------------------------


def _chars(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", value).casefold().replace("\u0451", "\u0435")
    return [char for char in normalized if char.isalnum()]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character-level edit distance over letters and digits (spaces, punctuation ignored).

    Preferred over WER for Korean, where spacing is inconsistent between transcribers.
    """

    ref, hyp = _chars(reference), _chars(hypothesis)
    return _edit_distance(ref, hyp) / max(len(ref), 1)


# --------------------------------------------------------------------------------------
# chrF (character n-gram F-score) for translation quality
# --------------------------------------------------------------------------------------


def _ngrams(items: Sequence[str], n: int) -> Counter[str]:
    return Counter("\x1f".join(items[i : i + n]) for i in range(len(items) - n + 1))


def chrf(
    reference: str,
    hypothesis: str,
    *,
    char_order: int = 6,
    word_order: int = 0,
    beta: float = 2.0,
) -> float:
    """chrF score on a 0-100 scale: character n-gram F-score (n = 1..6, beta = 2).

    Whitespace is ignored for character n-grams (as in sacreBLEU). ``word_order=2`` adds
    word 1-2-grams (chrF++). Precision and recall are averaged over the n-gram orders
    that exist in both strings, then combined as an F-beta score. Case-sensitive.
    """

    ref_text = unicodedata.normalize("NFC", reference)
    hyp_text = unicodedata.normalize("NFC", hypothesis)
    ref_chars = list(re.sub(r"\s+", "", ref_text))
    hyp_chars = list(re.sub(r"\s+", "", hyp_text))
    if not ref_chars and not hyp_chars:
        return 100.0
    if not ref_chars or not hyp_chars:
        return 0.0
    ref_words = re.findall(r"\w+|[^\w\s]", ref_text)
    hyp_words = re.findall(r"\w+|[^\w\s]", hyp_text)

    precisions: list[float] = []
    recalls: list[float] = []
    orders = [(ref_chars, hyp_chars, n) for n in range(1, char_order + 1)]
    orders += [(ref_words, hyp_words, n) for n in range(1, word_order + 1)]
    for ref_items, hyp_items, n in orders:
        ref_counts, hyp_counts = _ngrams(ref_items, n), _ngrams(hyp_items, n)
        ref_total, hyp_total = sum(ref_counts.values()), sum(hyp_counts.values())
        if not ref_total or not hyp_total:
            continue
        overlap = sum((ref_counts & hyp_counts).values())
        precisions.append(overlap / hyp_total)
        recalls.append(overlap / ref_total)
    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta**2
    return 100.0 * (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


# --------------------------------------------------------------------------------------
# Latency summaries
# --------------------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (``p`` in 0..100); ``None`` for no data."""

    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    rank = (len(data) - 1) * p / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    return float(data[low] + (data[high] - data[low]) * (rank - low))


def latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """``count``, ``p50``, ``p95``, ``mean`` and ``max`` of latency samples (seconds)."""

    data = [v for v in values if v is not None and not math.isnan(v)]
    return {
        "count": len(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "mean": sum(data) / len(data) if data else None,
        "max": max(data) if data else None,
    }

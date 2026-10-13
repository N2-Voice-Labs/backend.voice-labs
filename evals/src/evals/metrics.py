"""Small dependency-free baseline metrics."""

import re
import unicodedata


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


def terminology_accuracy(reference: str, hypothesis: str, terms: list[str]) -> float:
    normalized_terms = {
        unicodedata.normalize("NFC", term).casefold() for term in terms if term.strip()
    }
    reference = unicodedata.normalize("NFC", reference).casefold()
    hypothesis = unicodedata.normalize("NFC", hypothesis).casefold()
    expected = {term for term in normalized_terms if _contains_term(reference, term)}
    if not expected:
        return 1.0
    actual = {term for term in normalized_terms if _contains_term(hypothesis, term)}
    return len(expected & actual) / len(expected)


def _contains_term(text: str, term: str) -> bool:
    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return re.search(pattern, text) is not None

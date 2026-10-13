from src.evals.metrics import terminology_accuracy, word_error_rate


def test_metrics_treat_canonically_equivalent_unicode_as_equal() -> None:
    composed = "café Бишкек"
    decomposed = "cafe\u0301 Бишкек"
    assert word_error_rate(composed, decomposed) == 0.0
    assert terminology_accuracy(composed, decomposed, ["café", "Бишкек"]) == 1.0


def test_exact_transcript_has_zero_wer() -> None:
    assert word_error_rate("hello мир", "hello мир") == 0


def test_terms_are_measured_independently() -> None:
    assert (
        terminology_accuracy(
            "visa appointment", "visa tomorrow", ["visa", "appointment"]
        )
        == 0.5
    )


def test_terminology_matching_requires_word_boundaries_in_mixed_languages() -> None:
    assert terminology_accuracy("visa", "advisable", ["visa"]) == 0.0
    assert (
        terminology_accuracy(
            "виза appointment", "визаж appointment", ["виза", "appointment"]
        )
        == 0.5
    )

import pytest

from src.evals.metrics import (
    character_error_rate,
    chrf,
    code_switch_wer,
    latency_summary,
    percentile,
    script_of,
    terminology_accuracy,
    word_error_rate,
)


def test_wer_counts_substitution_deletion_insertion() -> None:
    assert word_error_rate("a b c d", "a x c") == pytest.approx(0.5)  # 1 sub + 1 del
    assert word_error_rate("a b", "a b c") == pytest.approx(0.5)  # 1 insertion


def test_script_detection() -> None:
    assert script_of("appointment") == "latin"
    assert script_of("завтра") == "cyrillic"
    assert script_of("예약") == "hangul"
    assert script_of("1988") == "other"


def test_code_switch_wer_attributes_errors_to_the_script_that_failed() -> None:
    reference = "Мне нужно book an appointment на завтра"
    hypothesis = "Мне нужно бук энд апойнтмент на завтра"  # English words transliterated
    result = code_switch_wer(reference, hypothesis)
    assert result["ref_words"] == 7
    assert result["errors"] == 3
    assert result["wer"] == pytest.approx(3 / 7)
    by_script = result["by_script"]
    assert by_script["latin"]["wer"] == pytest.approx(1.0)
    assert by_script["cyrillic"]["wer"] == 0.0


def test_code_switch_wer_perfect_and_yo_normalisation() -> None:
    assert code_switch_wer("Всё готово, visa ready", "все готово visa ready")["wer"] == 0.0


def test_code_switch_wer_charges_insertions_to_hypothesis_script() -> None:
    result = code_switch_wer("мне нужна справка", "мне нужна справка please")
    assert result["by_script"]["latin"]["errors"] == 1
    assert result["by_script"]["latin"]["wer"] is None  # no latin words in the reference


def test_korean_cer_ignores_spacing_and_punctuation_but_counts_characters() -> None:
    assert character_error_rate("비자 연장", "비자연장") == 0.0
    assert character_error_rate("안녕하세요, 예약하고 싶습니다.", "안녕하세요 예약하고 싶습니다") == 0.0
    # one wrong syllable out of four
    assert character_error_rate("비자연장", "비자연자") == pytest.approx(0.25)
    assert character_error_rate("비자", "") == 1.0


def test_chrf_identical_is_100_and_unrelated_is_low() -> None:
    sentence = "The doctor will see you in ten minutes."
    assert chrf(sentence, sentence) == 100.0
    assert chrf("The doctor will see you now", "Please bring your passport tomorrow") < 25
    assert chrf("Врач примет вас", "I need a visa") == 0.0
    assert chrf("", "") == 100.0
    assert chrf("abc", "") == 0.0


def test_chrf_orders_a_close_paraphrase_above_a_distant_one() -> None:
    reference = "I would like to extend my residence permit for another year."
    close = "I want to extend my residence permit for one more year."
    far = "The clinic opens at nine in the morning."
    assert chrf(reference, close) > chrf(reference, far) + 20
    assert 0 < chrf(reference, close) < 100


def test_chrf_is_case_sensitive_whitespace_insensitive_and_supports_korean() -> None:
    assert chrf("Visa Appointment", "visa appointment") < 100
    assert chrf("a b c d e f g", "abcdefg") == 100.0
    assert chrf("예약을 취소하고 싶습니다", "예약을 취소하고 싶습니다") == 100.0
    assert chrf("예약을 취소하고 싶습니다", "예약을 변경하고 싶습니다") > 40


def test_chrf_plus_plus_word_ngrams_option() -> None:
    assert chrf("see you now", "see you now", word_order=2) == 100.0
    reordered = chrf("see you now", "now you see", word_order=2)
    assert reordered != chrf("see you now", "now you see")  # word unigrams change the score
    assert 0 < reordered < 100


def test_terminology_handles_hangul_particles_and_alternatives() -> None:
    assert terminology_accuracy("비자 연장", "비자를 연장해야 합니다", ["비자", "연장"]) == 1.0
    assert terminology_accuracy("Fee 25,000 won", "Fee 25000 won", ["25,000|25000"]) == 1.0
    assert terminology_accuracy("born in 1988", "1988년에 태어났습니다", ["1988"]) == 1.0


def test_terminology_substring_inside_latin_word_is_not_a_hit() -> None:
    assert terminology_accuracy("a visa", "advisable", ["visa"]) == 0.0
    assert terminology_accuracy("Всё хорошо", "все хорошо", ["всё"]) == 1.0


def test_percentile_and_latency_summary() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == pytest.approx(8.8)
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 50) is None
    summary = latency_summary(values)
    assert summary["count"] == 5
    assert summary["p50"] == 3.0
    assert summary["max"] == 10.0
    assert summary["mean"] == pytest.approx(4.0)
    assert latency_summary([])["p95"] is None

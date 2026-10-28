import pytest

from src.evals.golden import checks as ck
from src.evals.golden.scenarios import Expect


def obs(**kw):
    return ck.Observation(**kw)


@pytest.mark.parametrize("text, expected", [
    ("Hello, how can I help you today?", "en"),
    ("Здравствуйте, чем могу помочь?", "ru"),
    ("안녕하세요, 무엇을 도와드릴까요?", "ko"),
    ("Вы можете оформить book an appointment на завтра", "ru"),   # code-switch: Cyrillic dominates
    ("Your visa expires, пожалуйста", "en"),
    ("1234 !!!", None),
    ("", None),
])
def test_detect_language(text, expected):
    assert ck.detect_language(text) == expected


def test_reply_heard():
    assert ck.check_reply_heard(obs(played=3, heard_latencies_ms=[1.0])).passed
    result = ck.check_reply_heard(obs(played=3))
    assert not result.passed and result.measured == 0


def test_min_replies():
    assert ck.check_min_replies(obs(replies=["a", "b"]), 2).passed
    result = ck.check_min_replies(obs(replies=["a"]), 2)
    assert not result.passed and (result.measured, result.threshold) == (1, ">= 2")


def test_reply_latency_uses_the_worst_reply():
    assert ck.check_reply_latency(obs(heard_latencies_ms=[800, 2900]), 3000).passed
    result = ck.check_reply_latency(obs(heard_latencies_ms=[800, 3100.04]), 3000)
    assert not result.passed and result.measured == 3100.0
    assert not ck.check_reply_latency(obs(), 3000).passed
    assert ck.check_reply_latency(obs(heard_latencies_ms=[-400]), 3000).passed  # spoke before the caller stopped


def test_p95():
    values = list(range(100, 2100, 100))  # 100..2000 ms
    assert ck.check_p95(obs(heard_latencies_ms=values), 2000).passed
    assert not ck.check_p95(obs(heard_latencies_ms=values), 1500).passed
    assert not ck.check_p95(obs(), 1500).passed


def test_reply_language():
    assert ck.check_reply_language(obs(replies=["Привет", "Как дела"]), "ru").passed
    result = ck.check_reply_language(obs(replies=["Привет", "Hello"]), "ru")
    assert not result.passed and result.measured == ["ru", "en"]
    assert not ck.check_reply_language(obs(), "ru").passed


def test_heard_ratio():
    assert ck.check_heard_ratio(obs(played=30, heard_latencies_ms=[1] * 29), 0.966).passed
    result = ck.check_heard_ratio(obs(played=30, heard_latencies_ms=[1] * 28), 0.966)
    assert not result.passed and result.detail == "28/30"
    assert not ck.check_heard_ratio(obs(), 0.5).passed


def test_glossary_terms():
    replies = ["Я хочу продлить Вид на жительство."]
    assert ck.check_glossary_terms(obs(replies=replies), ["вид на жительство"]).passed
    assert ck.check_glossary_terms(obs(replies=replies), ["permit|вид на жительство"]).passed
    result = ck.check_glossary_terms(obs(replies=replies), ["вид на жительство", "виза"])
    assert not result.passed and "виза" in result.detail and result.measured == "1/2 present"


def test_glossary_term_is_a_whole_word_match():
    assert not ck.check_glossary_terms(obs(replies=["permitted"]), ["permit"]).passed


def events(*types, **payloads):
    return [{"event_type": t, "payload": payloads.get(t, {})} for t in types]


def test_provider_error():
    assert ck.check_provider_error(obs(events=events("RuntimeStarted", "ProviderFailed"))).passed
    assert not ck.check_provider_error(obs(events=events("RuntimeStarted"))).passed


def test_session_failed_from_event_status_or_refusal():
    assert ck.check_session_failed(obs(events=events("SessionFailed"))).passed
    assert ck.check_session_failed(obs(events=events("RuntimeFailed"))).passed
    assert ck.check_session_failed(obs(session_status="failed")).passed
    assert ck.check_session_failed(obs(session_error="POST /voice/sessions -> 503 runtime_unavailable")).passed
    assert not ck.check_session_failed(obs(session_status="active", events=events("SessionEnded"))).passed
    assert not ck.check_session_failed(obs(session_error="POST /voice/sessions -> 409 agent_not_enabled")).passed


def test_tool_called():
    seen = obs(events=events("ToolCalled", ToolCalled={"tool_name": "lookup_order"}))
    assert ck.check_tool_called(seen, "lookup_order").passed
    assert not ck.check_tool_called(seen, "other").passed
    assert not ck.check_tool_called(obs(), "other").passed


def test_run_checks_default_is_reply_heard_only():
    results = ck.run_checks(Expect(), obs(played=1, heard_latencies_ms=[10]))
    assert [r.name for r in results] == ["reply_heard"]


def test_run_checks_skips_reply_heard_when_a_failure_is_expected():
    results = ck.run_checks(Expect(expect_session_failed=True), obs(session_status="failed"))
    assert [r.name for r in results] == ["session_failed"]
    assert all(r.passed for r in results)


def test_run_checks_covers_every_threshold():
    expect = Expect(reply_language="en", min_replies=1, max_reply_latency_ms=1000, max_p95_ms=1000,
                    min_heard_ratio=1.0, glossary_terms=("hi",), expect_provider_error=True,
                    expect_session_failed=True, tool_called="t")
    names = [r.name for r in ck.run_checks(expect, obs())]
    assert names == ["min_replies", "reply_latency", "p95_latency", "heard_ratio", "reply_language",
                     "glossary_terms", "provider_error", "session_failed", "tool_called"]

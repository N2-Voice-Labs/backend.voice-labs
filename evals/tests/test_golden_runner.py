import asyncio

import httpx

from src.evals.golden import runner as rn
from src.evals.golden import scenarios as sc
from tests.golden_fakes import FakeDriver, FakeStack, assistant, no_sleep

SETTINGS = rn.Settings(event_wait=2.0, poll_interval=1.0)


def scenario(tmp_path, **overrides):
    data = {"id": "t", "mode": "agent", "audio": ["a", "b"], "agent_config": {"system_prompt": "hi"},
            "expect": {"min_replies": 1, "reply_language": "ru", "max_reply_latency_ms": 2000}}
    data.update(overrides)
    for name in ("a", "b"):
        (tmp_path / f"{name}.wav").write_bytes(b"")
    return sc.parse_scenario(data, "t.json", tmp_path)


def run(s, stack, driver, tmp_path, env=None):
    return asyncio.run(rn.run_scenario(s, stack.platform(), driver, tmp_path, SETTINGS, "r1", env or {}, no_sleep))


def test_agent_scenario_passes_and_cleans_up(tmp_path):
    stack = FakeStack(messages=assistant("Здравствуйте!", "Мы работаем с девяти."))
    driver = FakeDriver([0.8, 1.2])
    result = run(scenario(tmp_path, audio=["a", "b"]), stack, driver, tmp_path)
    assert result.status == rn.PASSED, result.checks
    assert [c.name for c in result.checks] == ["reply_heard", "min_replies", "reply_latency", "reply_language"]
    assert result.observation.heard_latencies_ms == [800.0, 1200.0]
    assert driver.played == [(["a", "b"], ["A"])]
    assert stack.agent_bodies == [{"system_prompt": "hi", "name": "golden t r1", "mode": "agent"}]
    assert stack.calls.index("POST /voice/sessions/s1/end") < stack.calls.index("DELETE /agents/agent-1")
    assert result.agent_id == "agent-1" and result.conversation_id == "c1"


def test_failed_check_reports_measured_vs_threshold(tmp_path):
    stack = FakeStack(messages=assistant("Hello there"))
    result = run(scenario(tmp_path), stack, FakeDriver([3.5]), tmp_path)
    assert result.status == rn.FAILED
    failed = {c.name: c for c in result.checks if not c.passed}
    assert set(failed) == {"reply_latency", "reply_language"}
    assert failed["reply_latency"].measured == 3500.0 and failed["reply_latency"].threshold == "<= 2000 ms"


def test_assistant_messages_only_and_empty_ones_ignored(tmp_path):
    messages = [{"role": "user", "content": "Привет"}, {"role": "assistant", "content": ""},
                {"role": "assistant", "content": "Здравствуйте"}]
    result = run(scenario(tmp_path), FakeStack(messages=messages), FakeDriver([0.5]), tmp_path)
    assert result.observation.replies == ["Здравствуйте"]


def test_skipped_scenario_makes_no_calls(tmp_path):
    s = scenario(tmp_path, requires_env=["EVALS_TOOL_SERVER_URL"])
    stack, driver = FakeStack(), FakeDriver()
    result = run(s, stack, driver, tmp_path)
    assert result.status == rn.SKIPPED and "EVALS_TOOL_SERVER_URL" in result.reason
    assert stack.calls == [] and driver.played == []


def test_missing_human_audio_skips(tmp_path):
    s = scenario(tmp_path, audio=["human_v1:ru_001"])
    result = run(s, FakeStack(), FakeDriver(), tmp_path)
    assert result.status == rn.SKIPPED and "EVALS_HUMAN_AUDIO_DIR" in result.reason


def test_driver_error_fails_with_run_completed_check_but_still_cleans_up(tmp_path):
    stack = FakeStack()
    result = run(scenario(tmp_path), stack, FakeDriver(fail=RuntimeError("livekit down")), tmp_path)
    assert result.status == rn.FAILED
    assert result.checks[0].name == "run_completed" and "livekit down" in result.checks[0].measured
    assert "POST /voice/sessions/s1/end" in stack.calls and "DELETE /agents/agent-1" in stack.calls


def test_agent_creation_error_is_a_failed_result_not_an_exception(tmp_path):
    s = scenario(tmp_path, agent_config={"unknown": 1})
    platform = rn.GoldenPlatform("http://p", client=httpx.AsyncClient(
        base_url="http://p", transport=httpx.MockTransport(lambda r: httpx.Response(422, json={"detail": "bad"}))))
    result = asyncio.run(rn.run_scenario(s, platform, FakeDriver(), tmp_path, SETTINGS, "r1", {}, no_sleep))
    assert result.status == rn.FAILED and "422" in result.reason


def test_session_failed_expected_and_observed_passes(tmp_path):
    s = scenario(tmp_path, expect={"expect_session_failed": True}, audio=["a", "b"])
    state = {"session_id": "s1", "session_status": "failed",
             "events": [{"event_type": "ProviderFailed", "payload": {}}, {"event_type": "SessionFailed", "payload": {}}]}
    driver = FakeDriver([None])
    result = run(s, FakeStack(state=state), driver, tmp_path)
    assert result.status == rn.PASSED, result.checks
    assert driver.played[0][0] == ["a"]  # only the first clip is played when a failure is expected
    assert [c.name for c in result.checks] == ["session_failed"]


def test_session_failed_expected_but_stack_is_healthy_fails(tmp_path):
    s = scenario(tmp_path, expect={"expect_session_failed": True})
    result = run(s, FakeStack(), FakeDriver([0.5]), tmp_path)
    assert result.status == rn.FAILED and result.checks[0].name == "session_failed"


def test_runtime_unavailable_503_counts_as_session_failed(tmp_path):
    s = scenario(tmp_path, expect={"expect_session_failed": True})
    stack = FakeStack(create_session_status=503)
    driver = FakeDriver()
    result = run(s, stack, driver, tmp_path)
    assert result.status == rn.PASSED, result.checks
    assert driver.played == []
    assert "DELETE /agents/agent-1" in stack.calls  # agent still cleaned up


def test_unexpected_503_is_a_failure_with_the_reason(tmp_path):
    result = run(scenario(tmp_path), FakeStack(create_session_status=503), FakeDriver(), tmp_path)
    assert result.status == rn.FAILED
    assert result.checks[0].name == "reply_heard" and not result.checks[0].passed


def test_two_person_creates_guest_and_routes_speakers(tmp_path):
    config = {"interpreter": {"languages": ["ko", "en"], "direction": "one_way"}}
    s = scenario(tmp_path, mode="interpreter", agent_config=config, two_person=True,
                 audio=["a", {"ref": "b", "speaker": 1}], expect={"min_replies": 2})
    stack = FakeStack(messages=assistant("Hello", "안녕하세요"))
    driver = FakeDriver([0.5])
    result = run(s, stack, driver, tmp_path)
    assert result.status == rn.PASSED, result.checks
    assert "POST /voice/sessions/s1/participants" in stack.calls
    assert driver.played[0][1] == ["A-ko", "B-en"]


def test_polls_until_transcripts_arrive(tmp_path):
    stack = FakeStack()
    seen = {"n": 0}
    original = stack.handler

    def handler(request):
        if request.url.path == "/conversations/c1/messages":
            seen["n"] += 1
            stack.messages = assistant("Здравствуйте") if seen["n"] >= 3 else []
        return original(request)

    stack.handler = handler
    result = run(scenario(tmp_path, audio=["a"], expect={"min_replies": 1}), stack, FakeDriver([0.5]), tmp_path)
    assert result.status == rn.PASSED and seen["n"] == 3


def test_result_serialises(tmp_path):
    result = run(scenario(tmp_path), FakeStack(messages=assistant("Здравствуйте")), FakeDriver([0.5]), tmp_path)
    data = result.to_dict()
    assert data["status"] == "passed" and data["observation"]["replies"] == ["Здравствуйте"]
    assert data["checks"][0]["name"] == "reply_heard"

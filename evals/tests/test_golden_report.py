import json

import httpx

from src.evals.golden import cli, report
from src.evals.golden.checks import CheckResult, Observation
from src.evals.golden.runner import FAILED, PASSED, SKIPPED, ScenarioResult
from tests.golden_fakes import FakeDriver, FakeStack, assistant

META = {"run_id": "20260101T000000Z-golden", "date": "2026-01-01T00:00:00+00:00", "platform": "http://p"}


def result(id, status, enforce=False, checks=(), reason=""):
    obs = Observation(played=1, heard_latencies_ms=[900.0], replies=["Привет | мир"])
    return ScenarioResult(id, "agent", status, enforce, reason, list(checks), obs, 1.5)


FAIL_CHECK = CheckResult("reply_latency", False, 3100.0, "<= 3000 ms", "worst of 1 replies")
OK_CHECK = CheckResult("reply_heard", True, 1, ">= 1")

MIXED = [
    result("ok", PASSED, checks=[OK_CHECK]),
    result("soft", FAILED, enforce=False, checks=[FAIL_CHECK]),
    result("skipme", SKIPPED, reason="EVALS_TOOL_SERVER_URL is not set"),
]


# -- exit code ------------------------------------------------------------------------------------


def test_exit_code_is_zero_for_report_only_failures_and_skips():
    assert report.exit_code(MIXED) == 0
    assert report.exit_code([]) == 0


def test_exit_code_is_one_only_for_enforced_failures():
    assert report.exit_code(MIXED + [result("hard", FAILED, enforce=True, checks=[FAIL_CHECK])]) == 1
    assert report.exit_code(MIXED + [result("hard", PASSED, enforce=True, checks=[OK_CHECK])]) == 0
    assert report.exit_code(MIXED + [result("hard", SKIPPED, enforce=True, reason="x")]) == 0


# -- rendering ----------------------------------------------------------------------------------


def test_markdown_has_summary_table_and_check_details():
    text = report.render_markdown(MIXED, META)
    assert "**1 passed, 1 failed, 1 skipped** (0 enforced failure(s)); exit code 0." in text
    assert "| soft | agent | no | FAILED (report-only) | failed: reply_latency |" in text
    assert "| skipme | agent | no | SKIPPED | EVALS_TOOL_SERVER_URL is not set |" in text
    assert "| reply_latency | FAIL | 3100.0 | <= 3000 ms | worst of 1 replies |" in text
    assert "Привет \\| мир" in text  # pipes in replies cannot break the table


def test_enforced_failure_is_labelled_plainly():
    text = report.render_markdown([result("hard", FAILED, enforce=True, checks=[FAIL_CHECK])], META)
    assert "| hard | agent | yes | FAILED | failed: reply_latency |" in text
    assert "exit code 1" in text


def test_write_report_writes_both_files(tmp_path):
    json_path, md_path = report.write_report(MIXED, META, tmp_path / "out" / "run")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"] == {"passed": 1, "failed": 1, "skipped": 1, "exit_code": 0}
    assert [s["id"] for s in data["scenarios"]] == ["ok", "soft", "skipme"]
    assert data["scenarios"][1]["checks"][0]["measured"] == 3100.0
    assert md_path.read_text(encoding="utf-8").startswith("# Golden-path suite")


# -- discord --------------------------------------------------------------------------------------


class Recorder:
    def __init__(self, status=204):
        self.requests, self.status = [], status

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(self.status)


def test_discord_message_lists_failures_and_is_bounded():
    text = report.discord_message(MIXED, META)
    assert "1 passed, 1 failed, 1 skipped" in text and "- soft (report-only): reply_latency" in text
    many = [result(f"s{i}", FAILED, checks=[FAIL_CHECK]) for i in range(500)]
    assert len(report.discord_message(many, META)) <= report.DISCORD_LIMIT


def test_post_discord_sends_json_content_without_network():
    recorder = Recorder()
    error = report.post_discord("https://discord.example/api/webhooks/1/secret", "hello",
                                httpx.MockTransport(recorder))
    assert error is None
    assert json.loads(recorder.requests[0].content) == {"content": "hello"}


def test_post_discord_errors_never_leak_the_url():
    url = "https://discord.example/api/webhooks/1/secret"
    assert report.post_discord(url, "x", httpx.MockTransport(Recorder(500))) == "webhook answered 500"

    def boom(request):
        raise httpx.ConnectError("no route to " + str(request.url))

    error = report.post_discord(url, "x", httpx.MockTransport(boom))
    assert error == "webhook failed: ConnectError" and "secret" not in error


def test_notify_discord_only_posts_when_the_env_var_is_set():
    recorder = Recorder()
    transport = httpx.MockTransport(recorder)
    assert "not set" in report.notify_discord(MIXED, META, {}, transport)
    assert recorder.requests == []
    status = report.notify_discord(MIXED, META, {"EVALS_DISCORD_WEBHOOK_URL": "https://d.example/hook"}, transport)
    assert status == "discord: posted" and len(recorder.requests) == 1
    assert "hook" not in status


# -- CLI --------------------------------------------------------------------------------------------


def write_scenarios(directory, enforce):
    (directory / "a.json").write_text(json.dumps({
        "id": "a", "mode": "agent", "audio": ["en-ru-01"], "enforce": enforce,
        "agent_config": {"system_prompt": "x"}, "expect": {"reply_language": "ru"}}))
    (directory / "b.json").write_text(json.dumps({
        "id": "b", "mode": "agent", "audio": ["en-ru-01"], "requires_env": ["NOPE"], "agent_config": {}}))


def cli_run(tmp_path, enforce, replies, extra=(), env=None):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir(exist_ok=True)
    write_scenarios(scenarios, enforce)
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    (audio / "en-ru-01.wav").write_bytes(b"")
    stack = FakeStack(messages=assistant(*replies))
    recorder = Recorder()
    code = cli.main(
        ["--scenarios", str(scenarios), "--audio-dir", str(audio), "--report-dir", str(tmp_path / "rep"),
         "--platform", "http://platform", "--event-wait", "0", *extra],
        env=env or {}, driver=FakeDriver([0.5]), transport=httpx.MockTransport(stack.handler),
        webhook_transport=httpx.MockTransport(recorder))
    return code, recorder


def test_cli_refuses_without_confirm_billable(tmp_path, capsys):
    scenarios = tmp_path / "s"
    scenarios.mkdir()
    write_scenarios(scenarios, False)
    code = cli.main(["--scenarios", str(scenarios), "--audio-dir", str(tmp_path)], env={})
    assert code == 2
    assert "--confirm-billable" in capsys.readouterr().err


def test_cli_dry_run_prints_plan_with_skip_reasons(tmp_path, capsys):
    scenarios = tmp_path / "s"
    scenarios.mkdir()
    write_scenarios(scenarios, False)
    (tmp_path / "en-ru-01.wav").write_bytes(b"")
    assert cli.main(["--scenarios", str(scenarios), "--audio-dir", str(tmp_path), "--dry-run"], env={}) == 0
    out = capsys.readouterr().out
    assert "RUN  a" in out and "SKIP b" in out and "NOPE is not set" in out


def test_cli_report_only_failure_exits_zero_and_writes_reports(tmp_path):
    code, _ = cli_run(tmp_path, enforce=False, replies=["Hello"], extra=["--confirm-billable"])
    assert code == 0
    data = json.loads((tmp_path / "rep" / "report.json").read_text(encoding="utf-8"))
    assert {s["id"]: s["status"] for s in data["scenarios"]} == {"a": "failed", "b": "skipped"}
    assert (tmp_path / "rep" / "report.md").exists()


def test_cli_enforced_failure_exits_one(tmp_path):
    code, _ = cli_run(tmp_path, enforce=True, replies=["Hello"], extra=["--confirm-billable"])
    assert code == 1


def test_cli_enforced_pass_exits_zero_and_posts_to_discord_when_configured(tmp_path):
    env = {"EVALS_DISCORD_WEBHOOK_URL": "https://d.example/hook"}
    code, recorder = cli_run(tmp_path, enforce=True, replies=["Здравствуйте"],
                             extra=["--confirm-billable"], env=env)
    assert code == 0
    assert len(recorder.requests) == 1


def test_cli_only_filters_and_unknown_id_is_a_usage_error(tmp_path, capsys):
    code, _ = cli_run(tmp_path, enforce=False, replies=["Здравствуйте"], extra=["--confirm-billable", "--only", "a"])
    assert code == 0
    data = json.loads((tmp_path / "rep" / "report.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in data["scenarios"]] == ["a"]
    code, _ = cli_run(tmp_path, enforce=False, replies=[], extra=["--confirm-billable", "--only", "zzz"])
    assert code == 2 and "unknown scenario id" in capsys.readouterr().err


def test_cli_rejects_dataset_id_typos(tmp_path, capsys):
    scenarios = tmp_path / "s"
    scenarios.mkdir()
    (scenarios / "a.json").write_text(json.dumps({"id": "a", "mode": "agent", "audio": ["en-xx-99"]}))
    assert cli.main(["--scenarios", str(scenarios), "--dry-run"], env={}) == 2
    assert "unknown dataset ids: a:en-xx-99" in capsys.readouterr().err


def test_cli_all_skipped_needs_no_platform(tmp_path):
    scenarios = tmp_path / "s"
    scenarios.mkdir()
    (scenarios / "b.json").write_text(json.dumps({"id": "b", "mode": "agent", "audio": ["en-ru-01"],
                                                  "requires_env": ["NOPE"]}))
    code = cli.main(["--scenarios", str(scenarios), "--confirm-billable", "--report-dir", str(tmp_path / "r"),
                     "--audio-dir", str(tmp_path)], env={}, driver=FakeDriver(),
                    transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert code == 0
    assert json.loads((tmp_path / "r" / "report.json").read_text())["scenarios"][0]["status"] == "skipped"

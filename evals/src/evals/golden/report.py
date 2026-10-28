"""Golden-path report: JSON + Markdown files, exit-code rule, optional Discord post."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from .runner import FAILED, PASSED, SKIPPED, ScenarioResult

DISCORD_ENV = "EVALS_DISCORD_WEBHOOK_URL"
DISCORD_LIMIT = 1900  # Discord rejects messages over 2000 characters


def counts(results: list[ScenarioResult]) -> dict[str, int]:
    return {status: sum(r.status == status for r in results) for status in (PASSED, FAILED, SKIPPED)}


def enforced_failures(results: list[ScenarioResult]) -> list[ScenarioResult]:
    return [r for r in results if r.enforce and r.status == FAILED]


def exit_code(results: list[ScenarioResult]) -> int:
    """1 only when an enforced scenario failed; report-only failures and skips never fail the run."""

    return 1 if enforced_failures(results) else 0


def _cell(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _label(result: ScenarioResult) -> str:
    return result.status.upper() + ("" if result.status != FAILED or result.enforce else " (report-only)")


def render_markdown(results: list[ScenarioResult], meta: Mapping[str, Any]) -> str:
    total = counts(results)
    lines = [
        "# Golden-path suite",
        "",
        f"Run `{meta.get('run_id', '')}` on {meta.get('date', '')}"
        + (f" against {meta['platform']}" if meta.get("platform") else ""),
        "",
        f"**{total[PASSED]} passed, {total[FAILED]} failed, {total[SKIPPED]} skipped** "
        f"({len(enforced_failures(results))} enforced failure(s)); exit code {exit_code(results)}.",
        "",
        "| Scenario | Mode | Enforced | Result | Note |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        failing = [c.name for c in r.checks if not c.passed]
        note = r.reason if r.status == SKIPPED or not failing else "failed: " + ", ".join(failing)
        lines.append(f"| {r.id} | {r.mode} | {'yes' if r.enforce else 'no'} | {_label(r)} | {_cell(note)} |")
    for r in results:
        if not r.checks:
            continue
        lines += ["", f"## {r.id}", "", "| Check | Result | Measured | Threshold | Detail |", "|---|---|---|---|---|"]
        for c in r.checks:
            lines.append(f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {_cell(c.measured)} | "
                         f"{_cell(c.threshold)} | {_cell(c.detail)} |")
        if r.observation and r.observation.replies:
            lines += ["", "Replies:"] + [f"- {_cell(text)}" for text in r.observation.replies[:5]]
    lines.append("")
    return "\n".join(lines)


def to_json(results: list[ScenarioResult], meta: Mapping[str, Any]) -> dict[str, Any]:
    return {"meta": dict(meta), "summary": {**counts(results), "exit_code": exit_code(results)},
            "scenarios": [r.to_dict() for r in results]}


def write_report(results: list[ScenarioResult], meta: Mapping[str, Any], directory: Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` into ``directory`` (created)."""

    directory.mkdir(parents=True, exist_ok=True)
    json_path, md_path = directory / "report.json", directory / "report.md"
    json_path.write_text(json.dumps(to_json(results, meta), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(results, meta), encoding="utf-8")
    return json_path, md_path


def discord_message(results: list[ScenarioResult], meta: Mapping[str, Any]) -> str:
    total = counts(results)
    head = (f"Golden-path {meta.get('run_id', '')}: {total[PASSED]} passed, {total[FAILED]} failed, "
            f"{total[SKIPPED]} skipped; {len(enforced_failures(results))} enforced failure(s)")
    lines = [head]
    for r in results:
        if r.status == FAILED:
            failing = ", ".join(c.name for c in r.checks if not c.passed) or r.reason
            lines.append(f"- {r.id}{'' if r.enforce else ' (report-only)'}: {failing}")
    message = "\n".join(lines)
    return message if len(message) <= DISCORD_LIMIT else message[: DISCORD_LIMIT - 3] + "..."


def post_discord(url: str, message: str, transport: httpx.BaseTransport | None = None) -> str | None:
    """POST ``message`` to a Discord webhook; ``None`` on success, else a short error (never the URL)."""

    try:
        with httpx.Client(transport=transport, timeout=10.0) as client:
            response = client.post(url, json={"content": message})
        if response.status_code >= 300:
            return f"webhook answered {response.status_code}"
    except httpx.HTTPError as exc:
        return f"webhook failed: {type(exc).__name__}"
    return None


def notify_discord(results: list[ScenarioResult], meta: Mapping[str, Any], env: Mapping[str, str],
                   transport: httpx.BaseTransport | None = None) -> str:
    """Post the summary when ``EVALS_DISCORD_WEBHOOK_URL`` is set; returns a status line to print."""

    url = (env.get(DISCORD_ENV) or "").strip()
    if not url:
        return f"discord: {DISCORD_ENV} not set, not posted"
    error = post_discord(url, discord_message(results, meta), transport)
    return f"discord: {error}" if error else "discord: posted"

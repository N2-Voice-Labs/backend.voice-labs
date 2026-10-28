"""Golden-path suite command line.

    python -m src.evals.golden --scenarios scenarios/golden --confirm-billable \\
        [--only ru_agent_reply_language] [--report-dir DIR] [--platform http://localhost:8300]

Without ``--confirm-billable`` (or with ``--dry-run``) only the plan is printed: which scenarios
would run and which are skipped, with the reason. No network is touched. Exit codes: 0 all good
(report-only failures included), 1 an enforced scenario failed, 2 usage or setup error.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..dataset import DEFAULT_MANIFEST, load_manifest
from ..generate_audio import DEFAULT_OUT as DEFAULT_AUDIO_DIR
from ..run import machine_description
from .report import exit_code, notify_discord, render_markdown, write_report
from .runner import Driver, GoldenPlatform, LivekitDriver, ScenarioResult, Settings, SKIPPED, run_scenario
from .scenarios import SCENARIO_DIR, Scenario, ScenarioError, load_scenarios, select, skip_reason, unknown_dataset_ids

DEFAULT_REPORT_ROOT = Path.home() / ".cache" / "voice-labs-evals" / "golden"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals.golden", description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenarios", type=Path, default=SCENARIO_DIR,
                        help="scenario directory or file (default: evals/scenarios/golden)")
    parser.add_argument("--only", action="append", default=[], metavar="ID",
                        help="run only this scenario id (repeatable, or comma separated)")
    parser.add_argument("--report-dir", type=Path, help="where report.json / report.md go "
                        "(default ~/.cache/voice-labs-evals/golden/<run-id>)")
    parser.add_argument("--platform", default=os.environ.get("VL_API", "http://localhost:8000"),
                        help="Platform base URL (default $VL_API or http://localhost:8000)")
    parser.add_argument("--api-key", default=os.environ.get("PLATFORM_API_KEY", ""),
                        help="Platform admin key (default $PLATFORM_API_KEY)")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
                        help="rendered dataset v1 audio (python -m src.evals.generate_audio)")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds of silence between utterances")
    parser.add_argument("--wait", type=float, default=15.0,
                        help="seconds after the end of speech before a reply counts as missing")
    parser.add_argument("--event-wait", type=float, default=15.0,
                        help="seconds to wait after a call for transcripts and runtime events")
    parser.add_argument("--confirm-billable", action="store_true",
                        help="required: scenarios create sessions that call paid provider APIs")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, then exit 0")
    return parser


def plan(scenarios: list[Scenario], audio_dir: Path, env: Mapping[str, str]) -> tuple[str, list[Scenario]]:
    """Plan text and the scenarios that can run here."""

    lines, runnable = ["Golden-path plan:"], []
    for s in scenarios:
        reason = skip_reason(s, audio_dir, env)
        flags = f"{s.mode}, {'enforced' if s.enforce else 'report-only'}, {len(s.audio)} clip(s)"
        if reason:
            lines.append(f"  SKIP {s.id} ({flags}): {reason}")
        else:
            runnable.append(s)
            lines.append(f"  RUN  {s.id} ({flags})")
    lines.append(f"{len(runnable)} to run, {len(scenarios) - len(runnable)} skipped")
    return "\n".join(lines), runnable


def skipped_result(scenario: Scenario, reason: str) -> ScenarioResult:
    return ScenarioResult(scenario.id, scenario.mode, SKIPPED, scenario.enforce, reason)


async def run_all(scenarios: list[Scenario], platform: GoldenPlatform, driver: Driver, audio_dir: Path,
                  settings: Settings, run_id: str, env: Mapping[str, str]) -> list[ScenarioResult]:
    results = []
    for scenario in scenarios:
        print(f"[{scenario.id}] running", flush=True)
        result = await run_scenario(scenario, platform, driver, audio_dir, settings, run_id, env)
        print(f"[{scenario.id}] {result.status}", flush=True)
        results.append(result)
    return results


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None,
         driver: Driver | None = None, transport: httpx.AsyncBaseTransport | None = None,
         webhook_transport: httpx.BaseTransport | None = None) -> int:
    """``driver`` and the transports exist so tests can run the whole CLI offline."""

    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    try:
        scenarios = select(load_scenarios(args.scenarios),
                           [i.strip() for value in args.only for i in value.split(",") if i.strip()])
    except (ScenarioError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    known = {item["id"] for item in load_manifest(DEFAULT_MANIFEST)}
    typos = [(s.id, ref) for s in scenarios for ref in unknown_dataset_ids(s, known)]
    if typos:
        print("error: unknown dataset ids: " + ", ".join(f"{s}:{r}" for s, r in typos), file=sys.stderr)
        return 2

    audio_dir = args.audio_dir.expanduser()
    text, runnable = plan(scenarios, audio_dir, env)
    print(text)
    if args.dry_run:
        return 0
    if not args.confirm_billable:
        print("Refusing to run without --confirm-billable (sessions call paid APIs). "
              "Use --dry-run to only see the plan.", file=sys.stderr)
        return 2
    if driver is None and runnable:
        try:
            import livekit.rtc  # noqa: F401, PLC0415
        except ImportError:
            print("error: the livekit package is required for live runs (use ~/.venvs/voice-labs/bin/python)",
                  file=sys.stderr)
            return 2

    started = datetime.now(timezone.utc)
    meta: dict[str, Any] = {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ") + "-golden",
        "date": started.isoformat(timespec="seconds"),
        "platform": args.platform,
        "audio": "dataset v1 synthetic voices unless a scenario uses human_v1 recordings",
        "machine": machine_description(),
    }
    settings = Settings(pause=args.pause, wait=args.wait, event_wait=args.event_wait)
    runnable_ids = {s.id for s in runnable}

    async def go() -> list[ScenarioResult]:
        client = httpx.AsyncClient(
            base_url=args.platform.rstrip("/"), timeout=30.0, transport=transport,
            headers={"authorization": f"Bearer {args.api_key}"} if args.api_key else {})
        platform = GoldenPlatform(args.platform, args.api_key, client=client)
        try:
            if runnable:
                await platform.health()
            ran = await run_all(runnable, platform, driver or LivekitDriver(), audio_dir, settings,
                                meta["run_id"], env)
        finally:
            await client.aclose()
        by_id = {r.id: r for r in ran}
        return [by_id[s.id] if s.id in runnable_ids else skipped_result(s, skip_reason(s, audio_dir, env) or "")
                for s in scenarios]

    try:
        results = asyncio.run(go())
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"error: Platform unreachable or refused the request: {exc}", file=sys.stderr)
        return 2
    report_dir = (args.report_dir or DEFAULT_REPORT_ROOT / meta["run_id"]).expanduser()
    json_path, md_path = write_report(results, meta, report_dir)
    print(render_markdown(results, meta))
    print(f"wrote {md_path} and {json_path}")
    print(notify_discord(results, meta, env, webhook_transport))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())

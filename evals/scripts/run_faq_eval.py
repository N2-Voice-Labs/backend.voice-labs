#!/usr/bin/env python3
"""Run text-only FAQ checks against the Platform demo chat endpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

# Allow direct execution as ``python scripts/run_faq_eval.py`` from evals/.
EVALS_ROOT = Path(__file__).resolve().parents[1]
if str(EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALS_ROOT))

from src.evals.faq_eval import score_item

DEFAULT_DATASET = Path("evals/datasets/inha_admissions_faq.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--platform-url", default=os.environ.get("VL_API", "http://localhost:8000"))
    parser.add_argument("--agent-key", default="inha-admissions")
    parser.add_argument("--out", type=Path, default=Path("faq_eval_results.json"))
    parser.add_argument("--delay-s", type=float, default=1.0)
    return parser


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Read a YAML list, or a mapping containing one list value."""
    if not path.is_file() and path == DEFAULT_DATASET:
        path = Path(__file__).resolve().parents[1] / "datasets" / path.name
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if isinstance(data, dict):
        data = next((value for value in data.values() if isinstance(value, list)), None)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("dataset must be a YAML list of FAQ items (or a mapping containing one)")
    required = ("id", "question", "expected_answer", "must_include", "source_file")
    for index, item in enumerate(data, 1):
        missing = [field for field in required if field not in item]
        if missing:
            raise ValueError(f"dataset item {index} is missing: {', '.join(missing)}")
        if not isinstance(item["must_include"], list):
            raise ValueError(f"dataset item {item['id']!r}: must_include must be a list")
        item.setdefault("answerable", True)
    return data


def run_eval(items: list[dict[str, Any]], platform_url: str, agent_key: str,
             delay_s: float = 1.0, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    owns_client = client is None
    client = client or httpx.Client(base_url=platform_url.rstrip("/"), timeout=60.0)
    results: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(items):
            if index and delay_s > 0:
                time.sleep(delay_s)
            started = time.perf_counter()
            reply, handoff, error = "", False, None
            try:
                response = client.post(f"/demo/agents/{agent_key}/chat", json={"message": item["question"]})
                response.raise_for_status()
                body = response.json()
                reply = str(body.get("reply", ""))
                handoff = bool(body.get("handoff_suggested", False))
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                error = str(exc)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            scored = score_item(item, reply, handoff)
            scored.update(latency_ms=latency_ms, error=error)
            results.append(scored)
            status = "PASS" if scored["passed"] else ("REVIEW" if scored["needs_review"] else "FAIL")
            print(f"{item['id']:<24} {status:<7} {latency_ms:>8.1f} ms  {reply[:72].replace(chr(10), ' ')}")
    finally:
        if owns_client:
            client.close()
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        items = load_dataset(args.dataset)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 0
    print(f"{'ID':<24} {'RESULT':<7} {'LATENCY':>12}  REPLY")
    results = run_eval(items, args.platform_url, args.agent_key, args.delay_s)
    latencies = [row["latency_ms"] for row in results]
    passed = sum(row["passed"] for row in results)
    median = statistics.median(latencies) if latencies else 0
    print(f"passed {passed}/{len(items)}; median latency {median:.1f} ms")
    report = {"platform_url": args.platform_url, "agent_key": args.agent_key,
              "passed": passed, "total": len(items), "median_latency_ms": median, "items": results}
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    except OSError as exc:
        print(f"error writing report: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

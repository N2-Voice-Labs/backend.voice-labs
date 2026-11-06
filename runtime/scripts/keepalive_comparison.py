#!/usr/bin/env python
"""Does a longer HTTP keep-alive make the Runtime's Google translation faster? Costs no ElevenLabs credits.

    cd runtime && python scripts/keepalive_comparison.py [--blocks 10] [--per-block 10] [--expiries 5,30] [--out bench/keepalive]

Uses the Runtime's own ``GoogleTranslateTranslator`` with an ``httpx.AsyncClient`` per block,
identical requests, and a seeded 6-8 s idle gap before each one (the same gaps for every
configuration). Configurations alternate block by block (A B A B ...) so drift in the network or
in Google affects both. Each block starts like a call: new client, ``warm_up()``. The httpx trace
hook records whether every request opened a TCP connection, did a TLS handshake or reused one.

Needs GOOGLE_TRANSLATE_API_KEY in the environment (never printed). Sends tiny translation
requests only (a short sentence), a few hundred in all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics as stats
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from runtime.interpreter import GoogleTranslateTranslator  # noqa: E402

MESSAGES = [
    {"role": "system", "content": "Translate from Korean (ko) to English (en)."},
    {"role": "user", "content": "공항까지 가는 가장 빠른 길이 어디인가요?"},
]


class TracingClient:
    """An httpx client that records the connection events of every request it sends."""

    def __init__(self, keepalive_expiry: float | None) -> None:
        limits = httpx.Limits(keepalive_expiry=keepalive_expiry) if keepalive_expiry is not None else httpx.Limits()
        self.inner = httpx.AsyncClient(timeout=10.0, limits=limits)
        self.events: list[str] = []

    async def _trace(self, name: str, _info: object) -> None:
        self.events.append(name)

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        self.events = []
        kwargs["extensions"] = {"trace": self._trace}
        return await self.inner.post(*args, **kwargs)  # type: ignore[arg-type]

    async def get(self, *args: object, **kwargs: object) -> httpx.Response:
        return await self.inner.get(*args, **kwargs)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        await self.inner.aclose()


async def block(expiry: float | None, gaps: list[float]) -> list[dict]:
    client = TracingClient(expiry)
    translator = GoogleTranslateTranslator(api_key=os.environ["GOOGLE_TRANSLATE_API_KEY"], client=client, timeout=10.0)
    await translator.warm_up()
    rows = []
    for gap in gaps:
        await asyncio.sleep(gap)
        started = time.monotonic()
        error = None
        try:
            await translator(MESSAGES)
        except Exception as exc:  # recorded, not hidden
            error = type(exc).__name__
        ms = (time.monotonic() - started) * 1000
        rows.append({
            "ms": round(ms, 1), "gap_s": round(gap, 2), "error": error,
            "new_tcp": any(e.startswith("connection.connect_tcp.started") for e in client.events),
            "tls": any(e.startswith("connection.start_tls.started") for e in client.events),
        })
    await client.aclose()
    return rows


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


async def main(args: argparse.Namespace) -> None:
    expiries = [float(x) for x in args.expiries.split(",")]
    rng = random.Random(args.seed)
    plan = [[rng.uniform(6, 8) for _ in range(args.per_block)] for _ in range(args.blocks)]
    results: dict[float, list[dict]] = {e: [] for e in expiries}
    for index, gaps in enumerate(plan):
        order = expiries if index % 2 == 0 else expiries[::-1]  # A B, B A, ...
        for expiry in order:
            results[expiry] += await block(expiry, gaps)
        print(f"block {index + 1}/{args.blocks} done", flush=True)
    print(f"\n{'expiry':>8}{'n':>5}{'P50':>7}{'P95':>7}{'P99':>7}{'mean':>7}{'new TCP':>10}{'reused':>8}{'errors':>8}")
    for expiry, rows in results.items():
        ok = [r["ms"] for r in rows if not r["error"]]
        new = sum(r["new_tcp"] for r in rows)
        print(f"{expiry:>6.0f} s{len(rows):>5}{pct(ok, 50):>7.0f}{pct(ok, 95):>7.0f}{pct(ok, 99):>7.0f}{stats.mean(ok):>7.0f}"
              f"{new:>8} ({new / len(rows):.0%}){len(rows) - new:>6}{sum(bool(r['error']) for r in rows):>8}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "keepalive_comparison.json").write_text(json.dumps({str(k): v for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--per-block", type=int, default=10)
    parser.add_argument("--expiries", default="5,30")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    parsed = parser.parse_args()
    if not os.environ.get("GOOGLE_TRANSLATE_API_KEY"):
        sys.exit("missing environment: GOOGLE_TRANSLATE_API_KEY")
    asyncio.run(main(parsed))

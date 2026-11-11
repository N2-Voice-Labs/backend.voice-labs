#!/usr/bin/env python
"""Does a longer HTTP keep-alive make the Runtime's Google translation faster? Costs no ElevenLabs credits.

    cd runtime && python scripts/keepalive_comparison.py [--blocks 10] [--per-block 10] [--expiries 5,30] [--lanes 1] [--out bench/keepalive]

Uses the Runtime's own ``GoogleTranslateTranslator`` with an ``httpx.AsyncClient`` per block,
identical requests, and a seeded 6-8 s idle gap before each one (the same gaps for every
configuration). Configurations alternate block by block (A B A B ...) so drift in the network or
in Google affects both. Each block starts like a call: new client, ``warm_up()``. The httpx trace
hook records, per request: whether it opened a TCP connection or reused one, the connection's age
and the idle time since the client's previous response, and where the time went (waiting for a
connection, sending, waiting for the first response byte). ``--lanes N`` runs N independent
alternating sequences at once (different seeds) to collect more samples in less time; the requests
are tiny, so they do not disturb one another.

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
        self.events: dict[str, float] = {}
        self.connected_at: float | None = None  # when the current connection was opened
        self.last_response_at: float | None = None

    async def _trace(self, name: str, _info: object) -> None:
        self.events.setdefault(name, time.monotonic())

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        self.events = {}
        kwargs["extensions"] = {"trace": self._trace}
        try:
            return await self.inner.post(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            self.last_response_at = time.monotonic()

    async def get(self, *args: object, **kwargs: object) -> httpx.Response:
        try:
            return await self.inner.get(*args, **kwargs)  # type: ignore[arg-type]
        finally:
            now = time.monotonic()
            self.connected_at = self.connected_at or now  # warm-up opened the first connection
            self.last_response_at = now

    async def aclose(self) -> None:
        await self.inner.aclose()


def phases(events: dict[str, float], started: float, ended: float) -> dict[str, float]:
    """Milliseconds spent before the request was sent, waiting for the first response byte, and receiving."""
    sent = events.get("http11.send_request_body.complete") or events.get("http11.send_request_headers.complete")
    first = events.get("http11.receive_response_headers.complete")
    out = {}
    if sent is not None:
        out["to_send_ms"] = round((sent - started) * 1000, 1)  # pool wait + connect + TLS + write
    if sent is not None and first is not None:
        out["server_wait_ms"] = round((first - sent) * 1000, 1)  # Google's time plus the network round trip
    if first is not None:
        out["after_headers_ms"] = round((ended - first) * 1000, 1)
    return out


async def block(expiry: float | None, gaps: list[float]) -> list[dict]:
    client = TracingClient(expiry)
    translator = GoogleTranslateTranslator(api_key=os.environ["GOOGLE_TRANSLATE_API_KEY"], client=client, timeout=10.0)
    await translator.warm_up()
    rows = []
    for gap in gaps:
        await asyncio.sleep(gap)
        idle_s = time.monotonic() - client.last_response_at if client.last_response_at else None
        started = time.monotonic()
        error = None
        try:
            await translator(MESSAGES)
        except Exception as exc:  # recorded, not hidden
            error = type(exc).__name__
        ended = time.monotonic()
        new_tcp = "connection.connect_tcp.started" in client.events
        if new_tcp:
            age_s = 0.0
            client.connected_at = started
        else:
            age_s = started - client.connected_at if client.connected_at else None
        rows.append({
            "ms": round((ended - started) * 1000, 1), "gap_s": round(gap, 2), "error": error,
            "new_tcp": new_tcp, "tls": "connection.start_tls.started" in client.events,
            "idle_s": round(idle_s, 2) if idle_s is not None else None,
            "conn_age_s": round(age_s, 1) if age_s is not None else None,
            **phases(client.events, started, ended),
        })
    await client.aclose()
    return rows


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


async def lane(index: int, args: argparse.Namespace, expiries: list[float]) -> dict[float, list[dict]]:
    rng = random.Random(args.seed + index)
    results: dict[float, list[dict]] = {e: [] for e in expiries}
    for number in range(args.blocks):
        gaps = [rng.uniform(6, 8) for _ in range(args.per_block)]
        order = expiries if (number + index) % 2 == 0 else expiries[::-1]  # A B, B A, ...
        for expiry in order:
            results[expiry] += await block(expiry, gaps)
        print(f"lane {index} block {number + 1}/{args.blocks} done", flush=True)
    return results


async def main(args: argparse.Namespace) -> None:
    expiries = [float(x) for x in args.expiries.split(",")]
    lanes = await asyncio.gather(*(lane(i, args, expiries) for i in range(args.lanes)))
    results = {e: [row for result in lanes for row in result[e]] for e in expiries}
    print(f"\n{'expiry':>8}{'n':>5}{'P50':>7}{'P90':>7}{'P95':>7}{'P99':>7}{'max':>7}{'mean':>7}{'new TCP':>10}{'errors':>8}")
    for expiry, rows in results.items():
        ok = [r["ms"] for r in rows if not r["error"]]
        new = sum(r["new_tcp"] for r in rows)
        print(f"{expiry:>6.0f} s{len(rows):>5}{pct(ok, 50):>7.0f}{pct(ok, 90):>7.0f}{pct(ok, 95):>7.0f}{pct(ok, 99):>7.0f}"
              f"{max(ok):>7.0f}{stats.mean(ok):>7.0f}{new:>8} ({new / len(rows):.0%}){sum(bool(r['error']) for r in rows):>6}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / args.name).write_text(json.dumps({str(k): v for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--per-block", type=int, default=10)
    parser.add_argument("--expiries", default="5,30")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--name", default="keepalive_comparison.json")
    parser.add_argument("--out", default=None)
    parsed = parser.parse_args()
    if not os.environ.get("GOOGLE_TRANSLATE_API_KEY"):
        sys.exit("missing environment: GOOGLE_TRANSLATE_API_KEY")
    asyncio.run(main(parsed))

#!/usr/bin/env python
"""Fault-injection stand-in for Google Cloud Translation (TEST CONFIGURATION ONLY).

Forwards to the real API and fails or delays on command, so the Gemini fallback and the
deadline can be exercised against a live call. Normal benchmarks must NOT use it.

    uvicorn scripts.fake_google:app --port 9000      # from runtime/, needs GOOGLE_TRANSLATE_API_KEY
    curl -X POST localhost:9000/mode/slow1000        # ok | 429 | 500 | 503 | hang | slow<ms>
    GOOGLE_TRANSLATE_BASE_URL=http://localhost:9000/language/translate/v2   # on the Runtime

``slow<ms>`` delays the answer by that many milliseconds; ``GET /state`` shows request counts.
"""

import asyncio
import os

import httpx
from fastapi import FastAPI, Request, Response

REAL = "https://translation.googleapis.com/language/translate/v2"
app = FastAPI()
state = {"mode": "ok", "hits": 0, "translate_hits": 0}
client = httpx.AsyncClient(timeout=20)


def _key() -> dict[str, str]:
    return {"x-goog-api-key": os.environ["GOOGLE_TRANSLATE_API_KEY"]}


@app.post("/mode/{mode}")
async def set_mode(mode: str):
    state["mode"] = mode
    return state


@app.get("/state")
async def get_state():
    return state


@app.api_route("/language/translate/v2{rest:path}", methods=["GET", "POST"])
async def proxy(rest: str, request: Request):
    state["hits"] += 1
    if request.method == "GET":  # warm-up and health probe: always the real answer
        r = await client.get(REAL + rest, params=dict(request.query_params), headers=_key())
        return Response(r.content, r.status_code, media_type="application/json")
    state["translate_hits"] += 1
    mode = state["mode"]
    if mode in ("429", "500", "503"):
        return Response('{"error":{"message":"injected"}}', int(mode), media_type="application/json")
    if mode == "hang":
        await asyncio.sleep(60)
    if mode.startswith("slow"):
        await asyncio.sleep(float(mode[4:]) / 1000)  # milliseconds
    r = await client.post(REAL, json=await request.json(), headers=_key())
    return Response(r.content, r.status_code, media_type="application/json")

"""Does the Runtime's Google translator pay for a new connection when sentences are seconds apart?

Free of ElevenLabs credits; sends a few dozen small Google Translation requests (needs
GOOGLE_TRANSLATE_API_KEY). Compares httpx's default client (keep-alive expires after 5 s idle)
with a client that keeps connections for 60 s, at call-like gaps between sentences.

    cd runtime && python scripts/idle_connection_probe.py
"""

import asyncio, os, time, statistics as S, httpx, sys
sys.path.insert(0, ".")
from src.runtime.interpreter import GoogleTranslateTranslator
MSG = [{"role":"system","content":"Translate from Korean (ko) to English (en)."},{"role":"user","content":"공항까지 가는 가장 빠른 길이 어디인가요?"}]
async def run(label, gap, n, client=None):
    t=GoogleTranslateTranslator(api_key=os.environ["GOOGLE_TRANSLATE_API_KEY"], client=client)
    await t.warm_up(); lat=[]
    for _ in range(n):
        await asyncio.sleep(gap); s=time.monotonic(); await t(MSG); lat.append((time.monotonic()-s)*1000)
    print(f"{label:46} gap {gap}s  P50 {S.median(lat):.0f}  min {min(lat):.0f}  max {max(lat):.0f}")
    if client is not None: await client.aclose()
async def main():
    for i in range(2):
        await run("Runtime translator, default client", 6, 8)
        await run("Runtime translator, keepalive_expiry=60", 6, 8, httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(keepalive_expiry=60.0)))
    await run("Runtime translator, default client, gap 2 s", 2, 8)
if __name__ == "__main__":
    asyncio.run(main())

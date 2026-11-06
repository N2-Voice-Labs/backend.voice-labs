"""Do translator HTTP connections outlive a call? Free: talks to a local server it starts itself.

    cd runtime && python scripts/connection_leak_check.py

Simulates 40 calls (a new translator each, as the Runtime builds one per call) and counts open
sockets, first without and then with ``close_translator`` at the end of the call.
"""
import asyncio, os, sys, json
sys.path.insert(0, ".")
from src.runtime.interpreter import GoogleTranslateTranslator, close_translator
BODY = json.dumps({"data": {"translations": [{"translatedText": "ok"}]}}).encode()
async def handle(r, w):
    try:
        while True:
            head = await r.readuntil(b"\r\n\r\n")
            n = int([l.split(b":")[1] for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")][0]) if b"ontent-" in head else 0
            if n: await r.readexactly(n)
            w.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(BODY) + BODY); await w.drain()
    except Exception: pass
    finally: w.close()
def sockets():
    n = 0
    for f in os.listdir("/proc/self/fd"):
        try: n += "socket" in os.readlink(f"/proc/self/fd/{f}")
        except OSError: pass
    return n
MSG = [{"role":"system","content":"Translate from Korean (ko) to English (en)."},{"role":"user","content":"x"}]
async def calls(n, close):
    for _ in range(n):
        t = GoogleTranslateTranslator(api_key="k", url="http://127.0.0.1:%d" % PORT)
        await t.warm_up(); await t(MSG)
        if close: await close_translator(t)
async def main():
    global PORT
    srv = await asyncio.start_server(handle, "127.0.0.1", 0); PORT = srv.sockets[0].getsockname()[1]
    base = sockets()  # includes the server's listening socket
    await calls(40, close=False); await asyncio.sleep(0.2)
    print("sockets held after 40 calls, no close:", (sockets() - base) // 2, "client connections (client+server ends)")
    import gc; gc.collect()
    print("after gc.collect():", (sockets() - base) // 2)
    await calls(40, close=True); await asyncio.sleep(0.2)
    print("additional after 40 calls with close:", (sockets() - base) // 2)
if __name__ == "__main__":
    asyncio.run(main())

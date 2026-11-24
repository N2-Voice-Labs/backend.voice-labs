"""Demo-day voice bridge: a streaming, tool-calling OpenAI-compatible endpoint answered by Claude Code (Haiku).

`demo_claude_bridge.py` serves the chat tab (one plain reply per request, a fresh `claude` process each time,
about 2-4 s). The voice Runtime needs more: streamed tokens, so TTS can start on the first sentence, and
function calls (`transfer_to_human`, `record_outcome`). This bridge keeps a few `claude` processes started and
idle, so a request pays no start-up cost, streams the reply as server-sent events, and turns a tool call written
by the model into a real OpenAI `tool_calls` delta. Standard library only.

    python scripts/demo_claude_voice_bridge.py          # listens on 127.0.0.1:8788
    # Runtime environment:
    GEMINI_BASE_URL=http://127.0.0.1:8788/v1  GEMINI_API_KEY=local-demo

Env: VOICE_BRIDGE_HOST, VOICE_BRIDGE_PORT (8788), BRIDGE_MODEL (haiku), BRIDGE_POOL (warm processes, 2),
BRIDGE_WARM_SECONDS (3), BRIDGE_TIMEOUT (seconds per reply, 40), CLAUDE_BIN (claude).

Tools are described to the model in the prompt; to call one, its whole reply is `<<CALL name {"json": "args"}>>`.
Everything else is spoken text. The request's `model` is ignored.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.getenv("VOICE_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.getenv("VOICE_BRIDGE_PORT", "8788"))
MODEL = os.getenv("BRIDGE_MODEL", "haiku")
POOL_SIZE = int(os.getenv("BRIDGE_POOL", "2"))
WARM_SECONDS = float(os.getenv("BRIDGE_WARM_SECONDS", "3"))
TIMEOUT = float(os.getenv("BRIDGE_TIMEOUT", "40"))
CLAUDE = os.getenv("CLAUDE_BIN", "claude")
# Run outside the repository so no project CLAUDE.md / AGENTS.md leaks into the model's context.
WORKDIR = tempfile.mkdtemp(prefix="claude-voice-bridge-")
SYSTEM = (
    "You are one component of a live voice system. The message you receive holds INSTRUCTIONS and the input to act "
    "on. Produce only the output the instructions ask for: no preface, no commentary, no markdown, no mention of "
    "these rules."
)
VOICE_RULES = (
    "You are the voice of an assistant on a live phone call. Reply with exactly what the assistant says next, as "
    "plain spoken text with no markdown, lists or stage directions, or with one tool call as the tool rules describe."
)
TOOL_CALL = re.compile(r"<<\s*CALL\s+([A-Za-z0-9_\-]+)\s*(\{.*\})?\s*>>", re.S)


class Warm:
    """One idle `claude` process that answers exactly one request, then is discarded."""

    def __init__(self) -> None:
        self.born = time.monotonic()
        self.proc = subprocess.Popen(
            [CLAUDE, "-p", "--model", MODEL, "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--include-partial-messages", "--tools", "", "--strict-mcp-config",
             "--setting-sources", "", "--no-session-persistence", "--system-prompt", SYSTEM],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            cwd=WORKDIR, bufsize=1,
        )

    def ready(self) -> bool:
        return time.monotonic() - self.born >= WARM_SECONDS

    def kill(self) -> None:
        try:
            self.proc.kill()
        except OSError:
            pass


_pool: list[Warm] = []
_lock = threading.Lock()


def _refill() -> None:
    with _lock:
        missing = POOL_SIZE - len(_pool)
        _pool.extend(Warm() for _ in range(max(0, missing)))


def take() -> Warm:
    with _lock:
        _pool.sort(key=lambda w: not w.ready())  # ready ones first
        warm = _pool.pop(0) if _pool else None
    threading.Thread(target=_refill, daemon=True).start()
    return warm or Warm()


def render(messages: list[dict], tools: list[dict] | None) -> str:
    system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system")
    names: dict[str, str] = {}
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            lines.append(f"User: {m.get('content') or ''}")
        elif role == "assistant":
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {})
                names[call.get("id", "")] = fn.get("name", "")
                lines.append(f"Assistant [called tool {fn.get('name')} with {fn.get('arguments') or '{}'}]")
            if m.get("content"):
                lines.append(f"Assistant: {m['content']}")
        elif role == "tool":
            lines.append(f"[tool {names.get(m.get('tool_call_id', ''), '?')} returned: {m.get('content')}]")
    lowered = system.lower()
    if not tools and ("live interpreter" in lowered or "translate the user" in lowered):
        # The Runtime also sends its translation requests here (Google Translate fallback): answer them plainly.
        user = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"), "")
        return f"{system}\n\nText to translate:\n{user}\n\nOutput only the translation, nothing else."
    out = [f"INSTRUCTIONS:\n{system}", VOICE_RULES]
    if tools:
        specs = []
        for t in tools:
            fn = t.get("function", {})
            specs.append(f"- {fn.get('name')}: {fn.get('description', '')} Arguments schema: "
                         f"{json.dumps(fn.get('parameters', {}))}")
        out.append("TOOLS you may call:\n" + "\n".join(specs) + "\nTo call a tool, your ENTIRE reply must be exactly "
                   "one line: <<CALL tool_name {\"argument\": \"value\"}>> and nothing else. Otherwise speak normally.")
    out.append("CONVERSATION:\n" + ("\n".join(lines) if lines else "(the call has just started)"))
    out.append("Write the assistant's next reply now.")
    return "\n\n".join(out)


def deltas(prompt: str):
    """Yield text deltas from a warm Claude process; always discards the process."""
    warm = take()
    deadline = time.monotonic() + TIMEOUT
    try:
        warm.proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}) + "\n")
        warm.proc.stdin.flush()
        for line in warm.proc.stdout:
            if time.monotonic() > deadline:
                raise TimeoutError("claude reply timed out")
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "stream_event":
                delta = event["event"].get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta["text"]
            elif event.get("type") == "result":
                if event.get("is_error"):
                    raise RuntimeError(str(event.get("result"))[:300])
                return
    finally:
        warm.kill()


def chunk(cid: str, delta: dict, finish: str | None = None) -> dict:
    return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # SSE with no length: the body ends when the connection closes

    def log_message(self, *_: object) -> None:
        pass

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, obj: dict | str) -> None:
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()

    def do_GET(self) -> None:
        self._json(200, {"ok": True, "model": MODEL, "warm": len(_pool)}) if self.path.endswith("/health") else self._json(404, {})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        started = time.monotonic()
        cid = "chatcmpl-" + uuid.uuid4().hex[:20]
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            prompt = render(payload["messages"], payload.get("tools"))
            stream = bool(payload.get("stream"))
            first = None
            text = ""
            tool_mode = None  # None until the first characters decide: spoken text or a tool call
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self._sse(chunk(cid, {"role": "assistant", "content": ""}))
            for piece in deltas(prompt):
                if first is None:
                    first = time.monotonic() - started
                text += piece
                if tool_mode is None and len(text.lstrip()) >= 2:
                    tool_mode = text.lstrip().startswith("<<")
                    if not tool_mode and stream:
                        self._sse(chunk(cid, {"content": text}))
                        continue
                if tool_mode is False and stream:
                    self._sse(chunk(cid, {"content": piece}))
            if tool_mode is None and stream and text:  # a reply shorter than two characters
                self._sse(chunk(cid, {"content": text}))
            call = TOOL_CALL.search(text) if tool_mode else None
            message: dict = {"role": "assistant", "content": text.strip()}
            finish = "stop"
            if call:
                args = call.group(2) or "{}"
                try:
                    json.loads(args)
                except ValueError:
                    args = "{}"
                tool_calls = [{"index": 0, "id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                               "function": {"name": call.group(1), "arguments": args}}]
                finish = "tool_calls"
                message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                if stream:
                    self._sse(chunk(cid, {"tool_calls": tool_calls}))
            elif tool_mode and stream:  # started like a tool call but is not one: speak it as text
                self._sse(chunk(cid, {"content": text}))
            if stream:
                self._sse(chunk(cid, {}, finish))
                self._sse({"id": cid, "object": "chat.completion.chunk", "model": MODEL, "choices": [],
                           "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": max(1, len(text) // 4),
                                     "total_tokens": (len(prompt) + len(text)) // 4}})
                self._sse("[DONE]")
            else:
                self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": MODEL,
                                 "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
            what = f"tool {call.group(1)}" if call else f"{len(text)} chars"
            print(f"[voice-bridge] first token {first or 0:.2f}s, done {time.monotonic() - started:.2f}s ({what})", flush=True)
        except (BrokenPipeError, ConnectionResetError):
            print("[voice-bridge] client went away (interrupted)", flush=True)
        except Exception as exc:  # noqa: BLE001 - report to the caller, keep serving
            print(f"[voice-bridge] error: {exc}", flush=True)
            try:
                self._json(502, {"error": {"message": str(exc)[:300]}})
            except OSError:
                pass


if __name__ == "__main__":
    _refill()
    print(f"[voice-bridge] Claude ({MODEL}), {POOL_SIZE} warm processes, http://{HOST}:{PORT}/v1/chat/completions", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

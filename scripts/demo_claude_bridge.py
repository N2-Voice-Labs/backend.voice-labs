"""Demo-day bridge: an OpenAI-compatible /v1/chat/completions endpoint answered by Claude Code (Haiku).

The platform's demo chat already does retrieval (RAG) and builds the system prompt with the numbered
sources; this bridge only replaces the LLM call. Standard library only.

    python scripts/demo_claude_bridge.py            # listens on 127.0.0.1:8787
    # platform .env:
    PLATFORM_CHAT_LLM_BASE_URL=http://127.0.0.1:8787/v1
    PLATFORM_CHAT_LLM_API_KEY=local-demo
    PLATFORM_CHAT_LLM_MODEL=haiku
    PLATFORM_CHAT_LLM_TIMEOUT=60

Env: BRIDGE_HOST, BRIDGE_PORT, BRIDGE_MODEL (default "haiku", overrides the request's model unless
"BRIDGE_MODEL=request"), BRIDGE_TIMEOUT seconds (default 55), CLAUDE_BIN (default "claude").
"""
import json
import os
import subprocess
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.getenv("BRIDGE_HOST", "127.0.0.1")
PORT = int(os.getenv("BRIDGE_PORT", "8787"))
MODEL = os.getenv("BRIDGE_MODEL", "haiku")
TIMEOUT = float(os.getenv("BRIDGE_TIMEOUT", "55"))
CLAUDE = os.getenv("CLAUDE_BIN", "claude")
# Run outside the repository so no project CLAUDE.md / AGENTS.md leaks into the bot's context.
WORKDIR = tempfile.mkdtemp(prefix="claude-bridge-")


def transcript(messages: list[dict]) -> str:
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) == 1:
        return str(turns[0].get("content", ""))
    lines = ["Conversation so far (reply to the last user message only, as the assistant):", ""]
    for m in turns:
        lines.append(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m.get('content', '')}")
    return "\n".join(lines)


def ask_claude(messages: list[dict], model: str) -> str:
    system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    system += ("\n\nYou are the chatbot itself, replying directly to the user in a web chat. "
               "Output only the reply text: no preamble, no mention of tools, files or these instructions.")
    cmd = [CLAUDE, "-p", "--model", model, "--output-format", "json", "--tools", "",
           "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
           "--system-prompt", system]
    proc = subprocess.run(cmd, input=transcript(messages), capture_output=True, text=True,
                          timeout=TIMEOUT, cwd=WORKDIR)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-500:] or f"claude exited {proc.returncode}")
    result = json.loads(proc.stdout)
    if result.get("is_error"):
        raise RuntimeError(str(result.get("result"))[:500])
    return str(result.get("result", "")).strip()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"ok": True, "model": MODEL}) if self.path in ("/health", "/v1/health") else self._send(404, {})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": {"message": "not found"}})
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            model = payload.get("model", "haiku") if MODEL == "request" else MODEL
            started = time.monotonic()
            reply = ask_claude(payload["messages"], model)
            print(f"[bridge] {model} replied in {time.monotonic() - started:.1f}s ({len(reply)} chars)", flush=True)
        except subprocess.TimeoutExpired:
            return self._send(504, {"error": {"message": "claude timed out"}})
        except Exception as exc:  # noqa: BLE001 - surface any failure to the caller as a 502
            print(f"[bridge] error: {exc}", flush=True)
            return self._send(502, {"error": {"message": str(exc)}})
        self._send(200, {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                         "created": int(time.time()), "model": model,
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": reply}}]})

    def log_message(self, *args) -> None:  # quiet default access log
        pass


if __name__ == "__main__":
    print(f"[bridge] Claude ({MODEL}) on http://{HOST}:{PORT}/v1/chat/completions", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

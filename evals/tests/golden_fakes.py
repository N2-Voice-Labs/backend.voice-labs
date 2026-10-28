"""Offline fakes for the golden-path tests: a Platform served by httpx.MockTransport and a driver."""

import json

import httpx

from src.evals.golden.runner import GoldenPlatform, Heard


class FakeStack:
    """Records calls; answers like the Platform. ``messages`` / ``state`` are what the run 'produced'."""

    def __init__(self, messages=(), state=None, create_session_status=201):
        self.calls = []
        self.messages = list(messages)
        self.state = state or {"session_id": "s1", "session_status": "ended", "events": []}
        self.create_session_status = create_session_status
        self.agent_bodies = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        body = json.loads(request.content) if request.content else None
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if (request.method, path) == ("POST", "/agents"):
            self.agent_bodies.append(body)
            return httpx.Response(201, json={"id": "agent-1", **body})
        if (request.method, path) == ("POST", "/voice/sessions"):
            if self.create_session_status >= 400:
                return httpx.Response(self.create_session_status, json={"detail": "runtime_unavailable"})
            return httpx.Response(201, json={"session_id": "s1", "conversation_id": "c1",
                                             "connection": {"url": "ws://x", "token": "t"}})
        if path == "/voice/sessions/s1/participants":
            return httpx.Response(201, json={"connection": {"url": "ws://x", "token": "g"}})
        if path == "/voice/sessions/s1/end":
            return httpx.Response(200, json={})
        if request.method == "DELETE" and path == "/agents/agent-1":
            return httpx.Response(200, json={})
        if path == "/conversations/c1/messages":
            return httpx.Response(200, json=self.messages)
        if path == "/conversations/c1/runtime-state":
            return httpx.Response(200, json=self.state)
        return httpx.Response(404, json={"detail": path})

    def platform(self) -> GoldenPlatform:
        client = httpx.AsyncClient(base_url="http://platform", transport=httpx.MockTransport(self.handler))
        return GoldenPlatform("http://platform", client=client)


class FakeDriver:
    def __init__(self, after_end_s=(0.8,), fail=None):
        self.after_end_s = list(after_end_s)
        self.fail = fail
        self.played = []

    async def play(self, scenario, connections, clips, settings):
        self.played.append(([c.ref for c in clips], [label for label, _ in connections]))
        if self.fail:
            raise self.fail
        return [Heard(c.ref, self.after_end_s[i % len(self.after_end_s)] is not None,
                      self.after_end_s[i % len(self.after_end_s)]) for i, c in enumerate(clips)]


def assistant(*texts):
    return [{"role": "assistant", "content": t} for t in texts]


async def no_sleep(_):
    return None

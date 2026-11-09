import os
from uuid import uuid4

import httpx
import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.platform import health
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import Conversation, VoiceSession


def entry(kind, name, status="ok", category=None, reason=None):
    return {
        "kind": kind, "name": name, "status": status, "category": category,
        "latency_ms": 12, "checked_at": "2026-01-01T00:00:00+00:00", "reason": reason,
    }


def report(*entries, overall=None):
    return {"overall": overall or "ok", "providers": list(entries)}


HEALTHY = [entry("stt", "deepgram"), entry("llm", "openai"), entry("tts", "cartesia"), entry("realtime", "openai")]


class FakeResponse:
    def __init__(self, body, status_code=200):
        self.body, self.status_code = body, status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "x", request=httpx.Request("GET", "http://runtime.test"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self.body


class FakeRuntime:
    """Replaces httpx.AsyncClient: serves /health/providers and accepts /sessions."""

    def __init__(self, state):
        self.state = state

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def get(self, url, **kwargs):
        self.state["health_calls"] += 1
        if self.state["health"] is None:
            raise httpx.ConnectError("down")
        return FakeResponse(self.state["health"])

    async def post(self, url, **kwargs):
        self.state["posts"].append(url)
        return FakeResponse({"status": "accepted"})


@pytest.fixture
def runtime(monkeypatch):
    health.reset()
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared")
    monkeypatch.delenv("PLATFORM_ALERT_WEBHOOK_URL", raising=False)
    state = {"health": report(*HEALTHY), "health_calls": 0, "posts": []}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeRuntime(state))
    yield state
    health.reset()


def counts():
    with SessionLocal() as db:
        return db.query(VoiceSession).count(), db.query(Conversation).count()


def make_agent(client, **extra):
    body = {"name": f"agent-{uuid4()}", **extra}
    response = client.post("/agents", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def start(client, agent):
    return client.post("/voice/sessions", json={"agent_id": agent["id"]})


# --- required_providers / assert_providers_ready ----------------------------------


def test_required_providers_defaults_and_aliases():
    assert health.required_providers("agent", None, None) == [
        ("stt", "deepgram"), ("llm", "openai"), ("tts", "cartesia")
    ]
    assert health.required_providers(
        "agent", None, {"llm": "openai-compatible", "tts": "elevenlabs"}
    ) == [("stt", "deepgram"), ("llm", "openai"), ("tts", "elevenlabs")]
    cascaded = {"engine": "cascaded"}
    assert len(health.required_providers("interpreter", cascaded, None)) == 3
    assert health.required_providers("interpreter", {"engine": "realtime"}, {"stt": "x"}) == [
        ("realtime", "openai")
    ]


def test_assert_providers_ready_only_blocks_down():
    required = [("stt", "deepgram")]
    health.assert_providers_ready(None, required)
    health.assert_providers_ready(report(entry("stt", "deepgram", "degraded", "timeout")), required)
    health.assert_providers_ready(report(entry("stt", "other", "down", "auth")), required)
    with pytest.raises(HTTPException) as caught:
        health.assert_providers_ready(
            report(entry("stt", "deepgram", "down", "auth", "credentials were rejected")), required
        )
    assert caught.value.status_code == 503
    assert caught.value.detail == {
        "code": "provider_unavailable", "provider": "deepgram", "kind": "stt",
        "category": "auth",
        "message": "Voice service unavailable: deepgram credentials were rejected",
    }


# --- create_session ---------------------------------------------------------------


def test_create_session_refuses_when_stt_down_and_creates_no_rows(runtime):
    runtime["health"] = report(
        entry("stt", "deepgram", "down", "quota", "out of credit or quota"),
        entry("llm", "openai"), entry("tts", "cartesia"), overall="down",
    )
    with TestClient(app) as client:
        agent = make_agent(client)
        before = counts()
        response = start(client, agent)
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "provider_unavailable"
        assert (detail["provider"], detail["kind"], detail["category"]) == ("deepgram", "stt", "quota")
        assert detail["message"].startswith("Voice service unavailable: ")
        assert counts() == before
        assert runtime["posts"] == []


def test_create_session_refuses_when_runtime_has_no_provider_keys(runtime):
    missing = dict(status="down", category="auth", reason="API key not configured")
    runtime["health"] = report(
        entry("stt", "deepgram", **missing), entry("llm", "openai", **missing),
        entry("realtime", "openai", **missing), entry("tts", "cartesia", **missing),
        overall="down",
    )
    with TestClient(app) as client:
        agent = make_agent(client)
        before = counts()
        response = start(client, agent)
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "provider_unavailable"
        assert (detail["provider"], detail["category"]) == ("deepgram", "auth")
        assert detail["message"] == "Voice service unavailable: deepgram API key not configured"
        assert counts() == before and runtime["posts"] == []
        overall = client.get("/health/providers").json()
        assert overall["overall"] == "down"


def test_create_session_proceeds_when_health_fetch_fails(runtime):
    runtime["health"] = None
    with TestClient(app) as client:
        response = start(client, make_agent(client))
        assert response.status_code == 201
        assert runtime["posts"]


def test_create_session_proceeds_when_degraded(runtime):
    runtime["health"] = report(
        entry("stt", "deepgram", "degraded", "timeout"),
        entry("llm", "openai"), entry("tts", "cartesia"), overall="degraded",
    )
    with TestClient(app) as client:
        assert start(client, make_agent(client)).status_code == 201


def test_realtime_interpreter_needs_realtime_not_cascaded(runtime):
    runtime["health"] = report(
        entry("stt", "deepgram", "down", "auth"), entry("realtime", "openai"), overall="down"
    )
    with TestClient(app) as client:
        agent = make_agent(
            client, mode="interpreter",
            interpreter={"languages": ["en", "ko"], "direction": "one_way", "engine": "realtime"},
        )
        assert start(client, agent).status_code == 201
        runtime["health"] = report(entry("realtime", "openai", "down", "quota"), overall="down")
        health.reset()
        refused = start(client, agent)
        assert refused.status_code == 503
        assert refused.json()["detail"]["kind"] == "realtime"


def test_health_is_cached_for_30_seconds(runtime):
    with TestClient(app) as client:
        agent = make_agent(client)
        start(client, agent)
        start(client, agent)
        assert runtime["health_calls"] == 1
        health._cache["at"] -= health.CACHE_TTL_SECONDS + 1
        start(client, agent)
        assert runtime["health_calls"] == 2


def test_runtime_handoff_failure_is_structured(runtime, monkeypatch):
    class Failing(FakeRuntime):
        async def post(self, url, **kwargs):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Failing(runtime))
    with TestClient(app) as client:
        response = start(client, make_agent(client))
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "runtime_unavailable" and detail["message"]


# --- GET /health/providers --------------------------------------------------------


def test_health_providers_endpoint(runtime):
    with TestClient(app) as client:
        body = client.get("/health/providers").json()
        assert body["overall"] == "ok" and body["database"] == "ok" and body["runtime"] == "ok"
        assert len(body["providers"]) == len(HEALTHY)
        runtime["health"] = None
        down = client.get("/health/providers?refresh=true").json()
        assert (down["runtime"], down["overall"]) == ("unreachable", "down")


def _org_key(role):
    from src.platform import auth
    from src.platform.models import DEFAULT_ORGANIZATION_ID, ApiKey
    from src.platform.organizations import ensure_default_org

    plaintext, prefix, digest = auth.generate_api_key()
    with SessionLocal() as db:
        ensure_default_org(db)
        db.add(ApiKey(id=str(uuid4()), organization_id=DEFAULT_ORGANIZATION_ID, name="h",
                      key_prefix=prefix, key_hash=digest, role=role))
        db.commit()
    return {"Authorization": f"Bearer {plaintext}"}


def test_forced_refresh_requires_admin_and_is_rate_limited(runtime, monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", "health-super")
    member, admin = _org_key("member"), _org_key("admin")
    with TestClient(app) as client:
        # Members only ever see the cached view, even with refresh=true.
        assert client.get("/health/providers", headers=member).status_code == 200
        assert runtime["health_calls"] == 1
        runtime["health"] = None
        body = client.get("/health/providers?refresh=true", headers=member).json()
        assert runtime["health_calls"] == 1 and body["runtime"] == "ok"
        # An admin forces a live probe ...
        down = client.get("/health/providers?refresh=true", headers=admin).json()
        assert runtime["health_calls"] == 2 and down["runtime"] == "unreachable"
        # ... but repeated forced refreshes within 10 s are served from cache.
        runtime["health"] = report(*HEALTHY)
        client.get("/health/providers?refresh=true", headers=admin)
        assert runtime["health_calls"] == 2
        # Once the window passes, a forced refresh probes again.
        health._last_forced_at -= health.FORCED_REFRESH_MIN_INTERVAL_SECONDS + 1
        again = client.get("/health/providers?refresh=true", headers=admin).json()
        assert runtime["health_calls"] == 3 and again["runtime"] == "ok"


def test_health_providers_runtime_not_configured(monkeypatch):
    health.reset()
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    with TestClient(app) as client:
        body = client.get("/health/providers").json()
    assert body["runtime"] == "not_configured" and body["providers"] == []


# --- alerting ---------------------------------------------------------------------


@pytest.fixture
def webhook(monkeypatch):
    health.reset()
    monkeypatch.setenv("PLATFORM_ALERT_WEBHOOK_URL", "https://hooks.example/alert")
    posted = []

    def handler(request):
        import json
        posted.append(json.loads(request.content))
        return httpx.Response(204)

    monkeypatch.setattr(health, "_transport", httpx.MockTransport(handler))
    yield posted
    health.reset()


async def test_alerts_on_change_and_debounce(webhook):
    ok, down = entry("stt", "deepgram"), entry("stt", "deepgram", "down", "auth", "credentials were rejected")
    await health.notify_changes([ok])
    assert webhook == []  # steady ok: nothing
    await health.notify_changes([down])
    assert len(webhook) == 1 and "stt/deepgram is down" in webhook[0]["content"]
    await health.notify_changes([down])
    assert len(webhook) == 1  # no change
    await health.notify_changes([ok])
    assert len(webhook) == 1  # recovery inside the 5 min window is debounced
    health._alert_sent_at[("stt", "deepgram")] -= health.ALERT_DEBOUNCE_SECONDS + 1
    await health.notify_changes([down])
    assert len(webhook) == 2
    health._alert_sent_at[("stt", "deepgram")] -= health.ALERT_DEBOUNCE_SECONDS + 1
    await health.notify_changes([ok])
    assert len(webhook) == 3 and "recovered" in webhook[2]["content"]
    assert "sk-" not in str(webhook)


async def test_debounce_is_per_provider(webhook):
    await health.notify_changes(
        [entry("stt", "deepgram", "down", "auth"), entry("tts", "cartesia", "degraded", "timeout")]
    )
    assert len(webhook) == 2


async def test_no_webhook_configured_is_noop(monkeypatch):
    health.reset()
    monkeypatch.delenv("PLATFORM_ALERT_WEBHOOK_URL", raising=False)

    def handler(request):
        raise AssertionError("no request expected")

    monkeypatch.setattr(health, "_transport", httpx.MockTransport(handler))
    await health.notify_changes([entry("stt", "deepgram", "down", "auth")])
    health.reset()


async def test_webhook_failure_is_swallowed(monkeypatch):
    health.reset()
    monkeypatch.setenv("PLATFORM_ALERT_WEBHOOK_URL", "https://hooks.example/alert")
    monkeypatch.setattr(health, "_transport", httpx.MockTransport(lambda r: httpx.Response(500)))
    await health.notify_changes([entry("stt", "deepgram", "down", "auth")])
    health.reset()

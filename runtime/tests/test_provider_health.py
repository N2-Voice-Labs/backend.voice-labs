import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from src.runtime import api, provider_health

KEYS = ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY", "ELEVENLABS_API_KEY")
SECRET = "sk-super-secret-value"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in (*KEYS, "OPENAI_BASE_URL", "OPENAI_HEALTH_MODEL"):
        monkeypatch.delenv(name, raising=False)
    provider_health.reset()
    provider_health._transport = None
    yield
    provider_health.reset()
    provider_health._transport = None


def run(coro):
    return asyncio.run(coro)


def transport(handler):
    return httpx.MockTransport(handler)


def probe(monkeypatch, env, handler):
    for name in env:
        monkeypatch.setenv(name, SECRET)
    return run(provider_health.probe_providers(transport=transport(handler)))


def by_key(statuses):
    return {(s.kind, s.name): s for s in statuses}


@pytest.mark.parametrize(
    ("code", "body", "status", "category"),
    [
        (200, "{}", "ok", None),
        (401, "{}", "down", "auth"),
        (403, "{}", "down", "auth"),
        (402, "{}", "down", "quota"),
        (429, '{"error":{"code":"insufficient_quota"}}', "down", "quota"),
        (429, "{}", "degraded", "unavailable"),
        (503, "{}", "degraded", "unavailable"),
    ],
)
@pytest.mark.parametrize(
    ("env", "kind", "name"),
    [
        ("DEEPGRAM_API_KEY", "stt", "deepgram"),
        ("OPENAI_API_KEY", "llm", "openai"),
        ("CARTESIA_API_KEY", "tts", "cartesia"),
        ("ELEVENLABS_API_KEY", "tts", "elevenlabs"),
    ],
)
def test_probe_classification(monkeypatch, env, kind, name, code, body, status, category):
    statuses = by_key(probe(monkeypatch, [env], lambda r: httpx.Response(code, text=body)))
    got = statuses[(kind, name)]
    assert (got.status, got.category) == (status, category)
    assert got.checked_at and got.latency_ms is not None
    assert SECRET not in repr(got.as_dict()) and "insufficient_quota" not in repr(got.as_dict())


@pytest.mark.parametrize("env", KEYS)
def test_timeout_is_degraded(monkeypatch, env):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    statuses = probe(monkeypatch, [env], handler)
    assert all(s.status == "degraded" and s.category == "timeout" for s in statuses)


def test_network_error_is_degraded_unavailable(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    (got,) = probe(monkeypatch, ["DEEPGRAM_API_KEY"], handler)
    assert (got.status, got.category) == ("degraded", "unavailable")


def test_only_configured_providers_are_probed_with_expected_requests(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    statuses = by_key(probe(monkeypatch, ["DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY"], handler))
    assert set(statuses) == {
        ("stt", "deepgram"), ("llm", "openai"), ("realtime", "openai"), ("tts", "cartesia")
    }
    urls = {r.url.host + r.url.path for r in seen}
    assert urls == {
        "api.deepgram.com/v1/projects",
        "api.openai.com/v1/models/gpt-4.1-mini",
        "api.cartesia.ai/voices",
    }
    cartesia = next(r for r in seen if r.url.host == "api.cartesia.ai")
    assert cartesia.url.params["limit"] == "1" and cartesia.headers["Cartesia-Version"]
    assert next(r for r in seen if r.url.host == "api.deepgram.com").headers["Authorization"] == f"Token {SECRET}"


def test_openai_probe_uses_configured_base_url_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example/v1/")
    monkeypatch.setenv("OPENAI_HEALTH_MODEL", "my-model")
    seen = []
    probe(monkeypatch, ["OPENAI_API_KEY"], lambda r: seen.append(str(r.url)) or httpx.Response(200))
    assert seen == ["https://llm.example/v1/models/my-model"]


def test_elevenlabs_exhausted_characters_is_quota(monkeypatch):
    body = {"character_count": 10000, "character_limit": 10000}
    (got,) = probe(monkeypatch, ["ELEVENLABS_API_KEY"], lambda r: httpx.Response(200, json=body))
    assert (got.status, got.category) == ("down", "quota")


def test_no_configured_providers_means_no_requests():
    def handler(request):
        raise AssertionError("no request expected")

    assert run(provider_health.probe_providers(transport=transport(handler))) == []


def test_cache_ttl(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", SECRET)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200)

    provider_health._transport = transport(handler)
    run(provider_health.get_provider_health())
    run(provider_health.get_provider_health())
    assert len(calls) == 1
    provider_health._cached_at -= provider_health.CACHE_TTL_SECONDS + 1
    run(provider_health.get_provider_health())
    assert len(calls) == 2
    run(provider_health.get_provider_health(force=True))
    assert len(calls) == 3


def test_record_provider_failure_updates_cache_immediately(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    provider_health._transport = transport(lambda r: httpx.Response(200))
    run(provider_health.get_provider_health())
    # A passive quota/auth failure is one tenant's view: degraded, never a global ``down``.
    provider_health.record_provider_failure("openai-compatible", "quota")
    cached = by_key(run(provider_health.get_provider_health()))
    assert (cached[("llm", "openai")].status, cached[("llm", "openai")].category) == (
        "degraded",
        "quota",
    )
    assert cached[("realtime", "openai")].status == "degraded"
    provider_health.record_provider_failure("openai", "unavailable")
    assert by_key(run(provider_health.get_provider_health()))[("llm", "openai")].status == "degraded"


def test_record_provider_failure_without_cache_creates_entry():
    provider_health.record_provider_failure("deepgram", "auth")
    (got,) = provider_health._cache.values()
    assert (got.kind, got.name, got.status, got.category) == ("stt", "deepgram", "degraded", "auth")
    provider_health.record_provider_failure("deepgram", "bogus")
    assert provider_health._cache[("stt", "deepgram")].category == "other"


@pytest.mark.parametrize("category", ["auth", "quota"])
def test_passive_auth_failure_is_degraded_and_schedules_a_reprobe(monkeypatch, category):
    """One tenant's inaccessible model (403) must not flip the provider down for everyone."""

    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200)

    provider_health._transport = transport(handler)

    async def scenario():
        await provider_health.get_provider_health()
        calls.clear()
        provider_health.record_provider_failure("openai", category)
        immediate = by_key(list(provider_health._cache.values()))
        assert immediate[("llm", "openai")].status == "degraded"
        assert immediate[("realtime", "openai")].status == "degraded"
        assert len(provider_health._reprobe_tasks) == 1
        await asyncio.gather(*provider_health._reprobe_tasks)
        return by_key(list(provider_health._cache.values()))

    after = run(scenario())
    assert len(calls) == 1  # one cheap probe, only for that provider
    # the real key probe is healthy: the provider is back to ok
    assert after[("llm", "openai")].status == "ok"
    assert after[("realtime", "openai")].status == "ok"


def test_only_the_probe_result_can_mark_a_provider_down(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    provider_health._transport = transport(lambda r: httpx.Response(401))

    async def scenario():
        provider_health.record_provider_failure("openai", "auth")
        assert provider_health._cache[("llm", "openai")].status == "degraded"
        await asyncio.gather(*provider_health._reprobe_tasks)
        return by_key(list(provider_health._cache.values()))

    after = run(scenario())
    assert (after[("llm", "openai")].status, after[("llm", "openai")].category) == ("down", "auth")


def test_passive_reprobe_is_debounced_per_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("DEEPGRAM_API_KEY", SECRET)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(200)

    provider_health._transport = transport(handler)

    async def scenario():
        for _ in range(5):
            provider_health.record_provider_failure("openai", "auth")
        provider_health.record_provider_failure("deepgram", "auth")
        await asyncio.gather(*provider_health._reprobe_tasks)
        # outside the debounce window the provider may be re-probed again
        provider_health._reprobe_last["openai"] -= provider_health.REPROBE_DEBOUNCE_SECONDS + 1
        provider_health.record_provider_failure("openai", "auth")
        await asyncio.gather(*provider_health._reprobe_tasks)

    run(scenario())
    assert sorted(calls) == ["api.deepgram.com", "api.openai.com", "api.openai.com"]


def test_health_providers_endpoint(monkeypatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "t0ken")
    monkeypatch.setenv("DEEPGRAM_API_KEY", SECRET)
    monkeypatch.setenv("CARTESIA_API_KEY", SECRET)
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(402 if request.url.host == "api.cartesia.ai" else 200)

    provider_health._transport = transport(handler)
    with TestClient(api.app) as client:
        assert client.get("/health/providers").status_code == 401
        assert client.get("/health/providers", headers={"X-Runtime-Token": "no"}).status_code == 401
        body = client.get("/health/providers", headers={"X-Runtime-Token": "t0ken"}).json()
        assert body["overall"] == "down"
        providers = {(p["kind"], p["name"]): p for p in body["providers"]}
        assert set(providers) == {("stt", "deepgram"), ("tts", "cartesia")}
        assert set(providers[("tts", "cartesia")]) == {
            "kind", "name", "status", "category", "latency_ms", "checked_at", "reason"
        }
        assert providers[("stt", "deepgram")]["status"] == "ok"
        assert providers[("tts", "cartesia")]["category"] == "quota"
        assert SECRET not in str(body)
        client.get("/health/providers", headers={"X-Runtime-Token": "t0ken"})
        assert len(calls) == 2  # cached
        client.get("/health/providers?refresh=true", headers={"X-Runtime-Token": "t0ken"})
        assert len(calls) == 4
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["active_sessions"] == 0 and health["max_sessions"] >= 1

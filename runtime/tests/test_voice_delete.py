import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from src.runtime import api, voice_cloning

TOKEN = {"X-Runtime-Token": "delete-token"}
KEY = "cartesia-secret-key"
VOICE_ID = "a1b2c3d4-0000-4000-8000-123456789abc"
REAL_CLIENT = httpx.AsyncClient


def install(monkeypatch, handler):
    """Route voice_cloning's httpx client through a MockTransport; records requests."""
    requests: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def factory(**kwargs):
        return REAL_CLIENT(transport=httpx.MockTransport(recording), **kwargs)

    monkeypatch.setattr("src.runtime.voice_cloning.httpx.AsyncClient", factory)
    return requests


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "delete-token")
    monkeypatch.setenv("CARTESIA_API_KEY", KEY)


def delete(provider="cartesia", voice_id=VOICE_ID, headers=TOKEN):
    with TestClient(api.app) as client:
        return client.delete(f"/voices/{provider}/{voice_id}", headers=headers)


def test_delete_success_returns_204_and_calls_cartesia(monkeypatch) -> None:
    requests = install(monkeypatch, lambda r: httpx.Response(200, json={"ok": True}))
    response = delete()
    assert response.status_code == 204
    assert response.content == b""
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "DELETE"
    assert str(request.url) == f"https://api.cartesia.ai/voices/{VOICE_ID}"
    assert request.headers["X-API-Key"] == KEY
    assert request.headers["Cartesia-Version"] == voice_cloning.CLONE_CARTESIA_VERSION


def test_delete_provider_404_is_success(monkeypatch) -> None:
    install(monkeypatch, lambda r: httpx.Response(404, text="voice not found"))
    response = delete()
    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(401, False), (403, False), (400, False), (500, True), (503, True), (429, True)],
)
def test_delete_provider_errors_map_to_502(monkeypatch, status_code, retryable) -> None:
    install(
        monkeypatch,
        lambda r: httpx.Response(status_code, text=f"upstream body {KEY}"),
    )
    response = delete()
    assert response.status_code == 502
    assert response.json() == {
        "code": "voice_provider_delete_failed",
        "retryable": retryable,
    }
    assert KEY not in response.text
    assert "upstream" not in response.text


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectError("down")])
def test_delete_timeout_or_transport_error_is_retryable(monkeypatch, error) -> None:
    def boom(request):
        raise error

    install(monkeypatch, boom)
    response = delete()
    assert response.status_code == 502
    assert response.json() == {"code": "voice_provider_delete_failed", "retryable": True}


def test_delete_unknown_provider(monkeypatch) -> None:
    requests = install(monkeypatch, lambda r: httpx.Response(200))
    response = delete(provider="elevenlabs")
    assert response.status_code == 404
    assert response.json() == {"code": "unknown_provider"}
    assert requests == []


@pytest.mark.parametrize("headers", [{}, {"X-Runtime-Token": "wrong"}])
def test_delete_requires_runtime_token(monkeypatch, headers) -> None:
    requests = install(monkeypatch, lambda r: httpx.Response(200))
    response = delete(headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_runtime_token"}
    assert requests == []


def test_delete_fails_closed_without_configured_token(monkeypatch) -> None:
    monkeypatch.delenv("RUNTIME_PLATFORM_TOKEN")
    requests = install(monkeypatch, lambda r: httpx.Response(200))
    assert delete().status_code == 401
    assert requests == []


@pytest.mark.parametrize(
    "voice_id",
    ["a" * 129, "bad%20id", "has.dot", "has%2Fslash", "..%2Fvoices", "semi;colon", "q%3Fx"],
)
def test_delete_rejects_bad_voice_id(monkeypatch, voice_id) -> None:
    requests = install(monkeypatch, lambda r: httpx.Response(200))
    response = delete(voice_id=voice_id)
    assert response.status_code in (404, 422)
    if response.status_code == 422:
        assert response.json() == {"detail": "invalid_provider_voice_id"}
    assert requests == []


def test_delete_voice_id_with_space_is_422(monkeypatch) -> None:
    requests = install(monkeypatch, lambda r: httpx.Response(200))
    response = delete(voice_id="bad%20id")
    assert response.status_code == 422
    assert requests == []


def test_delete_without_api_key_is_503(monkeypatch) -> None:
    monkeypatch.delenv("CARTESIA_API_KEY")
    requests = install(monkeypatch, lambda r: httpx.Response(200))
    response = delete()
    assert response.status_code == 503
    assert requests == []


def test_delete_voice_unknown_provider_raises() -> None:
    with pytest.raises(voice_cloning.UnknownVoiceProvider):
        asyncio.run(voice_cloning.delete_voice("nope", VOICE_ID, api_key=KEY))

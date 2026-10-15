import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import httpx
import pytest
from fastapi.testclient import TestClient

from src.platform.main import app


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"RIFFwav") -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://runtime.test/tts/preview")
            raise httpx.HTTPStatusError(
                "failed",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )


def install_client(monkeypatch, response=None, error=None) -> list[dict]:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            calls.append({"timeout": timeout})

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            calls.append({"url": url, **kwargs})
            if error is not None:
                raise error
            return response or FakeResponse()

    monkeypatch.setattr("src.platform.voices.httpx.AsyncClient", FakeClient)
    return calls


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test/")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")


def make_voice(client: TestClient, provider="cartesia", configuration=None) -> dict:
    provider_id = client.post(
        "/providers",
        json={
            "kind": "tts",
            "provider": provider,
            "configuration": configuration or {},
        },
    ).json()["id"]
    response = client.post(
        "/voices",
        json={
            "name": f"v-{uuid4().hex[:8]}",
            "tts_provider_id": provider_id,
            "provider_voice_id": "pv-9",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_preview_calls_runtime_and_returns_wav(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_voice(
            client, configuration={"model": "sonic-x", "language": "ru", "other": 1}
        )
        response = client.post(f"/voices/{voice['id']}/preview", json={"text": "Hi"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFwav"
    assert calls[0]["timeout"] == 20.0
    assert calls[1]["url"] == "http://runtime.test/tts/preview"
    assert calls[1]["headers"] == {"X-Runtime-Token": "shared-secret"}
    assert calls[1]["json"] == {
        "provider": "cartesia",
        "provider_voice_id": "pv-9",
        "text": "Hi",
        "model": "sonic-x",
        "language": "ru",
    }


def test_preview_not_found_and_validation() -> None:
    with TestClient(app) as client:
        missing = client.post(f"/voices/{uuid4()}/preview", json={"text": "Hi"})
        assert missing.status_code == 404
        assert missing.json() == {"detail": "voice_not_found"}
        voice = make_voice(client)
        for text in ("", "x" * 301):
            response = client.post(
                f"/voices/{voice['id']}/preview", json={"text": text}
            )
            assert response.status_code == 422


def test_preview_requires_ready_voice(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        pending = client.post(
            "/voices/design",
            json={"name": "d", "design_prompt": "A calm narrator"},
        ).json()
        disabled = make_voice(client)
        client.post(f"/voices/{disabled['id']}/status", json={"status": "disabled"})
        for voice in (pending, disabled):
            response = client.post(
                f"/voices/{voice['id']}/preview", json={"text": "Hi"}
            )
            assert response.status_code == 409
            assert response.json() == {"detail": "voice_not_ready"}
    assert calls == []


def test_preview_runtime_configuration_errors(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_voice(client)
        url = f"/voices/{voice['id']}/preview"
        monkeypatch.delenv("RUNTIME_API_URL")
        response = client.post(url, json={"text": "Hi"})
        assert (response.status_code, response.json()) == (
            503,
            {"detail": "runtime_not_configured"},
        )
        monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
        monkeypatch.delenv("PLATFORM_RUNTIME_TOKEN")
        response = client.post(url, json={"text": "Hi"})
        assert (response.status_code, response.json()) == (
            503,
            {"detail": "runtime_auth_not_configured"},
        )
    assert calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response": FakeResponse(502)},
        {"response": FakeResponse(200, b"")},
        {"error": httpx.ConnectError("down")},
        {"error": httpx.ReadTimeout("slow")},
    ],
)
def test_preview_runtime_failure_maps_to_502(monkeypatch, kwargs) -> None:
    install_client(monkeypatch, **kwargs)
    with TestClient(app) as client:
        voice = make_voice(client)
        response = client.post(f"/voices/{voice['id']}/preview", json={"text": "Hi"})
    assert response.status_code == 502
    assert response.json() == {"detail": "voice_preview_failed"}

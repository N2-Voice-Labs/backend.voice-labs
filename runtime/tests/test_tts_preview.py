import httpx
import pytest
from fastapi.testclient import TestClient

from src.runtime import api, tts_preview

TOKEN = {"X-Runtime-Token": "preview-token"}
BODY = {"provider": "cartesia", "provider_voice_id": "voice-1", "text": "Hello there"}


class FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"RIFFwav") -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.cartesia.ai/tts/bytes")
            raise httpx.HTTPStatusError(
                "secret upstream body",
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

    monkeypatch.setattr("src.runtime.tts_preview.httpx.AsyncClient", FakeClient)
    return calls


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "preview-token")
    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-test-key")


@pytest.mark.parametrize("headers", [{}, {"X-Runtime-Token": "wrong"}])
def test_preview_requires_runtime_token(headers) -> None:
    with TestClient(api.app) as client:
        response = client.post("/tts/preview", json=BODY, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_runtime_token"}


def test_preview_returns_wav_and_sends_cartesia_request(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(api.app) as client:
        response = client.post(
            "/tts/preview",
            json={**BODY, "language": "ru", "model": "sonic-x"},
            headers=TOKEN,
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFwav"
    assert calls[0]["timeout"] == 15.0
    request = calls[1]
    assert request["url"] == "https://api.cartesia.ai/tts/bytes"
    assert request["headers"]["X-API-Key"] == "cartesia-test-key"
    assert request["headers"]["Cartesia-Version"] == tts_preview.CARTESIA_VERSION
    assert request["json"] == {
        "model_id": "sonic-x",
        "transcript": "Hello there",
        "voice": {"mode": "id", "id": "voice-1"},
        "language": "ru",
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
    }


def test_preview_uses_default_model_and_omits_language(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(api.app) as client:
        assert client.post("/tts/preview", json=BODY, headers=TOKEN).status_code == 200
    assert calls[1]["json"]["model_id"] == tts_preview.DEFAULT_MODEL
    assert "language" not in calls[1]["json"]


def test_preview_rejects_unsupported_provider(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(api.app) as client:
        response = client.post(
            "/tts/preview", json={**BODY, "provider": "other"}, headers=TOKEN
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported_tts_provider"}
    assert calls == []


def test_preview_dispatches_to_a_registered_provider(monkeypatch) -> None:
    seen: list[dict] = []

    async def fake_synth(**kwargs) -> bytes:
        seen.append(kwargs)
        return b"RIFFfake"

    monkeypatch.setitem(
        tts_preview.PREVIEW_PROVIDERS, "fake", tts_preview.PreviewProvider("FAKE_KEY", fake_synth)
    )
    monkeypatch.setenv("FAKE_KEY", "fake-secret")
    with TestClient(api.app) as client:
        response = client.post("/tts/preview", json={**BODY, "provider": "fake"}, headers=TOKEN)
    assert response.status_code == 200 and response.content == b"RIFFfake"
    assert seen[0]["api_key"] == "fake-secret" and seen[0]["provider_voice_id"] == "voice-1"
    monkeypatch.delenv("FAKE_KEY")
    with TestClient(api.app) as client:
        response = client.post("/tts/preview", json={**BODY, "provider": "fake"}, headers=TOKEN)
    assert response.status_code == 503
    with pytest.raises(tts_preview.TTSPreviewError, match="unsupported tts provider: nope"):
        tts_preview.get_preview_provider("nope")


def test_preview_requires_cartesia_key(monkeypatch) -> None:
    monkeypatch.delenv("CARTESIA_API_KEY")
    calls = install_client(monkeypatch)
    with TestClient(api.app) as client:
        response = client.post("/tts/preview", json=BODY, headers=TOKEN)
    assert response.status_code == 503
    assert response.json() == {"detail": "tts_not_configured"}
    assert calls == []


@pytest.mark.parametrize(
    "body",
    [
        {**BODY, "text": ""},
        {**BODY, "text": "x" * 301},
        {**BODY, "provider_voice_id": ""},
        {**BODY, "provider_voice_id": "v" * 201},
    ],
)
def test_preview_validates_body(body) -> None:
    with TestClient(api.app) as client:
        assert client.post("/tts/preview", json=body, headers=TOKEN).status_code == 422


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response": FakeResponse(401, b"upstream says key cartesia-test-key bad")},
        {"error": httpx.ConnectError("boom cartesia-test-key")},
        {"response": FakeResponse(200, b"")},
    ],
)
def test_preview_maps_provider_failures_without_leaking(monkeypatch, kwargs) -> None:
    install_client(monkeypatch, **kwargs)
    with TestClient(api.app) as client:
        response = client.post("/tts/preview", json=BODY, headers=TOKEN)
    assert response.status_code == 502
    assert response.json() == {"detail": "tts_provider_error"}
    assert "cartesia-test-key" not in response.text


def test_elevenlabs_preview_returns_a_wav_wrapping_the_pcm(monkeypatch) -> None:
    import io
    import wave

    pcm = b"\x01\x00" * 480
    calls = install_client(monkeypatch, FakeResponse(content=pcm))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")
    with TestClient(api.app) as client:
        response = client.post(
            "/tts/preview",
            json={**BODY, "provider": "elevenlabs", "language": "ru"},
            headers=TOKEN,
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    request = calls[1]
    assert request["url"] == "https://api.elevenlabs.io/v1/text-to-speech/voice-1"
    assert request["params"] == {"output_format": "pcm_24000"}
    assert request["headers"]["xi-api-key"] == "eleven-test-key"
    assert request["json"] == {
        "text": "Hello there",
        "model_id": tts_preview.ELEVENLABS_DEFAULT_MODEL,
        "language_code": "ru",
    }
    with wave.open(io.BytesIO(response.content), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 24000)
        assert handle.readframes(handle.getnframes()) == pcm


def test_elevenlabs_preview_errors_hide_the_upstream_body(monkeypatch) -> None:
    install_client(monkeypatch, FakeResponse(status_code=401))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")
    with TestClient(api.app) as client:
        response = client.post(
            "/tts/preview", json={**BODY, "provider": "elevenlabs"}, headers=TOKEN
        )
    assert response.status_code >= 400
    assert "secret upstream body" not in response.text
    assert "eleven-test-key" not in response.text


def test_elevenlabs_preview_requires_its_own_key(monkeypatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    calls = install_client(monkeypatch)
    with TestClient(api.app) as client:
        response = client.post(
            "/tts/preview", json={**BODY, "provider": "elevenlabs"}, headers=TOKEN
        )
    assert response.status_code == 503
    assert calls == []


@pytest.mark.parametrize(
    ("model", "language", "sent"),
    [
        (None, "ko", "ko"),  # default eleven_flash_v2_5
        ("eleven_flash_v2_5", "ko-KR", "ko"),  # region suffix reduced
        ("eleven_v3", "sw", "sw"),  # v3 covers more languages
        ("eleven_flash_v2_5", "sw", None),  # not listed for flash
        ("eleven_flash_v2_5", "not a language", None),
        ("eleven_multilingual_v2", "en", None),  # language_code unsupported for multilingual_v2
        ("eleven_flash_v2", "en", None),  # English-only model
        ("some_future_model", "en", None),  # unknown model
        ("eleven_flash_v2_5", None, None),
    ],
)
def test_elevenlabs_preview_sends_language_code_only_when_supported(
    monkeypatch, model, language, sent
) -> None:
    calls = install_client(monkeypatch, FakeResponse(content=b"\x01\x00" * 8))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")
    body = {**BODY, "provider": "elevenlabs"}
    if model:
        body["model"] = model
    if language:
        body["language"] = language
    with TestClient(api.app) as client:
        response = client.post("/tts/preview", json=body, headers=TOKEN)
    assert response.status_code == 200
    payload = calls[1]["json"]
    assert payload["model_id"] == (model or "eleven_flash_v2_5")
    assert payload.get("language_code") == sent
    assert ("language_code" in payload) is (sent is not None)


def test_preview_language_table_mirrors_pipecat() -> None:
    tts_base = pytest.importorskip("pipecat.services.elevenlabs.tts_base")
    assert tts_preview.ELEVENLABS_MODEL_LANGUAGES == tts_base.ELEVENLABS_MODEL_LANGUAGES


BAD_VOICE_IDS = [
    "../../v1/user",
    "a/b",
    "a?b=c",
    "%2e%2e",
    "",
    "x" * 129,
    "has space",
    "tab\tid",
    "abc\n",
]


@pytest.mark.parametrize("bad", BAD_VOICE_IDS)
def test_elevenlabs_function_rejects_unsafe_voice_ids_before_any_request(monkeypatch, bad) -> None:
    import asyncio

    calls = install_client(monkeypatch)
    with pytest.raises(tts_preview.TTSPreviewError, match="invalid voice id"):
        asyncio.run(
            tts_preview.synthesize_elevenlabs_preview(
                api_key="k", provider_voice_id=bad, text="hi"
            )
        )
    assert calls == []


@pytest.mark.parametrize("bad", [b for b in BAD_VOICE_IDS if b])
def test_elevenlabs_preview_endpoint_rejects_unsafe_voice_ids_without_a_request(
    monkeypatch, bad
) -> None:
    calls = install_client(monkeypatch)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")
    with TestClient(api.app) as client:
        response = client.post(
            "/tts/preview",
            json={**BODY, "provider": "elevenlabs", "provider_voice_id": bad},
            headers=TOKEN,
        )
    assert response.status_code == 502
    assert response.json() == {"detail": "tts_provider_error"}
    assert calls == []


def test_elevenlabs_valid_voice_id_is_sent_as_the_last_path_segment(monkeypatch) -> None:
    calls = install_client(monkeypatch, FakeResponse(content=b"\x01\x00" * 10))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "eleven-test-key")
    with TestClient(api.app) as client:
        response = client.post(
            "/tts/preview",
            json={**BODY, "provider": "elevenlabs", "provider_voice_id": "JBFqnCBsd6RMkjVDRZzb"},
            headers=TOKEN,
        )
    assert response.status_code == 200
    assert calls[1]["url"] == (
        "https://api.elevenlabs.io/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb"
    )


def test_cartesia_preview_still_sends_the_voice_id_in_the_json_body(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(api.app) as client:
        client.post(
            "/tts/preview", json={**BODY, "provider_voice_id": "any/odd id"}, headers=TOKEN
        )
    assert calls[1]["json"]["voice"]["id"] == "any/odd id"

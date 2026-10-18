import httpx
import pytest
from fastapi.testclient import TestClient

from src.runtime import api, voice_cloning

TOKEN = {"X-Runtime-Token": "clone-token"}
KEY = "cartesia-secret-key"
URL = "https://samples.example.com/me.wav"
BODY = {"provider": "cartesia", "name": "My voice", "language": "en", "sample_url": URL}
PUBLIC_IP = "93.184.216.34"


class FakeStream:
    def __init__(self, status=200, headers=None, chunks=(b"RIFFaudio",)):
        self.status_code = status
        self.headers = (
            {"content-type": "audio/wav"} if headers is None else headers
        )
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class FakePostResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = {"id": "voice-123"} if payload is None else payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.cartesia.ai/voices/clone")
            raise httpx.HTTPStatusError(
                f"upstream body {KEY}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self):
        return self._payload


def install(monkeypatch, streams=None, post=None, post_error=None, resolve=None):
    """Mock DNS + httpx. streams: list of FakeStream consumed per GET."""
    calls: list[dict] = []
    queue = list(streams or [FakeStream()])

    async def fake_resolve(host):
        calls.append({"resolve": host})
        return resolve(host) if resolve else [PUBLIC_IP]

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def stream(self, method, url, **kwargs):
            calls.append({"stream": (method, url), "kwargs": kwargs})
            return queue.pop(0)

        async def post(self, url, **kwargs):
            calls.append({"post": url, **kwargs})
            if post_error is not None:
                raise post_error
            return post or FakePostResponse()

    monkeypatch.setattr(voice_cloning, "_resolve_host", fake_resolve)
    monkeypatch.setattr("src.runtime.voice_cloning.httpx.AsyncClient", FakeClient)
    return calls


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "clone-token")
    monkeypatch.setenv("CARTESIA_API_KEY", KEY)


def post(body=BODY, headers=TOKEN):
    with TestClient(api.app) as client:
        return client.post("/voices/clone", json=body, headers=headers)


@pytest.mark.parametrize("headers", [{}, {"X-Runtime-Token": "wrong"}])
def test_clone_requires_runtime_token(monkeypatch, headers) -> None:
    calls = install(monkeypatch)
    response = post(headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_runtime_token"}
    assert calls == []


def test_clone_rejects_unknown_provider_and_missing_key(monkeypatch) -> None:
    install(monkeypatch)
    bad = post({**BODY, "provider": "other"})
    assert bad.status_code == 422
    assert bad.json() == {"detail": "unsupported_tts_provider"}
    monkeypatch.delenv("CARTESIA_API_KEY")
    missing = post()
    assert missing.status_code == 503
    assert missing.json() == {"detail": "tts_not_configured"}


@pytest.mark.parametrize("language", ["eng", "E", "e1", "EN"])
def test_clone_validates_language(monkeypatch, language) -> None:
    install(monkeypatch)
    assert post({**BODY, "language": language}).status_code == 422


def test_clone_success_downloads_then_uploads(monkeypatch) -> None:
    calls = install(monkeypatch)
    response = post({**BODY, "description": "calm"})
    assert response.status_code == 200
    assert response.json() == {"provider_voice_id": "voice-123"}
    stream = next(c for c in calls if "stream" in c)
    # Connects to the validated IP, keeping the hostname for Host header and TLS SNI.
    assert stream["stream"] == ("GET", f"https://{PUBLIC_IP}/me.wav")
    assert stream["kwargs"]["headers"] == {"Host": "samples.example.com"}
    assert stream["kwargs"]["extensions"] == {"sni_hostname": "samples.example.com"}
    assert {"client": {"timeout": 20.0, "follow_redirects": False}} in calls
    upload = next(c for c in calls if "post" in c)
    assert upload["post"] == "https://api.cartesia.ai/voices/clone"
    assert upload["data"] == {"name": "My voice", "language": "en", "description": "calm"}
    assert upload["files"]["clip"] == ("me.wav", b"RIFFaudio", "audio/wav")
    assert upload["headers"]["X-API-Key"] == KEY
    assert upload["headers"]["Cartesia-Version"] == voice_cloning.CLONE_CARTESIA_VERSION


def test_clone_accepts_octet_stream_with_params(monkeypatch) -> None:
    install(
        monkeypatch,
        streams=[FakeStream(headers={"content-type": "Application/Octet-Stream; x=y"})],
    )
    assert post().status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        "http://samples.example.com/me.wav",
        "file:///etc/passwd",
        "ftp://samples.example.com/me.wav",
        "https://user:pw@samples.example.com/me.wav",
        "https:///me.wav",
        "https://127.0.0.1/me.wav",
        "https://[::1]/me.wav",
        "https://169.254.169.254/latest",
        "https://10.0.0.5/me.wav",
        "https://[::ffff:192.168.1.1]/me.wav",
        "https://100.100.100.200/latest/meta-data",
        "https://100.64.0.1/me.wav",
        "https://[::ffff:100.100.100.200]/me.wav",
    ],
)
def test_clone_rejects_unsupported_urls(monkeypatch, url) -> None:
    calls = install(monkeypatch)
    response = post({**BODY, "sample_url": url})
    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported_sample_url"}
    assert not any("stream" in c or "post" in c for c in calls)


@pytest.mark.parametrize(
    "resolved",
    [
        ["10.1.2.3"],
        ["127.0.0.1"],
        ["169.254.169.254"],
        ["100.100.100.200"],
        ["100.64.0.1"],
        ["fe80::1"],
        [PUBLIC_IP, "192.168.0.9"],
    ],
)
def test_clone_rejects_hosts_resolving_to_private_ips(monkeypatch, resolved) -> None:
    calls = install(monkeypatch, resolve=lambda host: resolved)
    response = post()
    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported_sample_url"}
    assert not any("stream" in c or "post" in c for c in calls)


def test_clone_unresolvable_host_is_sample_unavailable(monkeypatch) -> None:
    async def fail(host):
        raise voice_cloning.SampleUnavailable("sample host did not resolve")

    install(monkeypatch)
    monkeypatch.setattr(voice_cloning, "_resolve_host", fail)
    response = post()
    assert response.status_code == 422
    assert response.json() == {"detail": "sample_unavailable"}


def test_clone_redirect_is_followed_and_rechecked(monkeypatch) -> None:
    hosts = {"samples.example.com": [PUBLIC_IP], "cdn.example.com": ["93.184.216.35"]}
    calls = install(
        monkeypatch,
        streams=[
            FakeStream(302, {"location": "https://cdn.example.com/a.mp3"}),
            FakeStream(headers={"content-type": "audio/mpeg"}),
        ],
        resolve=lambda host: hosts[host],
    )
    assert post().status_code == 200
    streams = [c for c in calls if "stream" in c]
    assert streams[1]["stream"] == ("GET", "https://93.184.216.35/a.mp3")
    assert streams[1]["kwargs"]["extensions"] == {"sni_hostname": "cdn.example.com"}
    upload = next(c for c in calls if "post" in c)
    assert upload["files"]["clip"][0] == "a.mp3"


def test_clone_redirect_to_private_host_or_http_rejected(monkeypatch) -> None:
    install(
        monkeypatch,
        streams=[FakeStream(302, {"location": "https://internal.example.com/x"})],
        resolve=lambda host: ["10.0.0.1"] if host == "internal.example.com" else [PUBLIC_IP],
    )
    assert post().json() == {"detail": "unsupported_sample_url"}
    install(
        monkeypatch,
        streams=[FakeStream(301, {"location": "http://samples.example.com/x"})],
    )
    response = post()
    assert response.status_code == 422
    assert response.json() == {"detail": "unsupported_sample_url"}


def test_clone_too_many_redirects(monkeypatch) -> None:
    redirect = {"location": "https://samples.example.com/next"}
    install(monkeypatch, streams=[FakeStream(302, redirect) for _ in range(4)])
    response = post()
    assert response.status_code == 422
    assert response.json() == {"detail": "sample_unavailable"}


def test_clone_three_redirects_allowed(monkeypatch) -> None:
    redirect = {"location": "https://samples.example.com/next"}
    install(monkeypatch, streams=[FakeStream(302, redirect) for _ in range(3)] + [FakeStream()])
    assert post().status_code == 200


@pytest.mark.parametrize(
    "stream",
    [
        FakeStream(404),
        FakeStream(headers={"content-type": "text/html"}),
        FakeStream(headers={}),
        FakeStream(chunks=()),
        FakeStream(headers={"content-type": "audio/wav", "content-length": str(10 * 1024 * 1024 + 1)}),
        FakeStream(chunks=(b"a" * (6 * 1024 * 1024), b"a" * (5 * 1024 * 1024))),
    ],
)
def test_clone_sample_unavailable_cases(monkeypatch, stream) -> None:
    calls = install(monkeypatch, streams=[stream])
    response = post()
    assert response.status_code == 422
    assert response.json() == {"detail": "sample_unavailable"}
    assert not any("post" in c for c in calls)


def test_clone_exactly_max_size_is_accepted(monkeypatch) -> None:
    install(monkeypatch, streams=[FakeStream(chunks=(b"a" * voice_cloning.MAX_SAMPLE_BYTES,))])
    assert post().status_code == 200


def test_clone_download_transport_error_is_sample_unavailable(monkeypatch) -> None:
    class Boom(FakeStream):
        async def __aenter__(self):
            raise httpx.ConnectTimeout("slow")

    install(monkeypatch, streams=[Boom()])
    response = post()
    assert response.status_code == 422
    assert response.json() == {"detail": "sample_unavailable"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"post": FakePostResponse(401)},
        {"post": FakePostResponse(500)},
        {"post": FakePostResponse(200, {"nope": 1})},
        {"post": FakePostResponse(200, ["x"])},
        {"post_error": httpx.ReadTimeout("boom")},
    ],
)
def test_clone_provider_failure_does_not_leak(monkeypatch, kwargs) -> None:
    install(monkeypatch, **kwargs)
    response = post()
    assert response.status_code == 502
    assert response.json() == {"detail": "clone_provider_error"}
    assert KEY not in response.text
    assert "upstream" not in response.text


def test_clone_invalid_json_from_provider(monkeypatch) -> None:
    class BadJson(FakePostResponse):
        def json(self):
            raise ValueError("bad")

    install(monkeypatch, post=BadJson())
    assert post().status_code == 502


def test_is_public_uses_global_ranges() -> None:
    assert voice_cloning._is_public(PUBLIC_IP)
    assert voice_cloning._is_public("2606:2800:220:1:248:1893:25c8:1946")
    for blocked in ("100.100.100.200", "100.64.0.0", "100.127.255.255", "0.0.0.0",
                    "192.0.0.1", "224.0.0.1", "::ffff:100.100.100.200", "::1", "not-an-ip"):
        assert not voice_cloning._is_public(blocked), blocked


def test_clone_dns_rebinding_cannot_reach_loopback(monkeypatch) -> None:
    """DNS answers public on the first lookup and loopback on any later one."""
    answers = iter([[PUBLIC_IP], ["127.0.0.1"], ["127.0.0.1"]])
    calls = install(monkeypatch, resolve=lambda host: next(answers))
    assert post().status_code == 200
    streams = [c["stream"] for c in calls if "stream" in c]
    assert streams == [("GET", f"https://{PUBLIC_IP}/me.wav")]
    assert not any("127.0.0.1" in url for _, url in streams)
    # The hostname is resolved exactly once per hop; the request never re-resolves it.
    assert sum(1 for c in calls if "resolve" in c) == 1


def test_clone_redirect_hop_is_revalidated_and_pinned(monkeypatch) -> None:
    """A redirect to a host that now resolves to loopback never gets a request."""
    answers = {"samples.example.com": [PUBLIC_IP], "evil.example.com": ["127.0.0.1"]}
    calls = install(
        monkeypatch,
        streams=[FakeStream(302, {"location": "https://evil.example.com/x.wav"})],
        resolve=lambda host: answers[host],
    )
    assert post().json() == {"detail": "unsupported_sample_url"}
    assert [c["stream"][1] for c in calls if "stream" in c] == [f"https://{PUBLIC_IP}/me.wav"]


def test_pinned_request_keeps_port_and_ipv6() -> None:
    url, headers, ext = voice_cloning._pinned_request(
        "https://samples.example.com:8443/a/b.wav?x=1", ["2606:2800::1", PUBLIC_IP]
    )
    assert url == f"https://{PUBLIC_IP}:8443/a/b.wav?x=1"
    assert headers == {"Host": "samples.example.com:8443"}
    assert ext == {"sni_hostname": "samples.example.com"}
    url, _, _ = voice_cloning._pinned_request("https://h.example.com/a", ["2606:2800::1"])
    assert url == "https://[2606:2800::1]/a"


def test_pinned_request_real_httpx_transport_targets_ip() -> None:
    """The built request goes to the IP with the original Host, no network involved."""
    url, headers, ext = voice_cloning._pinned_request(
        "https://samples.example.com/me.wav", [PUBLIC_IP]
    )
    request = httpx.Request("GET", url, headers=headers, extensions=ext)
    assert request.url.host == PUBLIC_IP
    assert request.headers["host"] == "samples.example.com"
    assert request.extensions["sni_hostname"] == "samples.example.com"

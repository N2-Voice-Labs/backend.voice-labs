"""POST /voices/clone/upload (recorded sample), ElevenLabs cloning and the stock-voice fallback."""

import json
import os
from uuid import uuid4

import httpx
import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform import auth, voices
from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import ApiKey, AuditLog, Organization, ProviderConfig, Voice, VoiceSession
from src.platform.organizations import ensure_default_org

SUPER_KEY = "clone-upload-super"
CONSENT = {
    "confirmed": True,
    "subject_name": "Ada Lovelace",
    "statement": "I consent to cloning my voice.",
    "method": "self",
}
AUDIO = b"RIFF-not-really-audio-but-bytes"
REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def env(monkeypatch):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER_KEY)
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test/")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def make_org() -> str:
    org_id = str(uuid4())
    with SessionLocal() as db:
        db.add(Organization(id=org_id, name="Org", slug=f"org-{org_id}", status="active"))
        db.commit()
    return org_id


def make_key(org_id: str, role: str) -> dict:
    plaintext, prefix, digest = auth.generate_api_key()
    with SessionLocal() as db:
        db.add(
            ApiKey(
                id=str(uuid4()),
                organization_id=org_id,
                name="upload-test",
                key_prefix=prefix,
                key_hash=digest,
                role=role,
            )
        )
        db.commit()
    return {"Authorization": f"Bearer {plaintext}"}


@pytest.fixture
def world(client):
    org_a, org_b = make_org(), make_org()
    return {
        "a_id": org_a,
        "b_id": org_b,
        "a": make_key(org_a, "admin"),
        "a_member": make_key(org_a, "member"),
        "b": make_key(org_b, "admin"),
    }


class Runtime:
    """Real httpx client over a MockTransport, so multipart encoding is exercised for real."""

    def __init__(self, monkeypatch, *, status=200, body=None, error=None) -> None:
        self.requests: list[httpx.Request] = []
        self.on_request = None
        self.timeouts: list[float] = []
        self.status, self.error = status, error
        self.body = {"provider_voice_id": "pv-uploaded-1"} if body is None else body
        runtime = self

        def handler(request: httpx.Request) -> httpx.Response:
            runtime.requests.append(request)
            if runtime.on_request is not None:
                runtime.on_request(request)
            if runtime.error is not None:
                raise runtime.error
            return httpx.Response(runtime.status, json=runtime.body)

        def factory(*, timeout: float):
            runtime.timeouts.append(timeout)
            return REAL_ASYNC_CLIENT(timeout=timeout, transport=httpx.MockTransport(handler))

        monkeypatch.setattr("src.platform.voices.httpx.AsyncClient", factory)


def upload(
    client,
    headers,
    *,
    name="Ada",
    language="en",
    consent=CONSENT,
    audio=AUDIO,
    content_type="audio/webm",
    filename="rec.webm",
    **fields,
):
    data = {"name": name, "language": language, **fields}
    if consent is not None:
        data["consent"] = consent if isinstance(consent, str) else json.dumps(consent)
    files = {"file": (filename, audio, content_type)} if audio is not None else None
    return client.post("/voices/clone/upload", data=data, files=files, headers=headers)


def load(voice_id: str) -> Voice:
    with SessionLocal() as db:
        return db.get(Voice, voice_id)


def make_provider(client, headers, provider: str) -> str:
    response = client.post(
        "/providers", json={"kind": "tts", "provider": provider}, headers=headers
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ------------------------------------------------------------------------- success


def test_upload_success_forwards_multipart_and_marks_ready(client, world, monkeypatch):
    runtime = Runtime(monkeypatch)
    response = upload(client, world["a"], description="warm tone", provider="cartesia")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["provider_voice_id"] == "pv-uploaded-1"
    assert body["source"] == "clone"
    assert body["sample_reference"] is None
    assert body["failure_reason"] is None
    assert body["languages"] == ["en"]
    assert body["consent"]["subject_name"] == "Ada Lovelace"
    assert body["consent"]["recorded_at"]

    assert runtime.timeouts == [90.0]
    (request,) = runtime.requests
    assert str(request.url) == "http://runtime.test/voices/clone-upload"
    assert request.headers["X-Runtime-Token"] == "shared-secret"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    raw = request.content
    for fragment in (
        b'name="provider"\r\n\r\ncartesia',
        b'name="name"\r\n\r\nAda',
        b'name="language"\r\n\r\nen',
        b'name="description"\r\n\r\nwarm tone',
        b'name="file"; filename="rec.webm"',
        b"Content-Type: audio/webm",
        AUDIO,
    ):
        assert fragment in raw, fragment

    stored = load(body["id"])
    assert stored.organization_id == world["a_id"]
    assert stored.sample_reference is None
    actions = [
        row.action
        for row in SessionLocal().query(AuditLog).filter(AuditLog.resource_id == body["id"])
    ]
    assert actions == ["voice.created", "voice.provisioned"]


def test_upload_never_persists_or_audits_audio(client, world, monkeypatch):
    Runtime(monkeypatch)
    voice = upload(client, world["a"], audio=b"UNIQUE-AUDIO-MARKER").json()
    with SessionLocal() as db:
        dump = json.dumps(
            [
                [row.metadata_json for row in db.query(AuditLog).filter(AuditLog.resource_id == voice["id"])],
                db.get(Voice, voice["id"]).consent,
            ]
        )
    assert "UNIQUE-AUDIO-MARKER" not in dump
    assert "UNIQUE-AUDIO-MARKER" not in json.dumps(voice)


def test_upload_processing_before_runtime_call(client, world, monkeypatch):
    seen: list[str] = []
    runtime = Runtime(monkeypatch)

    def spy(request):
        with SessionLocal() as db:
            seen.append(db.query(Voice).filter(Voice.name == "spy-voice").one().status)

    runtime.on_request = spy
    assert upload(client, world["a"], name="spy-voice").status_code == 201
    assert seen == ["processing"]


# ------------------------------------------------------------------ validation


def no_runtime_call(monkeypatch) -> Runtime:
    return Runtime(monkeypatch, error=AssertionError("Runtime must not be called"))


def count_voices(org_id: str) -> int:
    with SessionLocal() as db:
        return db.query(Voice).filter(Voice.organization_id == org_id).count()


@pytest.mark.parametrize(
    "consent",
    [
        None,
        "not json",
        {**CONSENT, "confirmed": False},
        {"confirmed": True},
        {**CONSENT, "method": "someone-else"},
    ],
)
def test_upload_requires_valid_confirmed_consent(client, world, monkeypatch, consent):
    runtime = no_runtime_call(monkeypatch)
    response = upload(client, world["a"], consent=consent)
    assert response.status_code == 422
    assert response.json() == {"detail": "voice_consent_required"}
    assert runtime.requests == []
    assert count_voices(world["a_id"]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audio": b""},
        {"audio": None},
        {"content_type": "text/plain"},
        {"content_type": "application/octet-stream"},
        {"audio": b"x" * (voices.MAX_SAMPLE_BYTES + 1)},
    ],
    ids=["empty", "missing", "text", "octet-stream", "oversized"],
)
def test_upload_rejects_invalid_sample(client, world, monkeypatch, kwargs):
    runtime = no_runtime_call(monkeypatch)
    response = upload(client, world["a"], **kwargs)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_sample"}
    assert runtime.requests == []
    assert count_voices(world["a_id"]) == 0


def test_upload_accepts_exactly_the_size_limit(client, world, monkeypatch):
    Runtime(monkeypatch)
    response = upload(client, world["a"], audio=b"x" * voices.MAX_SAMPLE_BYTES)
    assert response.status_code == 201, response.text


def test_upload_size_cap_counts_bytes_not_content_length(client, world, monkeypatch):
    """The cap is enforced on what is read, in chunks: an oversize body is refused."""

    no_runtime_call(monkeypatch)
    monkeypatch.setattr(voices, "MAX_SAMPLE_BYTES", 1024)
    monkeypatch.setattr(voices, "UPLOAD_CHUNK_BYTES", 100)
    assert upload(client, world["a"], audio=b"x" * 1025).status_code == 422
    Runtime(monkeypatch)
    assert upload(client, world["a"], audio=b"x" * 1024).status_code == 201


def test_upload_rejects_bad_language_and_provider(client, world, monkeypatch):
    runtime = no_runtime_call(monkeypatch)
    bad_language = upload(client, world["a"], language="klingon")
    assert (bad_language.status_code, bad_language.json()["detail"]) == (
        422,
        "invalid_voice_language",
    )
    bad_provider = upload(client, world["a"], provider="azure")
    assert (bad_provider.status_code, bad_provider.json()["detail"]) == (
        422,
        "unsupported_tts_provider",
    )
    assert runtime.requests == []


def test_upload_requires_runtime_configuration(client, world, monkeypatch):
    monkeypatch.delenv("RUNTIME_API_URL")
    assert upload(client, world["a"]).json() == {"detail": "runtime_not_configured"}
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test/")
    monkeypatch.delenv("PLATFORM_RUNTIME_TOKEN")
    assert upload(client, world["a"]).json() == {"detail": "runtime_auth_not_configured"}


# -------------------------------------------------------------- Runtime failures


@pytest.mark.parametrize(
    "status,detail,reason",
    [
        (502, "clone_plan_upgrade_required", "clone_plan_upgrade_required"),
        (502, "clone_provider_error", "clone_provider_error"),
        (422, "invalid_sample", "invalid_sample"),
        (422, "unsupported_tts_provider", "unsupported_tts_provider"),
        (503, "tts_not_configured", "tts_not_configured"),
        (401, "invalid_runtime_token", "invalid_runtime_token"),
        (500, "surprise", "runtime_error_500"),
    ],
)
def test_runtime_error_marks_voice_failed(client, world, monkeypatch, status, detail, reason):
    Runtime(monkeypatch, status=status, body={"detail": detail})
    name = f"will-fail-{uuid4()}"
    response = upload(client, world["a"], name=name)
    assert response.status_code == 502
    assert response.json() == {"detail": "voice_provisioning_failed"}
    with SessionLocal() as db:
        stored = db.query(Voice).filter(Voice.name == name).one()
        assert (stored.status, stored.failure_reason) == ("failed", reason)
        assert stored.provider_voice_id is None
        failed = (
            db.query(AuditLog)
            .filter(AuditLog.resource_id == stored.id, AuditLog.action == "voice.provision_failed")
            .one()
        )
        assert failed.metadata_json["failure_reason"] == reason


def test_runtime_unreachable_marks_voice_failed(client, world, monkeypatch):
    Runtime(monkeypatch, error=httpx.ConnectError("refused"))
    response = upload(client, world["a"], name=(name := f"unreachable-{uuid4()}"))
    assert (response.status_code, response.json()) == (
        502,
        {"detail": "voice_provisioning_failed"},
    )
    with SessionLocal() as db:
        stored = db.query(Voice).filter(Voice.name == name).one()
    assert (stored.status, stored.failure_reason) == ("failed", "runtime_unreachable")


def test_runtime_invalid_response_marks_voice_failed(client, world, monkeypatch):
    Runtime(monkeypatch, body={"provider_voice_id": ""})
    assert upload(client, world["a"], name=(name := f"empty-id-{uuid4()}")).status_code == 502
    with SessionLocal() as db:
        stored = db.query(Voice).filter(Voice.name == name).one()
    assert (stored.status, stored.failure_reason) == ("failed", "runtime_invalid_response")


# --------------------------------------------------------- tenancy and roles


def test_upload_voice_is_isolated_to_its_org(client, world, monkeypatch):
    Runtime(monkeypatch)
    voice = upload(client, world["a"]).json()
    assert client.get(f"/voices/{voice['id']}", headers=world["a"]).status_code == 200
    assert client.get(f"/voices/{voice['id']}", headers=world["b"]).status_code == 404
    assert voice["id"] not in {v["id"] for v in client.get("/voices", headers=world["b"]).json()}
    assert client.delete(f"/voices/{voice['id']}", headers=world["b"]).status_code == 404
    assert load(voice["id"]).organization_id == world["a_id"]


def test_member_cannot_upload(client, world, monkeypatch):
    runtime = no_runtime_call(monkeypatch)
    response = upload(client, world["a_member"])
    assert response.status_code == 403
    assert response.json()["detail"] == {"code": "insufficient_role"}
    assert runtime.requests == []
    assert count_voices(world["a_id"]) == 0


def test_upload_requires_authentication(client, world):
    assert upload(client, {}).status_code in {401, 403}


# ---------------------------------------------------------- provider resolution


def sent_provider(runtime: Runtime) -> str:
    raw = runtime.requests[-1].content
    return raw.split(b'name="provider"\r\n\r\n')[1].split(b"\r\n")[0].decode()


def test_provider_defaults_to_cartesia_without_provider_config(client, world, monkeypatch):
    runtime = Runtime(monkeypatch)
    voice = upload(client, world["a"]).json()
    assert sent_provider(runtime) == "cartesia"
    assert voice["tts_provider_id"] is None


def test_provider_follows_org_tts_provider_config(client, world, monkeypatch):
    elevenlabs_id = make_provider(client, world["a"], "elevenlabs")
    runtime = Runtime(monkeypatch)
    voice = upload(client, world["a"]).json()
    assert sent_provider(runtime) == "elevenlabs"
    assert voice["tts_provider_id"] == elevenlabs_id
    # Another org's configuration is not consulted.
    voice_b = upload(client, world["b"]).json()
    assert sent_provider(runtime) == "cartesia"
    assert voice_b["tts_provider_id"] is None


def test_explicit_provider_wins_and_links_matching_config(client, world, monkeypatch):
    cartesia_id = make_provider(client, world["a"], "cartesia")
    elevenlabs_id = make_provider(client, world["a"], "elevenlabs")
    runtime = Runtime(monkeypatch)
    # Oldest cloneable config decides when omitted.
    assert upload(client, world["a"]).json()["tts_provider_id"] == cartesia_id
    assert sent_provider(runtime) == "cartesia"
    chosen = upload(client, world["a"], provider="elevenlabs").json()
    assert sent_provider(runtime) == "elevenlabs"
    assert chosen["tts_provider_id"] == elevenlabs_id


def test_non_cloneable_or_disabled_configs_are_ignored(client, world, monkeypatch):
    make_provider(client, world["a"], "deepgram-tts")
    disabled = make_provider(client, world["a"], "elevenlabs")
    with SessionLocal() as db:
        db.get(ProviderConfig, disabled).enabled = False
        db.commit()
    runtime = Runtime(monkeypatch)
    upload(client, world["a"])
    assert sent_provider(runtime) == "cartesia"


# --------------------------------------------------------------- ElevenLabs


class DeleteRuntime:
    def __init__(self, monkeypatch) -> None:
        self.urls: list[str] = []
        runtime = self

        def handler(request: httpx.Request) -> httpx.Response:
            runtime.urls.append(str(request.url))
            return httpx.Response(204)

        monkeypatch.setattr(
            "src.platform.voices.httpx.AsyncClient",
            lambda *, timeout: REAL_ASYNC_CLIENT(
                timeout=timeout, transport=httpx.MockTransport(handler)
            ),
        )


def test_elevenlabs_clone_delete_goes_to_runtime_with_elevenlabs(client, world, monkeypatch):
    make_provider(client, world["a"], "elevenlabs")
    Runtime(monkeypatch, body={"provider_voice_id": "el/voice 1"})
    voice = upload(client, world["a"]).json()
    deleter = DeleteRuntime(monkeypatch)
    assert client.delete(f"/voices/{voice['id']}", headers=world["a"]).status_code == 204
    assert deleter.urls == ["http://runtime.test/voices/elevenlabs/el%2Fvoice%201"]
    assert load(voice["id"]).status == "deleted"


def test_elevenlabs_upload_without_provider_config_still_deletes_via_elevenlabs(
    client, world, monkeypatch
):
    Runtime(monkeypatch, body={"provider_voice_id": "el-voice"})
    voice = upload(client, world["a"], provider="elevenlabs").json()
    assert voice["tts_provider_id"] is None
    deleter = DeleteRuntime(monkeypatch)
    assert client.delete(f"/voices/{voice['id']}", headers=world["a"]).status_code == 204
    assert deleter.urls == ["http://runtime.test/voices/elevenlabs/el-voice"]


def test_url_clone_with_elevenlabs_provider_sends_elevenlabs(client, world, monkeypatch):
    make_provider(client, world["a"], "elevenlabs")
    clone = client.post(
        "/voices/clone",
        json={
            "name": "url-clone",
            "sample_reference": "https://samples.example.com/a.wav",
            "consent": CONSENT,
            "tts_provider_id": client.get("/providers", headers=world["a"]).json()[0]["id"],
        },
        headers=world["a"],
    ).json()
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"provider_voice_id": "el-1"})

    monkeypatch.setattr(
        "src.platform.voices.httpx.AsyncClient",
        lambda *, timeout: REAL_ASYNC_CLIENT(
            timeout=timeout, transport=httpx.MockTransport(handler)
        ),
    )
    response = client.post(f"/voices/{clone['id']}/provision", headers=world["a"])
    assert response.status_code == 200, response.text
    assert sent[0]["provider"] == "elevenlabs"
    assert response.json()["provider_voice_id"] == "el-1"


def test_url_provision_maps_plan_upgrade_error(client, world, monkeypatch):
    clone = client.post(
        "/voices/clone",
        json={
            "name": "c",
            "sample_reference": "https://samples.example.com/a.wav",
            "consent": CONSENT,
        },
        headers=world["a"],
    ).json()
    Runtime(monkeypatch, status=502, body={"detail": "clone_plan_upgrade_required"})
    assert client.post(f"/voices/{clone['id']}/provision", headers=world["a"]).status_code == 502
    assert load(clone["id"]).failure_reason == "clone_plan_upgrade_required"


# ------------------------------------------------ stock-voice contract fallback


@pytest.fixture
def no_runtime(monkeypatch):
    """Sessions are created without a live Runtime; the contract is stored for inspection."""

    monkeypatch.delenv("RUNTIME_API_URL")


def make_agent(client, headers, **extra) -> dict:
    response = client.post(
        "/agents", json={"name": f"agent-{uuid4()}", **extra}, headers=headers
    )
    assert response.status_code == 201, response.text
    return response.json()


def seed_voice(org_id: str, **fields) -> str:
    values = {
        "name": "seeded",
        "source": "clone",
        "status": "ready",
        "provider_voice_id": f"pv-{uuid4()}",
        "languages": ["en"],
        "consent": {**CONSENT, "recorded_at": "2026-01-01T00:00:00+00:00"},
        **fields,
    }
    with SessionLocal() as db:
        record = Voice(organization_id=org_id, **values)
        db.add(record)
        db.commit()
        return record.id


def contract_of(response) -> dict:
    assert response.status_code == 201, response.text
    with SessionLocal() as db:
        return db.get(VoiceSession, response.json()["session_id"]).runtime_contract


def start(client, headers, agent_id):
    return client.post("/voice/sessions", json={"agent_id": agent_id}, headers=headers)


def attach_voice(agent_id: str, voice_id: str) -> None:
    from src.platform.models import Agent

    with SessionLocal() as db:
        db.get(Agent, agent_id).voice_profile_id = voice_id
        db.commit()


def test_valid_voice_is_unchanged(client, world, no_runtime):
    voice_id = seed_voice(world["a_id"], provider_voice_id="pv-good")
    agent = make_agent(client, world["a"], voice_id="legacy", voice_profile_id=voice_id)
    assert contract_of(start(client, world["a"], agent["id"]))["voice_id"] == "pv-good"


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "deleted", "provider_voice_id": None},
        {"status": "deleting"},
        {"status": "failed", "provider_voice_id": None, "failure_reason": "invalid_sample"},
        {"status": "ready", "provider_voice_id": None},
    ],
    ids=["deleted", "deleting", "failed", "ready-without-id"],
)
def test_dead_agent_voice_falls_back_to_default_voice(client, world, caplog, fields, no_runtime):
    voice_id = seed_voice(world["a_id"], **fields)
    agent = make_agent(client, world["a"], voice_id="legacy-id")
    attach_voice(agent["id"], voice_id)
    caplog.clear()
    contract = contract_of(start(client, world["a"], agent["id"]))
    assert contract["voice_id"] is None
    assert voice_id in caplog.text and "falling back to the default voice" in caplog.text


@pytest.mark.parametrize("status", ["pending", "processing", "disabled"])
def test_not_yet_ready_agent_voice_is_still_refused(client, world, status, no_runtime):
    voice_id = seed_voice(world["a_id"], status=status, provider_voice_id=None)
    agent = make_agent(client, world["a"])
    attach_voice(agent["id"], voice_id)
    response = start(client, world["a"], agent["id"])
    assert (response.status_code, response.json()["detail"]) == (
        409,
        "selected_voice_unavailable",
    )


def test_foreign_agent_voice_is_still_unavailable(client, world, no_runtime):
    foreign = seed_voice(world["b_id"])
    agent = make_agent(client, world["a"])
    attach_voice(agent["id"], foreign)
    response = start(client, world["a"], agent["id"])
    assert (response.status_code, response.json()["detail"]) == (
        409,
        "selected_voice_unavailable",
    )


def test_explicit_participant_voice_stays_validated(client, world, no_runtime):
    """A caller who picks a failed voice for a participant is told, not silently re-voiced."""

    agent = client.post(
        "/agents",
        json={
            "name": "interp",
            "mode": "interpreter",
            "interpreter": {
                "direction": "one_way",
                "languages": ["en", "ko"],
                "engine": "cascaded",
            },
        },
        headers=world["a"],
    )
    assert agent.status_code == 201, agent.text
    failed = seed_voice(world["a_id"], status="failed", provider_voice_id=None)
    response = client.post(
        "/voice/sessions",
        json={
            "agent_id": agent.json()["id"],
            "participant": {"language": "en", "voice_profile_id": failed},
        },
        headers=world["a"],
    )
    assert (response.status_code, response.json()["detail"]) == (
        409,
        "selected_voice_unavailable",
    )

import os
from datetime import timedelta
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import httpx
import pytest
from fastapi.testclient import TestClient

from src.platform.main import app

CONSENT = {
    "confirmed": True,
    "subject_name": "Ada Lovelace",
    "statement": "I consent to cloning my voice.",
    "method": "self",
}
SAMPLE = "https://samples.example.com/ada.wav"


class FakeResponse:
    def __init__(self, status_code=200, payload=None) -> None:
        self.status_code = status_code
        self._payload = {"provider_voice_id": "cart-voice-1"} if payload is None else payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://runtime.test/voices/clone")
            raise httpx.HTTPStatusError(
                "failed",
                request=request,
                response=httpx.Response(
                    self.status_code, request=request, json=self._payload
                )
                if isinstance(self._payload, dict)
                else httpx.Response(self.status_code, request=request, text="x"),
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


def make_clone(client: TestClient, sample: str = SAMPLE, **overrides) -> dict:
    response = client.post(
        "/voices/clone",
        json={
            "name": f"clone-{uuid4().hex[:8]}",
            "sample_reference": sample,
            "consent": CONSENT,
            **overrides,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def provision(client: TestClient, voice_id: str):
    return client.post(f"/voices/{voice_id}/provision")


def test_provision_success_marks_ready(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client, name="Ada", languages=["ru", "en"])
        response = provision(client, voice["id"])
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ready"
        assert body["provider_voice_id"] == "cart-voice-1"
        assert body["failure_reason"] is None
        assert client.get(f"/voices/{voice['id']}").json()["status"] == "ready"
    assert calls[0]["timeout"] == 60.0
    assert calls[1]["url"] == "http://runtime.test/voices/clone"
    assert calls[1]["headers"] == {"X-Runtime-Token": "shared-secret"}
    assert calls[1]["json"] == {
        "provider": "cartesia",
        "name": "Ada",
        "language": "ru",
        "sample_url": SAMPLE,
    }


def test_provision_sets_processing_before_calling_runtime(monkeypatch) -> None:
    seen: list[str] = []
    install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client)

        class SpyClient:
            def __init__(self, *, timeout: float) -> None: ...

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def post(self, url, **kwargs):
                from src.platform.db import SessionLocal
                from src.platform.models import Voice as VoiceRecord

                db = SessionLocal()
                try:
                    seen.append(db.get(VoiceRecord, voice["id"]).status)
                finally:
                    db.close()
                return FakeResponse()

        monkeypatch.setattr("src.platform.voices.httpx.AsyncClient", SpyClient)
        assert provision(client, voice["id"]).status_code == 200
    assert seen == ["processing"]


def test_provision_not_found() -> None:
    with TestClient(app) as client:
        response = provision(client, str(uuid4()))
    assert response.status_code == 404
    assert response.json() == {"detail": "voice_not_found"}


def test_provision_rejects_non_clone_and_wrong_status(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        design = client.post(
            "/voices/design", json={"name": "d", "design_prompt": "calm narrator"}
        ).json()
        provider_id = client.post(
            "/providers", json={"kind": "tts", "provider": "cartesia"}
        ).json()["id"]
        library = client.post(
            "/voices",
            json={"name": "l", "tts_provider_id": provider_id, "provider_voice_id": "p"},
        ).json()
        ready_clone = make_clone(client)
        assert provision(client, ready_clone["id"]).status_code == 200
        calls.clear()
        processing_clone = make_clone(client)
        client.post(f"/voices/{processing_clone['id']}/status", json={"status": "processing"})
        disabled_clone = make_clone(client)
        client.post(f"/voices/{disabled_clone['id']}/status", json={"status": "failed"})
        client.post(f"/voices/{disabled_clone['id']}/status", json={"status": "pending"})
        for voice in (design, library, ready_clone, processing_clone):
            response = provision(client, voice["id"])
            assert response.status_code == 409, voice
            assert response.json() == {"detail": "invalid_voice_status_transition"}
        # ready clone stays ready, untouched
        assert client.get(f"/voices/{ready_clone['id']}").json()["status"] == "ready"
    assert calls == []


def test_provision_requires_confirmed_consent(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    from sqlalchemy import update

    from src.platform.db import SessionLocal
    from src.platform.models import Voice as VoiceRecord

    with TestClient(app) as client:
        missing = make_clone(client)
        unconfirmed = make_clone(client)
        db = SessionLocal()
        try:
            db.execute(update(VoiceRecord).where(VoiceRecord.id == missing["id"]).values(consent=None))
            db.execute(
                update(VoiceRecord)
                .where(VoiceRecord.id == unconfirmed["id"])
                .values(consent={**CONSENT, "confirmed": False})
            )
            db.commit()
        finally:
            db.close()
        for voice in (missing, unconfirmed):
            response = provision(client, voice["id"])
            assert response.status_code == 409
            assert response.json() == {"detail": "voice_consent_required"}
            assert client.get(f"/voices/{voice['id']}").json()["status"] == "pending"
    assert calls == []


@pytest.mark.parametrize("sample", ["http://x.example.com/a.wav", "s3://bucket/a.wav"])
def test_provision_requires_https_sample(monkeypatch, sample) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client, sample=sample)
        response = provision(client, voice["id"])
        assert response.status_code == 422
        assert response.json() == {"detail": "unsupported_sample_url"}
        assert client.get(f"/voices/{voice['id']}").json()["status"] == "pending"
    assert calls == []


def test_provision_requires_runtime_configuration(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client)
        monkeypatch.delenv("PLATFORM_RUNTIME_TOKEN")
        assert provision(client, voice["id"]).json() == {
            "detail": "runtime_auth_not_configured"
        }
        monkeypatch.delenv("RUNTIME_API_URL")
        assert provision(client, voice["id"]).json() == {"detail": "runtime_not_configured"}
        assert client.get(f"/voices/{voice['id']}").json()["status"] == "pending"
    assert calls == []


@pytest.mark.parametrize(
    ("response", "error", "reason"),
    [
        (FakeResponse(422, {"detail": "sample_unavailable"}), None, "sample_unavailable"),
        (FakeResponse(502, {"detail": "clone_provider_error"}), None, "clone_provider_error"),
        (
            FakeResponse(502, {"detail": "clone_plan_upgrade_required"}),
            None,
            "clone_plan_upgrade_required",
        ),
        (FakeResponse(500, {"detail": "boom secret"}), None, "runtime_error_500"),
        (FakeResponse(503, "plain"), None, "runtime_error_503"),
        (None, httpx.ConnectError("down"), "runtime_unreachable"),
        (None, httpx.ReadTimeout("slow"), "runtime_unreachable"),
        (FakeResponse(200, {"provider_voice_id": ""}), None, "runtime_invalid_response"),
        (FakeResponse(200, {}), None, "runtime_invalid_response"),
        (FakeResponse(200, ValueError("bad json")), None, "runtime_invalid_response"),
    ],
)
def test_provision_failure_marks_failed_and_can_retry(monkeypatch, response, error, reason) -> None:
    install_client(monkeypatch, response=response, error=error)
    with TestClient(app) as client:
        voice = make_clone(client)
        failed = provision(client, voice["id"])
        assert failed.status_code == 502
        assert failed.json() == {"detail": "voice_provisioning_failed"}
        stored = client.get(f"/voices/{voice['id']}").json()
        assert stored["status"] == "failed"
        assert stored["failure_reason"] == reason
        assert stored["provider_voice_id"] is None
        # failed voices can be retried and then succeed
        install_client(monkeypatch)
        retry = provision(client, voice["id"])
        assert retry.status_code == 200
        assert retry.json()["status"] == "ready"
        assert retry.json()["failure_reason"] is None


def test_provision_normalizes_language_subtag(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client, languages=["EN-us", "ru"])
        assert provision(client, voice["id"]).status_code == 200
    assert calls[1]["json"]["language"] == "en"


@pytest.mark.parametrize("languages", [["english"], ["e1"], [""], ["-US"], ["x"], ["zh_CN"]])
def test_provision_rejects_invalid_language_before_state_change(monkeypatch, languages) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client, languages=languages)
        response = provision(client, voice["id"])
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid_voice_language"}
        stored = client.get(f"/voices/{voice['id']}").json()
        assert stored["status"] == "pending"
        assert stored["failure_reason"] is None
    assert calls == []


@pytest.mark.parametrize(
    "error", [RuntimeError("boom"), KeyError("x"), TypeError("t"), ZeroDivisionError()]
)
def test_provision_unexpected_exception_never_leaves_processing(monkeypatch, error) -> None:
    install_client(monkeypatch, error=error)
    with TestClient(app, raise_server_exceptions=False) as client:
        voice = make_clone(client)
        response = provision(client, voice["id"])
        assert response.status_code == 502
        assert response.json() == {"detail": "voice_provisioning_failed"}
        stored = client.get(f"/voices/{voice['id']}").json()
        assert stored["status"] == "failed"
        assert stored["failure_reason"] in {"runtime_error", "runtime_invalid_response"}
        if not isinstance(error, (KeyError, TypeError)):
            assert stored["failure_reason"] == "runtime_error"
        # and it is retryable
        install_client(monkeypatch)
        assert provision(client, voice["id"]).json()["status"] == "ready"


def _set_updated_at(voice_id: str, status: str, age: timedelta) -> None:
    from sqlalchemy import update

    from src.platform.db import SessionLocal
    from src.platform.models import Voice as VoiceRecord, utcnow

    with SessionLocal() as db:
        db.execute(
            update(VoiceRecord)
            .where(VoiceRecord.id == voice_id)
            .values(status=status, updated_at=utcnow() - age)
            # Core update with an explicit value bypasses the ORM onupdate default.
        )
        db.commit()


def test_provision_retries_voice_stuck_in_processing(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        stale = make_clone(client)
        fresh = make_clone(client)
        _set_updated_at(stale["id"], "processing", timedelta(minutes=6))
        _set_updated_at(fresh["id"], "processing", timedelta(minutes=1))
        assert provision(client, fresh["id"]).status_code == 409
        assert client.get(f"/voices/{fresh['id']}").json()["status"] == "processing"
        assert calls == []
        response = provision(client, stale["id"])
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "ready"


def test_auto_provision_clones_without_a_second_call(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_AUTO_PROVISION_CLONES", "1")
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client)
        # The response is written first; provisioning follows as a background task.
        assert voice["status"] == "pending"
        stored = client.get(f"/voices/{voice['id']}").json()
        assert stored["status"] == "ready"
        assert stored["provider_voice_id"] == "cart-voice-1"
    assert calls[1]["json"]["sample_url"] == SAMPLE


def test_auto_provision_is_off_by_default(monkeypatch) -> None:
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client)
        assert client.get(f"/voices/{voice['id']}").json()["status"] == "pending"
    assert calls == []


def test_auto_provision_failure_is_visible_and_retry_provisions_again(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_AUTO_PROVISION_CLONES", "1")
    install_client(
        monkeypatch, response=FakeResponse(502, {"detail": "clone_plan_upgrade_required"})
    )
    with TestClient(app) as client:
        voice = make_clone(client)
        stored = client.get(f"/voices/{voice['id']}").json()
        assert stored["status"] == "failed"
        assert stored["failure_reason"] == "clone_plan_upgrade_required"
        # What the frontend's Retry button sends.
        install_client(monkeypatch)
        retry = client.post(f"/voices/{voice['id']}/status", json={"status": "pending"})
        assert retry.status_code == 200
        assert client.get(f"/voices/{voice['id']}").json()["status"] == "ready"


def test_auto_provision_records_a_sample_it_cannot_fetch(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_AUTO_PROVISION_CLONES", "1")
    calls = install_client(monkeypatch)
    with TestClient(app) as client:
        voice = make_clone(client, sample="s3://bucket/a.wav")
        stored = client.get(f"/voices/{voice['id']}").json()
        assert stored["status"] == "failed"
        assert stored["failure_reason"] == "unsupported_sample_url"
    assert calls == []

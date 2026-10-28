import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app

CONSENT = {
    "confirmed": True,
    "subject_name": "Ada Lovelace",
    "statement": "I consent to cloning my voice.",
    "method": "self",
}


def unique(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def make_provider(client: TestClient, kind: str = "tts", enabled: bool = True) -> str:
    response = client.post(
        "/providers", json={"kind": kind, "provider": "test", "enabled": enabled}
    )
    assert response.status_code == 201
    return response.json()["id"]


def make_library_voice(client: TestClient, **overrides) -> dict:
    payload = {
        "name": unique("lib"),
        "tts_provider_id": make_provider(client),
        "provider_voice_id": "pv-1",
        **overrides,
    }
    response = client.post("/voices", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def make_design_voice(client: TestClient) -> dict:
    response = client.post(
        "/voices/design",
        json={"name": unique("design"), "design_prompt": "A calm, warm narrator"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def set_status(client: TestClient, voice_id: str, **body):
    return client.post(f"/voices/{voice_id}/status", json=body)


def test_library_voice_create_get_and_list_filters() -> None:
    with TestClient(app) as client:
        voice = make_library_voice(
            client, style={"energy": 0.5, "style": "friendly"}
        )
        assert voice["source"] == "library"
        assert voice["status"] == "ready"
        assert voice["languages"] == ["en", "ru"]
        assert voice["style"] == {"energy": 0.5, "style": "friendly"}
        assert voice["consent"] is None
        assert client.get(f"/voices/{voice['id']}").json() == voice

        ready = client.get("/voices", params={"status": "ready", "source": "library"})
        assert voice["id"] in [item["id"] for item in ready.json()]
        pending = client.get("/voices", params={"status": "pending"})
        assert voice["id"] not in [item["id"] for item in pending.json()]
        clones = client.get("/voices", params={"source": "clone"})
        assert voice["id"] not in [item["id"] for item in clones.json()]


def test_library_voice_validates_provider_and_style() -> None:
    with TestClient(app) as client:
        base = {"name": unique("lib"), "provider_voice_id": "pv-1"}
        for provider_id in (
            str(uuid4()),
            make_provider(client, kind="llm"),
            make_provider(client, enabled=False),
        ):
            response = client.post(
                "/voices", json={**base, "tts_provider_id": provider_id}
            )
            assert response.status_code == 422
            assert response.json()["detail"] == "invalid_tts_provider_id"

        tts = make_provider(client)
        assert client.post("/voices", json={**base, "tts_provider_id": tts}).status_code == 201
        for extra in (
            {"provider_voice_id": ""},
            {"style": {"energy": 1.5}},
            {"style": {"loudness": 0.2}},
            {"style": {"style": "x" * 51}},
        ):
            response = client.post(
                "/voices", json={**base, "tts_provider_id": tts, **extra}
            )
            assert response.status_code == 422


def test_get_unknown_voice_is_404() -> None:
    with TestClient(app) as client:
        response = client.get(f"/voices/{uuid4()}")
        assert response.status_code == 404
        assert response.json()["detail"] == "voice_not_found"


def test_clone_requires_consent() -> None:
    with TestClient(app) as client:
        base = {"name": unique("clone"), "sample_reference": "s3://bucket/a.wav"}
        for consent in (
            {},
            {"consent": None},
            {"consent": {**CONSENT, "confirmed": False}},
        ):
            response = client.post("/voices/clone", json={**base, **consent})
            assert response.status_code == 422
            assert response.json()["detail"] == "voice_consent_required"


def test_clone_rejects_invalid_sample_reference() -> None:
    with TestClient(app) as client:
        for reference in (
            "data:audio/wav;base64,AAAA",
            "not a uri",
            "",
            "s3://" + "a" * 1100,
        ):
            response = client.post(
                "/voices/clone",
                json={
                    "name": unique("clone"),
                    "sample_reference": reference,
                    "consent": CONSENT,
                },
            )
            assert response.status_code == 422, reference
            assert response.json()["detail"] == "invalid_sample_reference"


def test_clone_create_stores_consent_with_timestamp() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/voices/clone",
            json={
                "name": unique("clone"),
                "sample_reference": "https://example.com/sample.wav",
                "consent": CONSENT,
                "languages": ["en"],
            },
        )
        assert response.status_code == 201, response.text
        voice = response.json()
        assert voice["source"] == "clone"
        assert voice["status"] == "pending"
        assert voice["sample_reference"] == "https://example.com/sample.wav"
        assert voice["consent"]["subject_name"] == "Ada Lovelace"
        assert voice["consent"]["confirmed"] is True
        assert voice["consent"]["recorded_at"]
        assert voice["languages"] == ["en"]


def test_design_create() -> None:
    with TestClient(app) as client:
        voice = make_design_voice(client)
        assert voice["source"] == "design"
        assert voice["status"] == "pending"
        assert voice["design_prompt"] == "A calm, warm narrator"
        assert voice["provider_voice_id"] is None

        response = client.post(
            "/voices/design", json={"name": unique("design"), "design_prompt": ""}
        )
        assert response.status_code == 422


def test_voice_lifecycle_to_ready_and_disabled() -> None:
    with TestClient(app) as client:
        voice = make_design_voice(client)
        vid = voice["id"]
        assert set_status(client, vid, status="processing").json()["status"] == "processing"
        ready = set_status(client, vid, status="ready", provider_voice_id="pv-9")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["provider_voice_id"] == "pv-9"
        assert set_status(client, vid, status="disabled").json()["status"] == "disabled"
        assert set_status(client, vid, status="ready").json()["status"] == "ready"


def test_failed_stores_reason_and_retry_clears_it() -> None:
    with TestClient(app) as client:
        vid = make_design_voice(client)["id"]
        failed = set_status(client, vid, status="failed", failure_reason="provider down")
        assert failed.json()["status"] == "failed"
        assert failed.json()["failure_reason"] == "provider down"
        retry = set_status(client, vid, status="pending", failure_reason="ignored")
        assert retry.json()["status"] == "pending"
        assert retry.json()["failure_reason"] is None


def test_invalid_status_transition_is_409() -> None:
    with TestClient(app) as client:
        vid = make_design_voice(client)["id"]
        for target in ("ready", "disabled", "pending"):
            response = set_status(client, vid, status=target, provider_voice_id="pv")
            assert response.status_code == 409
            assert response.json()["detail"] == "invalid_voice_status_transition"
        assert set_status(client, vid, status="bogus").status_code == 422


def test_ready_requires_provider_voice_id() -> None:
    with TestClient(app) as client:
        vid = make_design_voice(client)["id"]
        set_status(client, vid, status="processing")
        response = set_status(client, vid, status="ready")
        assert response.status_code == 422
        assert response.json()["detail"] == "provider_voice_id_required"
        assert client.get(f"/voices/{vid}").json()["status"] == "processing"


def test_patch_voice() -> None:
    with TestClient(app) as client:
        voice = make_library_voice(client)
        new_name = unique("renamed")
        response = client.patch(
            f"/voices/{voice['id']}",
            json={"name": new_name, "languages": ["en"], "style": {"warmth": 0.9}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == new_name
        assert body["languages"] == ["en"]
        assert body["style"] == {"warmth": 0.9}
        assert body["status"] == "ready"

        for bad in ({"name": None}, {"languages": None}, {"style": None}, {"name": ""}):
            assert client.patch(f"/voices/{voice['id']}", json=bad).status_code == 422
        assert client.patch(f"/voices/{uuid4()}", json={"name": "x"}).status_code == 404


def test_delete_voice_tombstones() -> None:
    with TestClient(app) as client:
        voice = make_library_voice(client)
        assert client.delete(f"/voices/{voice['id']}").status_code == 204
        assert client.get(f"/voices/{voice['id']}").json()["status"] == "deleted"
        assert client.delete(f"/voices/{uuid4()}").status_code == 404


def test_disabled_unprovisioned_voice_cannot_skip_to_ready() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/voices/design",
            json={"name": unique("design"), "design_prompt": "calm support voice"},
        )
        vid = response.json()["id"]
        assert client.delete(f"/voices/{vid}").status_code == 204
        response = set_status(client, vid, status="ready", provider_voice_id="v-1")
        assert response.status_code == 409
        assert response.json()["detail"] == "invalid_voice_status_transition"

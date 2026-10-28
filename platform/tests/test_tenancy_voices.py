"""Organisation isolation, roles and audit for the /voices routes (organisations plan, T5b)."""

import os
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform import auth
from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import ApiKey, AuditLog, Organization, Voice
from src.platform.organizations import ensure_default_org

SUPER_KEY = "tenancy-voices-super"
SUPER = {"Authorization": f"Bearer {SUPER_KEY}"}
CONSENT = {
    "confirmed": True,
    "subject_name": "Ada",
    "statement": "I consent to cloning my voice.",
    "method": "self",
}
SAMPLE = "https://example.com/secret-sample.wav"
MISSING = "00000000-0000-4000-8000-0000000000ff"

# (method, path) pairs exercised below; consumed by the later isolation completeness check.
ISOLATION_CASES = [
    ("POST", "/voices"),
    ("POST", "/voices/clone"),
    ("POST", "/voices/design"),
    ("GET", "/voices"),
    ("GET", "/voices/{voice_id}"),
    ("PATCH", "/voices/{voice_id}"),
    ("POST", "/voices/{voice_id}/status"),
    ("DELETE", "/voices/{voice_id}"),
    ("POST", "/voices/{voice_id}/preview"),
    ("POST", "/voices/{voice_id}/provision"),
]


@pytest.fixture(autouse=True)
def schema(monkeypatch):
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
                name="voices-test",
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
    ctx = {
        "a_id": org_a,
        "b_id": org_b,
        "a": make_key(org_a, "admin"),
        "a_member": make_key(org_a, "member"),
        "b": make_key(org_b, "admin"),
        "b_member": make_key(org_b, "member"),
    }
    provider = client.post(
        "/providers", json={"kind": "tts", "provider": "test"}, headers=ctx["a"]
    )
    assert provider.status_code == 201, provider.text
    ctx["a_provider"] = provider.json()["id"]
    provider_b = client.post(
        "/providers", json={"kind": "tts", "provider": "test"}, headers=ctx["b"]
    )
    assert provider_b.status_code == 201, provider_b.text
    ctx["b_provider"] = provider_b.json()["id"]
    return ctx


def audit_actions(org_id: str, voice_id: str) -> list[str]:
    with SessionLocal() as db:
        rows = (
            db.query(AuditLog)
            .filter(AuditLog.organization_id == org_id, AuditLog.resource_id == voice_id)
            .order_by(AuditLog.created_at, AuditLog.id)
            .all()
        )
        return [row.action for row in rows]


def audit_rows(org_id: str, voice_id: str, action: str) -> list[AuditLog]:
    with SessionLocal() as db:
        return (
            db.query(AuditLog)
            .filter(
                AuditLog.organization_id == org_id,
                AuditLog.resource_id == voice_id,
                AuditLog.action == action,
            )
            .all()
        )


def make_design(client, headers, **overrides) -> dict:
    response = client.post(
        "/voices/design",
        json={"name": f"d-{uuid4().hex[:6]}", "design_prompt": "calm", **overrides},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_clone(client, headers) -> dict:
    response = client.post(
        "/voices/clone",
        json={"name": "Ada", "sample_reference": SAMPLE, "consent": CONSENT},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_library(client, headers, provider_id, provider_voice_id="pv-lib") -> dict:
    response = client.post(
        "/voices",
        json={
            "name": f"l-{uuid4().hex[:6]}",
            "tts_provider_id": provider_id,
            "provider_voice_id": provider_voice_id,
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def set_ready(voice_id: str) -> None:
    with SessionLocal() as db:
        voice = db.get(Voice, voice_id)
        voice.status = "ready"
        voice.provider_voice_id = "pv-ready"
        db.commit()


# ------------------------------------------------------------------ isolation


def test_list_excludes_other_org(client, world):
    voice_a = make_design(client, world["a"])
    voice_b = make_design(client, world["b"])
    ids_a = {v["id"] for v in client.get("/voices", headers=world["a"]).json()}
    ids_b = {v["id"] for v in client.get("/voices", headers=world["b"]).json()}
    assert voice_a["id"] in ids_a and voice_b["id"] not in ids_a
    assert voice_b["id"] in ids_b and voice_a["id"] not in ids_b
    filtered = client.get("/voices?source=design", headers=world["b"]).json()
    assert voice_a["id"] not in {v["id"] for v in filtered}


def test_get_patch_delete_cross_org_404(client, world):
    voice = make_design(client, world["a"])
    vid = voice["id"]
    for method, path, body in [
        ("GET", f"/voices/{vid}", None),
        ("PATCH", f"/voices/{vid}", {"name": "stolen"}),
        ("DELETE", f"/voices/{vid}", None),
    ]:
        response = client.request(method, path, json=body, headers=world["b"])
        assert response.status_code == 404, (method, response.text)
        assert response.json()["detail"] == "voice_not_found"
    own = client.get(f"/voices/{vid}", headers=world["a"]).json()
    assert own["name"] == voice["name"] and own["status"] == "pending"


def test_status_cross_org_404_for_superadmin_in_other_org(client, world):
    voice = make_design(client, world["a"])
    response = client.post(
        f"/voices/{voice['id']}/status",
        json={"status": "processing"},
        headers={**SUPER, "X-Organization-Id": world["b_id"]},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "voice_not_found"
    ok = client.post(
        f"/voices/{voice['id']}/status",
        json={"status": "processing"},
        headers={**SUPER, "X-Organization-Id": world["a_id"]},
    )
    assert ok.status_code == 200


def test_preview_and_provision_cross_org_404(client, world):
    clone = make_clone(client, world["a"])
    set_ready(clone["id"])
    for path, body in [
        (f"/voices/{clone['id']}/preview", {"text": "hello"}),
        (f"/voices/{clone['id']}/provision", None),
    ]:
        response = client.post(path, json=body, headers=world["b"])
        assert response.status_code == 404, path
        assert response.json()["detail"] == "voice_not_found"


def test_unknown_voice_404_all_routes(client, world):
    for method, path, body in [
        ("GET", f"/voices/{MISSING}", None),
        ("PATCH", f"/voices/{MISSING}", {"name": "n"}),
        ("DELETE", f"/voices/{MISSING}", None),
        ("POST", f"/voices/{MISSING}/preview", {"text": "t"}),
        ("POST", f"/voices/{MISSING}/provision", None),
    ]:
        response = client.request(method, path, json=body, headers=world["a"])
        assert response.status_code == 404, (method, path)


def test_cross_org_tts_provider_in_body_is_404(client, world):
    body = {"name": "n", "tts_provider_id": world["b_provider"]}
    for path, extra in [
        ("/voices", {"provider_voice_id": "pv"}),
        ("/voices/design", {"design_prompt": "calm"}),
        ("/voices/clone", {"sample_reference": SAMPLE, "consent": CONSENT}),
    ]:
        response = client.post(path, json={**body, **extra}, headers=world["a"])
        assert response.status_code == 404, path
        assert response.json()["detail"] == "provider_config_not_found"
    assert client.get("/voices", headers=world["a"]).json() == []
    # Own-org provider still works; a nonexistent provider keeps the old 422.
    assert make_library(client, world["a"], world["a_provider"])["tts_provider_id"]
    bad = client.post(
        "/voices/design",
        json={"name": "n", "design_prompt": "p", "tts_provider_id": MISSING},
        headers=world["a"],
    )
    assert bad.status_code == 422 and bad.json()["detail"] == "invalid_tts_provider_id"


def test_library_voice_rejects_other_orgs_clone_provider_id(client, world):
    clone = make_clone(client, world["b"])
    with SessionLocal() as db:
        row = db.get(Voice, clone["id"])
        row.provider_voice_id = "cloned-secret-id"
        db.commit()
    response = client.post(
        "/voices",
        json={
            "name": "n",
            "tts_provider_id": world["a_provider"],
            "provider_voice_id": "cloned-secret-id",
        },
        headers=world["a"],
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid_provider_voice_id"


def test_created_voices_stamped_with_callers_org(client, world):
    voice = make_library(client, world["b"], world["b_provider"])
    with SessionLocal() as db:
        assert db.get(Voice, voice["id"]).organization_id == world["b_id"]


# ----------------------------------------------------------------------- roles


def test_member_can_read_and_preview_but_not_write(client, world, monkeypatch):
    voice = make_design(client, world["a"])
    member = world["a_member"]
    assert client.get("/voices", headers=member).status_code == 200
    assert client.get(f"/voices/{voice['id']}", headers=member).status_code == 200
    # Preview passes the role gate (409: the design voice is not ready yet).
    preview = client.post(
        f"/voices/{voice['id']}/preview", json={"text": "hi"}, headers=member
    )
    assert preview.status_code == 409

    for method, path, body in [
        ("POST", "/voices", {"name": "n", "tts_provider_id": world["a_provider"],
                             "provider_voice_id": "pv"}),
        ("POST", "/voices/clone", {"name": "n", "sample_reference": SAMPLE,
                                   "consent": CONSENT}),
        ("POST", "/voices/design", {"name": "n", "design_prompt": "p"}),
        ("PATCH", f"/voices/{voice['id']}", {"name": "n"}),
        ("DELETE", f"/voices/{voice['id']}", None),
        ("POST", f"/voices/{voice['id']}/provision", None),
    ]:
        response = client.request(method, path, json=body, headers=member)
        assert response.status_code == 403, (method, path, response.text)
        assert response.json()["detail"] == {"code": "insufficient_role"}
    assert client.get(f"/voices/{voice['id']}", headers=member).json()["status"] == "pending"


def test_admin_can_write(client, world):
    voice = make_design(client, world["a"])
    assert client.patch(
        f"/voices/{voice['id']}", json={"name": "renamed"}, headers=world["a"]
    ).status_code == 200
    assert client.delete(f"/voices/{voice['id']}", headers=world["a"]).status_code == 204


def test_status_requires_superadmin(client, world):
    voice = make_design(client, world["a"])
    for headers in (world["a"], world["a_member"]):
        response = client.post(
            f"/voices/{voice['id']}/status", json={"status": "processing"}, headers=headers
        )
        assert response.status_code == 403
        assert response.json()["detail"] == {"code": "superadmin_required"}
    ok = client.post(
        f"/voices/{voice['id']}/status",
        json={"status": "processing"},
        headers={**SUPER, "X-Organization-Id": world["a_id"]},
    )
    assert ok.status_code == 200 and ok.json()["status"] == "processing"


def test_status_clone_cannot_set_provider_voice_id(client, world):
    clone = make_clone(client, world["a"])
    superadmin = {**SUPER, "X-Organization-Id": world["a_id"]}
    response = client.post(
        f"/voices/{clone['id']}/status",
        json={"status": "processing", "provider_voice_id": "forged"},
        headers=superadmin,
    )
    assert response.status_code == 422
    assert response.json()["detail"] == {"code": "provider_voice_id_managed"}
    assert client.get(f"/voices/{clone['id']}", headers=world["a"]).json()["status"] == "pending"
    # Without a provider id the superadmin may still move a clone through states.
    ok = client.post(
        f"/voices/{clone['id']}/status", json={"status": "processing"}, headers=superadmin
    )
    assert ok.status_code == 200
    # Non-clone voices keep accepting provider_voice_id.
    design = make_design(client, world["a"])
    client.post(f"/voices/{design['id']}/status", json={"status": "processing"}, headers=superadmin)
    ready = client.post(
        f"/voices/{design['id']}/status",
        json={"status": "ready", "provider_voice_id": "pv-ok"},
        headers=superadmin,
    )
    assert ready.status_code == 200 and ready.json()["provider_voice_id"] == "pv-ok"


# ----------------------------------------------------------------------- audit


def test_audit_created_for_each_source(client, world):
    library = make_library(client, world["a"], world["a_provider"])
    design = make_design(client, world["a"])
    clone = make_clone(client, world["a"])
    for voice, source in ((library, "library"), (design, "design"), (clone, "clone")):
        rows = audit_rows(world["a_id"], voice["id"], "voice.created")
        assert len(rows) == 1
        assert rows[0].resource_type == "voice"
        assert rows[0].metadata_json["source"] == source
    metadata = audit_rows(world["a_id"], clone["id"], "voice.created")[0].metadata_json
    assert metadata["consent"] == {"method": "self", "subject_name": "Ada"}
    assert SAMPLE not in str(metadata)
    assert "statement" not in str(metadata)
    assert audit_rows(world["b_id"], clone["id"], "voice.created") == []


def test_audit_status_changed(client, world):
    voice = make_design(client, world["a"])
    client.post(
        f"/voices/{voice['id']}/status",
        json={"status": "processing"},
        headers={**SUPER, "X-Organization-Id": world["a_id"]},
    )
    row = audit_rows(world["a_id"], voice["id"], "voice.status_changed")[0]
    assert row.metadata_json == {"from": "pending", "to": "processing"}


def install_runtime(monkeypatch, *, payload=None, content=b"RIFFwav"):
    class Resp:
        def __init__(self):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return payload or {}

    class FakeClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, **kwargs):
            return Resp()

    monkeypatch.setattr("src.platform.voices.httpx.AsyncClient", FakeClient)


def test_audit_provisioned_and_preview(client, world, monkeypatch):
    install_runtime(monkeypatch, payload={"provider_voice_id": "pv-new"})
    clone = make_clone(client, world["a"])
    response = client.post(f"/voices/{clone['id']}/provision", headers=world["a"])
    assert response.status_code == 200, response.text
    assert response.json()["provider_voice_id"] == "pv-new"
    assert "voice.provisioned" in audit_actions(world["a_id"], clone["id"])

    preview = client.post(
        f"/voices/{clone['id']}/preview", json={"text": "hello"}, headers=world["a_member"]
    )
    assert preview.status_code == 200, preview.text
    row = audit_rows(world["a_id"], clone["id"], "voice.previewed")[0]
    assert row.metadata_json == {"text_length": 5}
    assert audit_rows(world["b_id"], clone["id"], "voice.previewed") == []


def test_audit_provision_failed(client, world, monkeypatch):
    install_runtime(monkeypatch, payload={})
    clone = make_clone(client, world["a"])
    response = client.post(f"/voices/{clone['id']}/provision", headers=world["a"])
    assert response.status_code == 502
    row = audit_rows(world["a_id"], clone["id"], "voice.provision_failed")[0]
    assert row.metadata_json["failure_reason"] == "runtime_invalid_response"
    assert "voice.provisioned" not in audit_actions(world["a_id"], clone["id"])

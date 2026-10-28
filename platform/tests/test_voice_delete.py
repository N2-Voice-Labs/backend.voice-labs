"""DELETE /voices/{id}: hard delete with provider cleanup, tombstone and audit (ST-32)."""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from src.platform import auth, storage
from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import ApiKey, AuditLog, Organization, Voice
from src.platform.organizations import ensure_default_org

SUPER_KEY = "voice-delete-super"
SUPER = {"Authorization": f"Bearer {SUPER_KEY}"}
CONSENT = {
    "confirmed": True,
    "subject_name": "Ada",
    "statement": "I consent.",
    "method": "self",
}
SAMPLE = "https://example.com/secret-sample.wav"
SENSITIVE = ("sample_reference", "design_prompt", "provider_voice_id")

ISOLATION_CASES = [("DELETE", "/voices/{voice_id}")]


class FakeRuntime:
    """Stands in for httpx.AsyncClient; each DELETE pops the next scripted outcome."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.outcomes: list[object] = []

    def install(self, monkeypatch) -> None:
        runtime = self

        class FakeClient:
            def __init__(self, *, timeout: float) -> None:
                pass

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

            async def delete(self, url: str, **kwargs: object) -> httpx.Response:
                runtime.calls.append({"url": url, **kwargs})
                outcome = runtime.outcomes.pop(0) if runtime.outcomes else 204
                if isinstance(outcome, Exception):
                    raise outcome
                status, body = outcome if isinstance(outcome, tuple) else (outcome, None)
                return httpx.Response(status, json=body)

        monkeypatch.setattr("src.platform.voices.httpx.AsyncClient", FakeClient)


@pytest.fixture(autouse=True)
def schema(monkeypatch):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER_KEY)
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test/")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")


@pytest.fixture
def runtime(monkeypatch) -> FakeRuntime:
    fake = FakeRuntime()
    fake.install(monkeypatch)
    return fake


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
    key_id = str(uuid4())
    with SessionLocal() as db:
        db.add(
            ApiKey(
                id=key_id,
                organization_id=org_id,
                name="voice-delete",
                key_prefix=prefix,
                key_hash=digest,
                role=role,
            )
        )
        db.commit()
    return {"Authorization": f"Bearer {plaintext}", "_key_id": key_id}


def hdr(keys: dict) -> dict:
    return {"Authorization": keys["Authorization"]}


@pytest.fixture
def world():
    org_a, org_b = make_org(), make_org()
    return {
        "a_id": org_a,
        "a": make_key(org_a, "admin"),
        "a_member": make_key(org_a, "member"),
        "b": make_key(org_b, "admin"),
    }


def seed_voice(org_id: str, **fields) -> str:
    values = {
        "name": "v",
        "source": "clone",
        "status": "ready",
        "provider_voice_id": f"pv-{uuid4()}",
        "sample_reference": SAMPLE,
        "consent": {**CONSENT, "recorded_at": "2026-01-01T00:00:00+00:00"},
        "languages": ["en"],
        "style": {"warmth": 0.5},
        **fields,
    }
    with SessionLocal() as db:
        record = Voice(organization_id=org_id, **values)
        db.add(record)
        db.commit()
        return record.id


def load(voice_id: str) -> dict:
    with SessionLocal() as db:
        row = db.get(Voice, voice_id)
        return {c.name: getattr(row, c.name) for c in Voice.__table__.columns}


def audit_rows(voice_id: str, action: str) -> list[AuditLog]:
    with SessionLocal() as db:
        return (
            db.query(AuditLog)
            .filter(AuditLog.resource_id == voice_id, AuditLog.action == action)
            .all()
        )


def make_agent(client, headers, **extra) -> dict:
    response = client.post("/agents", json={"name": f"a-{uuid4()}", **extra}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def test_clone_delete_success_tombstones_and_retains_consent(client, world, runtime):
    voice_id = seed_voice(world["a_id"], design_prompt="warm")
    before = load(voice_id)

    response = client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    assert response.status_code == 204
    assert response.content == b""

    assert len(runtime.calls) == 1
    assert runtime.calls[0]["url"] == (
        f"http://runtime.test/voices/cartesia/{before['provider_voice_id']}"
    )
    assert runtime.calls[0]["headers"] == {"X-Runtime-Token": "shared-secret"}

    row = load(voice_id)
    assert row["status"] == "deleted"
    for field in SENSITIVE:
        assert row[field] is None, field
    assert row["style"] == {}
    assert row["consent"] == before["consent"]  # consent is retained
    assert row["deleted_at"] is not None and row["provider_deleted_at"] is not None
    assert row["deleted_by"] == world["a"]["_key_id"]
    # Object storage is not wired yet (ST-31): not confirmed deleted.
    assert row["sample_deleted_at"] is None

    audits = audit_rows(voice_id, "voice.deleted")
    assert len(audits) == 1
    assert audits[0].metadata_json["provider_deleted"] is True
    assert audits[0].metadata_json["sample_deleted"] is False
    assert SAMPLE not in str(audits[0].metadata_json)


def test_sample_deleted_at_set_only_when_storage_confirms(client, world, runtime, monkeypatch):
    monkeypatch.setattr(storage, "delete", lambda uri: "deleted")
    prefixes = []
    monkeypatch.setattr(storage, "delete_prefix", lambda p: prefixes.append(p) or "deleted")
    voice_id = seed_voice(world["a_id"])
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    assert load(voice_id)["sample_deleted_at"] is not None
    assert prefixes == [f"voices/{voice_id}/previews/"]
    assert audit_rows(voice_id, "voice.deleted")[0].metadata_json["sample_deleted"] is True


def test_library_voice_skips_provider(client, world, runtime):
    voice_id = seed_voice(world["a_id"], source="library", sample_reference=None, consent=None)
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    assert runtime.calls == []
    row = load(voice_id)
    assert row["status"] == "deleted" and row["provider_voice_id"] is None
    assert row["provider_deleted_at"] is None
    assert audit_rows(voice_id, "voice.deleted")[0].metadata_json["provider_deleted"] is False


def test_unprovisioned_design_voice_skips_provider(client, world, runtime):
    voice_id = seed_voice(
        world["a_id"], source="design", status="pending", provider_voice_id=None,
        sample_reference=None, design_prompt="calm", consent=None,
    )
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    assert runtime.calls == []
    assert load(voice_id)["design_prompt"] is None


def test_provider_404_is_success_because_runtime_returns_204(client, world, runtime):
    runtime.outcomes = [204]  # runtime maps a provider-side 404 to 204
    voice_id = seed_voice(world["a_id"], source="design")
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    assert load(voice_id)["provider_deleted_at"] is not None


def test_runtime_failure_marks_deleting_then_retry_succeeds(client, world, runtime):
    runtime.outcomes = [(502, {"code": "voice_provider_delete_failed", "retryable": True})]
    voice_id = seed_voice(world["a_id"])
    pvid = load(voice_id)["provider_voice_id"]

    first = client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    assert first.status_code == 502
    assert first.json()["detail"] == {
        "code": "voice_provider_delete_failed",
        "retryable": True,
    }
    row = load(voice_id)
    assert row["status"] == "deleting"
    assert row["provider_voice_id"] == pvid  # kept so the retry can finish the job
    assert row["deleted_at"] is None

    # Unusable while deleting, but still visible to admins.
    assert client.get(f"/voices/{voice_id}", headers=hdr(world["a"])).json()["status"] == "deleting"
    assert client.post(
        f"/voices/{voice_id}/preview", json={"text": "hi"}, headers=hdr(world["a"])
    ).status_code == 409

    second = client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    assert second.status_code == 204
    row = load(voice_id)
    assert row["status"] == "deleted" and row["provider_voice_id"] is None
    assert len(runtime.calls) == 2


@pytest.mark.parametrize(
    "outcome,retryable",
    [
        (httpx.ConnectError("down"), True),
        (httpx.ReadTimeout("slow"), True),
        ((503, {"code": "tts_not_configured"}), True),
        ((502, {"code": "voice_provider_delete_failed", "retryable": False}), False),
        ((422, {"code": "invalid_voice_id"}), False),
        ((404, {"code": "unknown_provider"}), False),
    ],
)
def test_runtime_error_variants_keep_voice_deleting(client, world, runtime, outcome, retryable):
    runtime.outcomes = [outcome]
    voice_id = seed_voice(world["a_id"])
    response = client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    assert response.status_code == 502
    assert response.json()["detail"]["retryable"] is retryable
    assert load(voice_id)["status"] == "deleting"
    failed = audit_rows(voice_id, "voice.delete_failed")
    assert len(failed) == 1 and failed[0].metadata_json["retryable"] is retryable
    assert audit_rows(voice_id, "voice.deleted") == []


def test_runtime_not_configured_returns_503_and_deletes_nothing(client, world, runtime, monkeypatch):
    monkeypatch.delenv("RUNTIME_API_URL")
    voice_id = seed_voice(world["a_id"])
    response = client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "runtime_unavailable"
    assert runtime.calls == []
    assert load(voice_id)["status"] == "ready"
    # Library voices need no provider, so they still delete.
    library = seed_voice(world["a_id"], source="library", sample_reference=None, consent=None)
    assert client.delete(f"/voices/{library}", headers=hdr(world["a"])).status_code == 204


def test_voice_in_use_409_lists_agents_then_archived_agent_unblocks(client, world, runtime):
    voice_id = seed_voice(world["a_id"], source="library", consent=None, sample_reference=None)
    agent = make_agent(client, hdr(world["a"]), voice_profile_id=voice_id)

    response = client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    assert response.status_code == 409
    assert response.json()["detail"] == {"code": "voice_in_use", "agent_ids": [agent["id"]]}
    assert load(voice_id)["status"] == "ready"
    assert runtime.calls == []

    # Archiving the agent (DELETE /agents) releases the voice.
    assert client.delete(f"/agents/{agent['id']}", headers=hdr(world["a"])).status_code == 204
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204


def test_agent_with_raw_provider_voice_id_blocks_delete(client, world, runtime):
    voice_id = seed_voice(world["a_id"], provider_voice_id="pv-raw-1")
    agent = make_agent(client, hdr(world["a"]), voice_id="pv-raw-1")
    other_id = seed_voice(world["a_id"], provider_voice_id="pv-raw-2")
    linked = make_agent(client, hdr(world["a"]), voice_profile_id=other_id)

    response = client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    assert response.status_code == 409
    assert response.json()["detail"] == {"code": "voice_in_use", "agent_ids": [agent["id"]]}
    assert runtime.calls == []

    response = client.delete(f"/voices/{other_id}", headers=hdr(world["a"]))
    assert response.json()["detail"]["agent_ids"] == [linked["id"]]

    assert client.delete(f"/agents/{agent['id']}", headers=hdr(world["a"])).status_code == 204
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204


def test_other_orgs_agent_does_not_block_nor_leak(client, world, runtime):
    voice_id = seed_voice(world["a_id"], source="library", consent=None, sample_reference=None)
    # Only the owning org's agents are considered; a plain delete is fine.
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204


def test_session_with_deleted_voice_is_unavailable(client, world, runtime):
    voice_id = seed_voice(world["a_id"], source="library", consent=None, sample_reference=None)
    agent = make_agent(client, hdr(world["a"]), voice_profile_id=voice_id)
    client.delete(f"/agents/{agent['id']}", headers=hdr(world["a"]))
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    # Re-enable the agent (patch does not re-validate an unchanged voice_profile_id).
    reenabled = client.patch(
        f"/agents/{agent['id']}", json={"status": "enabled"}, headers=hdr(world["a"])
    )
    assert reenabled.status_code == 200, reenabled.text
    session = client.post(
        "/voice/sessions", json={"agent_id": agent["id"]}, headers=hdr(world["a"])
    )
    assert session.status_code == 409
    assert session.json()["detail"] == "selected_voice_unavailable"


def test_delete_is_idempotent(client, world, runtime):
    voice_id = seed_voice(world["a_id"])
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    stamp = load(voice_id)["deleted_at"]
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    assert len(runtime.calls) == 1
    assert load(voice_id)["deleted_at"] == stamp
    assert len(audit_rows(voice_id, "voice.deleted")) == 1


def test_cross_org_404_and_member_403(client, world, runtime):
    voice_id = seed_voice(world["a_id"])
    other = client.delete(f"/voices/{voice_id}", headers=hdr(world["b"]))
    assert other.status_code == 404
    assert other.json()["detail"] == "voice_not_found"
    member = client.delete(f"/voices/{voice_id}", headers=hdr(world["a_member"]))
    assert member.status_code == 403
    assert member.json()["detail"] == {"code": "insufficient_role"}
    assert runtime.calls == [] and load(voice_id)["status"] == "ready"
    # Deleted tombstones stay invisible to other orgs, too.
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["a"])).status_code == 204
    assert client.get(f"/voices/{voice_id}", headers=hdr(world["b"])).status_code == 404
    assert client.delete(f"/voices/{voice_id}", headers=hdr(world["b"])).status_code == 404


def test_lists_hide_tombstones_and_get_shows_them_to_admins_only(client, world, runtime):
    keep = seed_voice(world["a_id"], source="library", consent=None, sample_reference=None)
    gone = seed_voice(world["a_id"])
    assert client.delete(f"/voices/{gone}", headers=hdr(world["a"])).status_code == 204

    for key in ("a", "a_member"):
        ids = {v["id"] for v in client.get("/voices", headers=hdr(world[key])).json()}
        assert keep in ids and gone not in ids
    included = client.get("/voices?include_deleted=true", headers=hdr(world["a"])).json()
    assert {keep, gone} <= {v["id"] for v in included}
    member_included = client.get(
        "/voices?include_deleted=true", headers=hdr(world["a_member"])
    ).json()
    assert gone not in {v["id"] for v in member_included}
    assert client.get("/voices?status=deleted", headers=hdr(world["a_member"])).json() == []
    assert client.get("/voices?status=deleted", headers=hdr(world["a"])).json()[0]["id"] == gone

    tomb = client.get(f"/voices/{gone}", headers=hdr(world["a"]))
    assert tomb.status_code == 200
    body = tomb.json()
    assert body["status"] == "deleted" and body["deleted_at"] is not None
    assert body["consent"]["subject_name"] == "Ada"
    assert all(body[f] is None for f in SENSITIVE)
    assert client.get(f"/voices/{gone}", headers=hdr(world["a_member"])).status_code == 404


def test_deleted_voice_cannot_be_patched_previewed_or_provisioned(client, world, runtime):
    voice_id = seed_voice(world["a_id"])
    client.delete(f"/voices/{voice_id}", headers=hdr(world["a"]))
    h = hdr(world["a"])
    assert client.patch(f"/voices/{voice_id}", json={"name": "x"}, headers=h).status_code == 404
    assert client.post(f"/voices/{voice_id}/preview", json={"text": "x"}, headers=h).status_code == 404
    assert client.post(f"/voices/{voice_id}/provision", headers=h).status_code == 404
    assert load(voice_id)["status"] == "deleted"
    # The superadmin status route (default org) cannot resurrect a tombstone either.
    own = seed_voice("00000000-0000-0000-0000-000000000001")
    client.delete(f"/voices/{own}", headers=SUPER)
    revived = client.post(f"/voices/{own}/status", json={"status": "ready"}, headers=SUPER)
    assert revived.status_code == 409
    assert load(own)["status"] == "deleted"


def test_dev_mode_delete_records_principal(monkeypatch, runtime):
    monkeypatch.delenv("PLATFORM_API_KEYS")
    voice_id = seed_voice("00000000-0000-0000-0000-000000000001", source="library")
    with TestClient(app) as c:
        assert c.delete(f"/voices/{voice_id}").status_code == 204
    assert load(voice_id)["deleted_by"] in {"dev", "superadmin"}


# ------------------------------------------------------------------ migration


def run_alembic(db_url: str, *args: str) -> None:
    platform_dir = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PLATFORM_DATABASE_URL": db_url, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PLATFORM_API_KEYS", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=platform_dir, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_migration_0011_up_down_on_sqlite():
    cache = Path.home() / ".cache" / "voice-labs-test"
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / f"voice-tombstone-{uuid4().hex}.db"
    url = f"sqlite:///{db_path}"
    new_columns = {"deleted_at", "deleted_by", "provider_deleted_at", "sample_deleted_at"}
    try:
        run_alembic(url, "upgrade", "0010_org_not_null")
        eng = create_engine(url)
        assert not new_columns & {c["name"] for c in inspect(eng).get_columns("voices")}
        with eng.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO voices (id, name, source, status, languages, style, "
                    "organization_id, created_at, updated_at) VALUES ('v1','n','library',"
                    "'ready','[]','{}','00000000-0000-0000-0000-000000000001',:n,:n)"
                ),
                {"n": datetime.now(timezone.utc)},
            )

        run_alembic(url, "upgrade", "0011_voice_tombstone")
        cols = {c["name"]: c for c in inspect(eng).get_columns("voices")}
        assert new_columns <= set(cols)
        assert all(cols[name]["nullable"] for name in new_columns)
        with eng.connect() as conn:
            row = conn.execute(text("SELECT id, status, deleted_at FROM voices")).one()
        assert tuple(row) == ("v1", "ready", None)

        run_alembic(url, "downgrade", "0010_org_not_null")
        assert not new_columns & {c["name"] for c in inspect(eng).get_columns("voices")}
        with eng.connect() as conn:
            assert conn.execute(text("SELECT count(*) FROM voices")).scalar() == 1
        run_alembic(url, "upgrade", "head")
        eng.dispose()
    finally:
        db_path.unlink(missing_ok=True)

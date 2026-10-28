"""Organisation and API-key management API (organisations plan, T7)."""

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient
from sqlalchemy import select

from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import ApiKey, AuditLog
from src.platform.organizations import ensure_default_org

SUPER_KEY = "super-key-1"
SUPER = {"Authorization": f"Bearer {SUPER_KEY}"}


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER_KEY)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    with TestClient(app) as test_client:
        yield test_client


_counter = iter(range(10_000))


def make_org(client, owner_key: bool = True) -> dict:
    slug = f"org-t7-{next(_counter)}-{os.urandom(3).hex()}"
    response = client.post(
        "/organizations",
        json={"name": "Org", "slug": slug, "create_owner_key": owner_key},
        headers=SUPER,
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_key(client, caller_key: str, role: str, **extra) -> dict:
    response = client.post(
        "/api-keys",
        json={"name": f"{role} key", "role": role, **extra},
        headers=bearer(caller_key),
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def org(client):
    created = make_org(client)
    owner = created["owner_key"]["key"]
    return {
        "id": created["id"],
        "owner": owner,
        "admin": make_key(client, owner, "admin")["key"],
        "member": make_key(client, owner, "member")["key"],
    }


def audit_rows(org_id: str, action: str) -> list[AuditLog]:
    with SessionLocal() as db:
        return list(
            db.scalars(
                select(AuditLog).where(
                    AuditLog.organization_id == org_id, AuditLog.action == action
                )
            )
        )


# --- organisations -----------------------------------------------------------------


def test_superadmin_creates_org_with_owner_key_once(client):
    created = make_org(client)
    assert created["status"] == "active"
    owner = created["owner_key"]
    assert owner["role"] == "owner" and owner["key"].startswith("vlk_")

    # The new owner key works and is bound to the new org.
    me = client.get("/organization", headers=bearer(owner["key"]))
    assert me.status_code == 200 and me.json()["id"] == created["id"]

    listing = client.get("/organizations", headers=SUPER).json()["items"]
    assert created["id"] in {o["id"] for o in listing}
    assert "owner_key" not in next(o for o in listing if o["id"] == created["id"])

    assert len(audit_rows(created["id"], "organization.created")) == 1
    assert len(audit_rows(created["id"], "api_key.created")) == 1


def test_create_org_without_owner_key(client):
    created = make_org(client, owner_key=False)
    assert "owner_key" not in created


@pytest.mark.parametrize(
    "slug", ["ab", "UPPER-case", "has_underscore", "-leading", "trailing-", "a" * 64, "sp ace"]
)
def test_slug_validation(client, slug):
    response = client.post(
        "/organizations", json={"name": "X", "slug": slug}, headers=SUPER
    )
    assert response.status_code == 422


def test_duplicate_slug_conflicts(client):
    slug = f"dup-{os.urandom(3).hex()}"
    assert (
        client.post("/organizations", json={"name": "A", "slug": slug}, headers=SUPER).status_code
        == 201
    )
    again = client.post("/organizations", json={"name": "B", "slug": slug}, headers=SUPER)
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "slug_taken"


def test_superadmin_only_routes_reject_org_keys(client, org):
    for role in ("owner", "admin", "member"):
        headers = bearer(org[role])
        for method, path, body in (
            ("post", "/organizations", {"name": "X", "slug": "xyz-1"}),
            ("get", "/organizations", None),
            ("post", f"/organizations/{org['id']}/status", {"status": "suspended"}),
        ):
            response = client.request(method, path, json=body, headers=headers)
            assert response.status_code == 403, (role, path)
            assert response.json()["detail"] == {"code": "superadmin_required"}


def test_status_change_suspends_keys_and_is_audited(client, org):
    response = client.post(
        f"/organizations/{org['id']}/status", json={"status": "suspended"}, headers=SUPER
    )
    assert response.status_code == 200 and response.json()["status"] == "suspended"
    assert client.get("/organization", headers=bearer(org["owner"])).status_code == 401
    assert len(audit_rows(org["id"], "organization.status_changed")) == 1

    client.post(
        f"/organizations/{org['id']}/status", json={"status": "active"}, headers=SUPER
    )
    assert client.get("/organization", headers=bearer(org["owner"])).status_code == 200

    bad = client.post(
        f"/organizations/{org['id']}/status", json={"status": "gone"}, headers=SUPER
    )
    assert bad.status_code == 422
    missing = client.post(
        "/organizations/nope/status", json={"status": "active"}, headers=SUPER
    )
    assert missing.status_code == 404


def test_get_and_patch_own_organization(client, org):
    for role in ("owner", "admin", "member"):
        response = client.get("/organization", headers=bearer(org[role]))
        assert response.status_code == 200
        assert set(response.json()) == {"id", "name", "slug", "status", "created_at"}

    for role in ("admin", "member"):
        denied = client.patch("/organization", json={"name": "New"}, headers=bearer(org[role]))
        assert denied.status_code == 403

    ok = client.patch("/organization", json={"name": "  Renamed  "}, headers=bearer(org["owner"]))
    assert ok.status_code == 200 and ok.json()["name"] == "Renamed"
    assert len(audit_rows(org["id"], "organization.updated")) == 1
    empty = client.patch("/organization", json={"name": " "}, headers=bearer(org["owner"]))
    assert empty.status_code == 422


# --- API keys ----------------------------------------------------------------------


def test_role_ceiling_on_create(client, org):
    admin = bearer(org["admin"])
    assert client.post("/api-keys", json={"name": "k", "role": "owner"}, headers=admin).status_code == 403
    assert client.post("/api-keys", json={"name": "k", "role": "admin"}, headers=admin).status_code == 201
    assert client.post("/api-keys", json={"name": "k", "role": "member"}, headers=admin).status_code == 201

    member = bearer(org["member"])
    denied = client.post("/api-keys", json={"name": "k", "role": "member"}, headers=member)
    assert denied.status_code == 403
    assert client.get("/api-keys", headers=member).status_code == 403

    # Superadmin may mint any role, including owner, in the org it is acting for.
    created = client.post(
        "/api-keys",
        json={"name": "root", "role": "owner"},
        headers={**SUPER, "X-Organization-Id": org["id"]},
    )
    assert created.status_code == 201
    me = client.get("/organization", headers=bearer(created.json()["key"]))
    assert me.json()["id"] == org["id"]

    assert client.post("/api-keys", json={"name": "k", "role": "root"}, headers=admin).status_code == 422


def test_plaintext_only_on_create(client, org):
    created = make_key(client, org["owner"], "member")
    assert set(created) >= {"id", "name", "role", "key_prefix", "created_at", "expires_at", "key"}
    assert created["key"].startswith(created["key_prefix"])

    listing = client.get("/api-keys", headers=bearer(org["owner"])).json()["items"]
    row = next(k for k in listing if k["id"] == created["id"])
    assert "key" not in row and "key_hash" not in row
    assert created["key"] not in str(listing)
    assert row["revoked_at"] is None

    with SessionLocal() as db:
        stored = db.get(ApiKey, created["id"])
        assert stored.key_hash != created["key"] and len(stored.key_hash) == 64

    rows = audit_rows(org["id"], "api_key.created")
    assert rows and all(
        created["key"] not in str(r.metadata_json) and stored.key_hash not in str(r.metadata_json)
        for r in rows
    )


def test_expires_at(client, org):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    bad = client.post(
        "/api-keys",
        json={"name": "k", "role": "member", "expires_at": past},
        headers=bearer(org["owner"]),
    )
    assert bad.status_code == 422
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    created = make_key(client, org["owner"], "member", expires_at=future)
    assert created["expires_at"] is not None


def test_revoke_then_key_is_rejected_and_idempotent(client, org):
    target = make_key(client, org["owner"], "member")
    owner = bearer(org["owner"])
    assert client.get("/organization", headers=bearer(target["key"])).status_code == 200

    first = client.delete(f"/api-keys/{target['id']}", headers=owner)
    assert first.status_code == 200 and first.json()["revoked_at"] is not None
    assert "key" not in first.json()
    assert client.get("/organization", headers=bearer(target["key"])).status_code == 401

    second = client.delete(f"/api-keys/{target['id']}", headers=owner)
    assert second.status_code == 200
    assert second.json()["revoked_at"] == first.json()["revoked_at"]
    assert len(audit_rows(org["id"], "api_key.revoked")) == 1

    listing = client.get("/api-keys", headers=owner).json()["items"]
    assert next(k for k in listing if k["id"] == target["id"])["revoked_at"] is not None


def test_cannot_revoke_higher_role(client, org):
    owner_keys = [
        k for k in client.get("/api-keys", headers=bearer(org["owner"])).json()["items"]
        if k["role"] == "owner"
    ]
    response = client.delete(f"/api-keys/{owner_keys[0]['id']}", headers=bearer(org["admin"]))
    assert response.status_code == 403
    assert client.get("/organization", headers=bearer(org["owner"])).status_code == 200


def test_revoking_own_key_is_allowed(client, org):
    admin_id = next(
        k["id"]
        for k in client.get("/api-keys", headers=bearer(org["owner"])).json()["items"]
        if k["role"] == "admin"
    )
    assert client.delete(f"/api-keys/{admin_id}", headers=bearer(org["admin"])).status_code == 200
    assert client.get("/api-keys", headers=bearer(org["admin"])).status_code == 401


def test_cross_org_keys_are_404(client, org):
    other = make_org(client)
    other_owner = other["owner_key"]
    mine = bearer(org["owner"])

    assert client.delete(f"/api-keys/{other_owner['id']}", headers=mine).status_code == 404
    with SessionLocal() as db:
        assert db.get(ApiKey, other_owner["id"]).revoked_at is None

    ids = {k["id"] for k in client.get("/api-keys", headers=mine).json()["items"]}
    assert other_owner["id"] not in ids
    assert client.get("/organization", headers=mine).json()["id"] == org["id"]
    assert client.delete("/api-keys/does-not-exist", headers=mine).status_code == 404


def test_audit_rows_written_for_key_lifecycle(client, org):
    created = make_key(client, org["owner"], "member")
    client.delete(f"/api-keys/{created['id']}", headers=bearer(org["owner"]))
    log = client.get("/audit-log", headers=bearer(org["owner"]), params={"resource_id": created["id"]})
    actions = {item["action"] for item in log.json()["items"]}
    assert actions == {"api_key.created", "api_key.revoked"}

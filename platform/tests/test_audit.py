"""Audit log service and GET /audit-log (organisations plan, T9)."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform import auth, audit
from src.platform.context import AuthContext
from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import (
    DEFAULT_ORGANIZATION_ID,
    ApiKey,
    AuditLog,
    Organization,
)
from src.platform.organizations import ensure_default_org

ADMIN_KEY = "audit-admin-key"
ADMIN = {"Authorization": f"Bearer {ADMIN_KEY}"}


@pytest.fixture(autouse=True)
def schema(monkeypatch):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
        db.query(AuditLog).delete()
        db.commit()
    monkeypatch.setenv("PLATFORM_API_KEYS", ADMIN_KEY)


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


def make_key(org_id: str, role: str) -> str:
    plaintext, prefix, digest = auth.generate_api_key()
    with SessionLocal() as db:
        db.add(
            ApiKey(
                id=str(uuid4()),
                organization_id=org_id,
                name="audit-test",
                key_prefix=prefix,
                key_hash=digest,
                role=role,
            )
        )
        db.commit()
    return plaintext


def api_ctx(org_id: str, **overrides) -> AuthContext:
    values = {
        "org_id": org_id,
        "principal": "api_key",
        "role": "admin",
        "key_id": str(uuid4()),
        "request_id": "req-audit",
    }
    values.update(overrides)
    return AuthContext(**values)


def seed(org_id: str, action: str, *, created_at: datetime, resource_type=None,
         resource_id=None, ctx_overrides=None) -> None:
    """Insert rows via record() with a fixed timestamp for ordering assertions."""

    with SessionLocal() as db:
        entry = audit.record(
            db,
            api_ctx(org_id, **(ctx_overrides or {})),
            action,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        entry.created_at = created_at
        db.commit()


# -- record() ----------------------------------------------------------------------------


def test_record_writes_row_with_context_fields_without_committing() -> None:
    org_id = make_org()
    ctx = api_ctx(org_id, session_id=None)
    with SessionLocal() as db:
        audit.record(
            db,
            ctx,
            "agent.updated",
            resource_type="agent",
            resource_id="agent-1",
            metadata={"fields": ["name"]},
        )
        # Not committed yet: a rollback must discard the audit row too.
        db.rollback()
    with SessionLocal() as db:
        assert db.query(AuditLog).filter_by(organization_id=org_id).count() == 0

    with SessionLocal() as db:
        audit.record(
            db,
            ctx,
            "agent.updated",
            resource_type="agent",
            resource_id="agent-1",
            metadata={"fields": ["name"]},
        )
        db.commit()
    with SessionLocal() as db:
        row = db.query(AuditLog).filter_by(organization_id=org_id).one()
        assert row.action == "agent.updated"
        assert row.actor_type == "api_key"
        assert row.actor_id == ctx.key_id
        assert row.resource_type == "agent"
        assert row.resource_id == "agent-1"
        assert row.metadata_json == {"fields": ["name"]}
        assert row.request_id == "req-audit"
        assert row.created_at is not None


def test_record_actor_id_falls_back_to_session_id() -> None:
    org_id = make_org()
    ctx = AuthContext(
        org_id=org_id, principal="session", role=None, session_id="sess-1", key_id=None
    )
    with SessionLocal() as db:
        audit.record(db, ctx, "session.started")
        db.commit()
    with SessionLocal() as db:
        row = db.query(AuditLog).filter_by(organization_id=org_id).one()
        assert (row.actor_type, row.actor_id) == ("session", "sess-1")
        assert row.metadata_json == {}


def test_record_strips_secret_looking_metadata_keys() -> None:
    org_id = make_org()
    metadata = {
        "name": "voice",
        "api_key": "sk-live-123",
        "provider_secret": "abc",
        "Password": "hunter2",
        "auth_token": "t",
        "nested": {"secret_reference": "ref", "language": "en"},
        "items": [{"apikey": "x", "label": "ok"}],
    }
    with SessionLocal() as db:
        audit.record(db, api_ctx(org_id), "provider.configured", metadata=metadata)
        db.commit()
    with SessionLocal() as db:
        stored = db.query(AuditLog).filter_by(organization_id=org_id).one().metadata_json
    assert stored == {
        "name": "voice",
        "nested": {"language": "en"},
        "items": [{"label": "ok"}],
    }
    assert "sk-live-123" not in repr(stored) and "hunter2" not in repr(stored)


def test_record_does_not_mutate_caller_metadata() -> None:
    org_id = make_org()
    metadata = {"api_key": "sk-1", "keep": 1}
    with SessionLocal() as db:
        audit.record(db, api_ctx(org_id), "x", metadata=metadata)
        db.commit()
    assert metadata == {"api_key": "sk-1", "keep": 1}


# -- GET /audit-log ----------------------------------------------------------------------


def test_list_returns_only_caller_org_rows_newest_first(client) -> None:
    org_id = make_org()
    other_org = make_org()
    key = make_key(org_id, "admin")
    now = datetime.now(timezone.utc)
    seed(org_id, "old", created_at=now - timedelta(minutes=5))
    seed(org_id, "new", created_at=now - timedelta(minutes=1))
    seed(other_org, "foreign", created_at=now)

    response = client.get("/audit-log", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["action"] for item in items] == ["new", "old"]
    assert set(items[0]) == {
        "id",
        "action",
        "actor_type",
        "actor_id",
        "resource_type",
        "resource_id",
        "metadata",
        "request_id",
        "created_at",
    }
    assert items[0]["request_id"] == "req-audit"


def test_list_filters_by_resource_action_and_before_cursor(client) -> None:
    org_id = make_org()
    key = make_key(org_id, "admin")
    now = datetime.now(timezone.utc)
    seed(org_id, "agent.updated", created_at=now - timedelta(minutes=3),
         resource_type="agent", resource_id="a1")
    seed(org_id, "agent.updated", created_at=now - timedelta(minutes=2),
         resource_type="agent", resource_id="a2")
    seed(org_id, "tool.created", created_at=now - timedelta(minutes=1),
         resource_type="tool", resource_id="t1")
    headers = {"Authorization": f"Bearer {key}"}

    by_type = client.get("/audit-log", params={"resource_type": "agent"}, headers=headers)
    assert [i["resource_id"] for i in by_type.json()["items"]] == ["a2", "a1"]

    by_resource = client.get(
        "/audit-log", params={"resource_type": "agent", "resource_id": "a1"}, headers=headers
    )
    assert [i["resource_id"] for i in by_resource.json()["items"]] == ["a1"]

    by_action = client.get("/audit-log", params={"action": "tool.created"}, headers=headers)
    assert [i["action"] for i in by_action.json()["items"]] == ["tool.created"]

    cursor = (now - timedelta(minutes=2)).isoformat()
    before = client.get("/audit-log", params={"before": cursor}, headers=headers)
    assert [i["resource_id"] for i in before.json()["items"]] == ["a1"]


def test_list_limit_defaults_and_caps_at_200(client) -> None:
    org_id = make_org()
    key = make_key(org_id, "admin")
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        for index in range(205):
            entry = audit.record(db, api_ctx(org_id), f"bulk.{index}")
            entry.created_at = now - timedelta(seconds=index)
        db.commit()
    headers = {"Authorization": f"Bearer {key}"}

    default = client.get("/audit-log", headers=headers).json()["items"]
    assert len(default) == 50

    capped = client.get("/audit-log", params={"limit": 200}, headers=headers).json()["items"]
    assert len(capped) == 200

    too_large = client.get("/audit-log", params={"limit": 201}, headers=headers)
    assert too_large.status_code == 422


def test_list_serves_metadata_without_secret_keys(client) -> None:
    org_id = make_org()
    key = make_key(org_id, "admin")
    with SessionLocal() as db:
        audit.record(db, api_ctx(org_id), "x", metadata={"token": "t", "ok": True})
        db.commit()
    items = client.get("/audit-log", headers={"Authorization": f"Bearer {key}"}).json()["items"]
    assert items[0]["metadata"] == {"ok": True}


def test_member_role_is_forbidden(client) -> None:
    org_id = make_org()
    key = make_key(org_id, "member")
    response = client.get("/audit-log", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 403
    assert response.json() == {"detail": {"code": "insufficient_role"}}


def test_unauthenticated_request_is_rejected(client) -> None:
    assert client.get("/audit-log").status_code in (401, 403)


def test_superadmin_sees_default_org_rows(client) -> None:
    seed(DEFAULT_ORGANIZATION_ID, "default.event", created_at=datetime.now(timezone.utc))
    seed(make_org(), "foreign.event", created_at=datetime.now(timezone.utc))
    items = client.get("/audit-log", headers=ADMIN).json()["items"]
    assert [i["action"] for i in items] == ["default.event"]

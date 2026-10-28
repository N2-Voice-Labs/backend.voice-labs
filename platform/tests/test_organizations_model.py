"""Organisation, API key, and audit log models and migration 0008."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.platform.db import Base
from src.platform.models import (
    DEFAULT_ORGANIZATION_ID,
    ORGANIZATION_ROLES,
    ApiKey,
    AuditLog,
    Organization,
)
from src.platform.organizations import ensure_default_org


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_constants() -> None:
    assert DEFAULT_ORGANIZATION_ID == "00000000-0000-0000-0000-000000000001"
    assert ORGANIZATION_ROLES == ("owner", "admin", "member")


def test_models_create_and_defaults(db) -> None:
    org = Organization(name="Acme", slug="acme")
    db.add(org)
    db.commit()
    assert len(org.id) == 36 and org.status == "active"
    assert org.created_at and org.updated_at

    key = ApiKey(
        organization_id=org.id,
        name="ci",
        key_prefix="vlk_abcd",
        key_hash="a" * 64,
        role="admin",
    )
    audit = AuditLog(
        organization_id="not-a-real-org",
        actor_type="api_key",
        action="agent.create",
        metadata_json={"k": "v"},
    )
    db.add_all([key, audit])
    db.commit()
    assert key.revoked_at is None and key.last_used_at is None
    assert db.get(AuditLog, audit.id).metadata_json == {"k": "v"}  # no FK on audit_log


def test_unique_slug_and_key_hash(db) -> None:
    db.add(Organization(name="A", slug="dup"))
    db.commit()
    db.add(Organization(name="B", slug="dup"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    org = ensure_default_org(db)
    for _ in range(2):
        db.add(
            ApiKey(
                organization_id=org.id,
                name="k",
                key_prefix="vlk_",
                key_hash="b" * 64,
                role="member",
            )
        )
    with pytest.raises(IntegrityError):
        db.commit()


def test_indexes_declared() -> None:
    api = {i.name for i in ApiKey.__table__.indexes}
    audit = {i.name for i in AuditLog.__table__.indexes}
    assert "ix_api_keys_organization_id_revoked_at" in api
    assert "ix_api_keys_organization_id" in api
    assert {
        "ix_audit_log_organization_id_created_at",
        "ix_audit_log_resource_type_resource_id",
    } <= audit


def test_ensure_default_org_is_idempotent(db) -> None:
    first = ensure_default_org(db)
    second = ensure_default_org(db)
    assert first.id == second.id == DEFAULT_ORGANIZATION_ID
    rows = db.scalars(select(Organization)).all()
    assert len(rows) == 1
    assert (rows[0].name, rows[0].slug) == ("Default", "default")


def test_ensure_default_org_keeps_existing_edits(db) -> None:
    ensure_default_org(db)
    org = db.get(Organization, DEFAULT_ORGANIZATION_ID)
    org.name = "Renamed"
    db.commit()
    assert ensure_default_org(db).name == "Renamed"


# -- migration 0008 ----------------------------------------------------------------------


def run_alembic(db_url: str, *args: str) -> None:
    platform_dir = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PLATFORM_DATABASE_URL": db_url,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("PLATFORM_API_KEYS", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=platform_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


NEW_TABLES = {"organizations", "api_keys", "audit_log"}


def test_migration_0008_up_and_down_on_sqlite() -> None:
    cache = Path.home() / ".cache" / "voice-labs-test"
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / f"orgs-migration-{uuid4().hex}.db"
    url = f"sqlite:///{db_path}"
    try:
        run_alembic(url, "upgrade", "0007_session_tokens")
        engine = create_engine(url)
        assert not NEW_TABLES & set(inspect(engine).get_table_names())

        run_alembic(url, "upgrade", "0008_organizations")
        insp = inspect(engine)
        assert NEW_TABLES <= set(insp.get_table_names())
        index_names = {i["name"] for t in ("api_keys", "audit_log") for i in insp.get_indexes(t)}
        assert {
            "ix_api_keys_organization_id",
            "ix_api_keys_organization_id_revoked_at",
            "ix_audit_log_organization_id_created_at",
            "ix_audit_log_resource_type_resource_id",
        } <= index_names
        assert {c["name"] for c in insp.get_columns("audit_log")} >= {"metadata", "request_id"}
        assert insp.get_foreign_keys("audit_log") == []
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT id, name, slug, status FROM organizations")
            ).all()
        assert [tuple(r) for r in row] == [
            (DEFAULT_ORGANIZATION_ID, "Default", "default", "active")
        ]

        # ensure_default_org after the migration is a no-op.
        session = sessionmaker(bind=engine)()
        ensure_default_org(session)
        assert len(session.scalars(select(Organization)).all()) == 1
        session.close()

        run_alembic(url, "downgrade", "0007_session_tokens")
        assert not NEW_TABLES & set(inspect(engine).get_table_names())
        engine.dispose()
    finally:
        db_path.unlink(missing_ok=True)

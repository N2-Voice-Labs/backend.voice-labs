"""Tenant columns: migrations 0009/0010, constraints, and the before_flush stamping hook."""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.platform.db import Base
from src.platform.models import (
    DEFAULT_ORGANIZATION_ID,
    Agent,
    Deployment,
    Organization,
    Tool,
)

OTHER_ORG = "00000000-0000-0000-0000-0000000000aa"
NOW = datetime.now(timezone.utc)
TENANT_TABLES = (
    "agents", "provider_configs", "conversations", "messages", "voice_sessions",
    "runtime_events", "voices", "deployments", "campaigns", "campaign_targets",
    "tools", "knowledge_documents",
)


def run_alembic(db_url: str, *args: str) -> None:
    platform_dir = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PLATFORM_DATABASE_URL": db_url, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PLATFORM_API_KEYS", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=platform_dir, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def _seed(conn) -> None:
    ts = {"now": NOW}
    conn.execute(text("INSERT INTO provider_configs (id, kind, provider, configuration, enabled, created_at) VALUES ('p1','llm','x','{}',1,:now)"), ts)
    conn.execute(text("INSERT INTO agents (id, name, description, system_prompt, enabled_languages, mode, status, created_at, updated_at) VALUES ('a1','A','','','[]','agent','enabled',:now,:now)"), ts)
    conn.execute(text("INSERT INTO conversations (id, agent_id, status, started_at, metadata, channel, actions) VALUES ('c1','a1','active',:now,'{}','web','[]')"), ts)
    conn.execute(text("INSERT INTO messages (id, conversation_id, role, content, detected_languages, created_at) VALUES ('m1','c1','user','hi','[]',:now)"), ts)
    conn.execute(text("INSERT INTO tools (id, name, description, tool_type, input_schema, configuration, enabled, created_at, updated_at) VALUES ('t1','lookup','d','http','{}','{}',1,:now,:now)"), ts)
    conn.execute(text("INSERT INTO deployments (id, agent_id, name, channel, status, phone_number, configuration, created_at, updated_at) VALUES ('d1','a1','D','sip','active','+1555',  '{}',:now,:now)"), ts)


@pytest.fixture
def sqlite_url():
    cache = Path.home() / ".cache" / "voice-labs-test"
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / f"tenancy-migration-{uuid4().hex}.db"
    try:
        yield f"sqlite:///{db_path}"
    finally:
        db_path.unlink(missing_ok=True)


def test_migrations_0009_0010_backfill_and_roundtrip(sqlite_url) -> None:
    run_alembic(sqlite_url, "upgrade", "0008_organizations")
    engine = create_engine(sqlite_url)
    with engine.begin() as conn:
        _seed(conn)

    run_alembic(sqlite_url, "upgrade", "0010_org_not_null")
    insp = inspect(engine)
    for table in TENANT_TABLES:
        col = next(c for c in insp.get_columns(table) if c["name"] == "organization_id")
        assert col["nullable"] is False, table
        assert "ix_%s_organization_id" % table in {i["name"] for i in insp.get_indexes(table)}
        assert any(
            fk["referred_table"] == "organizations" and fk["constrained_columns"] == ["organization_id"]
            for fk in insp.get_foreign_keys(table)
        ), table
    with engine.connect() as conn:
        for table in ("provider_configs", "agents", "conversations", "messages", "tools", "deployments"):
            orgs = conn.execute(text(f"SELECT DISTINCT organization_id FROM {table}")).all()
            assert [tuple(r) for r in orgs] == [(DEFAULT_ORGANIZATION_ID,)], table
        assert conn.execute(text("SELECT count(*) FROM organizations")).scalar() == 1

    run_alembic(sqlite_url, "downgrade", "0008_organizations")
    insp = inspect(engine)
    for table in TENANT_TABLES:
        assert "organization_id" not in {c["name"] for c in insp.get_columns(table)}, table
    assert "uq_deployments_active_phone_number" not in {i["name"] for i in insp.get_indexes("deployments")}
    with engine.begin() as conn:
        # Global tool-name uniqueness is back.
        with pytest.raises(IntegrityError):
            conn.execute(text("INSERT INTO tools (id, name, description, tool_type, input_schema, configuration, enabled, created_at, updated_at) VALUES ('t2','lookup','d','http','{}','{}',1,:now,:now)"), {"now": NOW})
    # And it can go back up again.
    run_alembic(sqlite_url, "upgrade", "head")
    engine.dispose()


def test_migration_0010_refuses_duplicate_active_phone_numbers(sqlite_url) -> None:
    run_alembic(sqlite_url, "upgrade", "0008_organizations")
    engine = create_engine(sqlite_url)
    with engine.begin() as conn:
        _seed(conn)
        conn.execute(text("INSERT INTO deployments (id, agent_id, name, channel, status, phone_number, configuration, created_at, updated_at) VALUES ('d2','a1','D2','sip','active','+1555','{}',:now,:now)"), {"now": NOW})
        # Inactive / null duplicates are fine and must not be reported.
        conn.execute(text("INSERT INTO deployments (id, agent_id, name, channel, status, phone_number, configuration, created_at, updated_at) VALUES ('d3','a1','D3','sip','disabled','+1555','{}',:now,:now)"), {"now": NOW})
    platform_dir = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PLATFORM_DATABASE_URL": sqlite_url, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PLATFORM_API_KEYS", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "0010_org_not_null"],
        cwd=platform_dir, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert "+1555" in result.stderr and "duplicate" in result.stderr.lower()
    # Resolve the duplicate and the upgrade goes through.
    with engine.begin() as conn:
        conn.execute(text("UPDATE deployments SET status='disabled' WHERE id='d2'"))
    run_alembic(sqlite_url, "upgrade", "0010_org_not_null")
    engine.dispose()


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        Organization(id=DEFAULT_ORGANIZATION_ID, name="Default", slug="default"),
        Organization(id=OTHER_ORG, name="Other", slug="other"),
    ])
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _tool(name: str, org: str | None = None) -> Tool:
    return Tool(name=name, description="d", organization_id=org)


def test_tool_name_unique_per_org(db) -> None:
    db.add_all([_tool("lookup", DEFAULT_ORGANIZATION_ID), _tool("lookup", OTHER_ORG)])
    db.commit()
    db.add(_tool("lookup", OTHER_ORG))
    with pytest.raises(IntegrityError):
        db.commit()


def _deployment(agent_id: str, phone: str | None, status: str = "active") -> Deployment:
    return Deployment(agent_id=agent_id, name="d", channel="sip", status=status, phone_number=phone)


def test_active_deployment_phone_numbers_globally_unique(db) -> None:
    agent = Agent(name="a")
    db.add(agent)
    db.commit()
    db.add(_deployment(agent.id, "+1555"))
    db.commit()
    # Inactive duplicates and null numbers are fine.
    db.add_all([_deployment(agent.id, "+1555", "paused"), _deployment(agent.id, None), _deployment(agent.id, None)])
    db.commit()
    db.add(_deployment(agent.id, "+1555"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    # Another org cannot take an active number either.
    other = Agent(name="b", organization_id=OTHER_ORG)
    db.add(other)
    db.commit()
    db.add(Deployment(agent_id=other.id, name="d", channel="sip", phone_number="+1555", organization_id=OTHER_ORG))
    with pytest.raises(IntegrityError):
        db.commit()


def test_stamp_defaults_to_default_org(db) -> None:
    agent = Agent(name="a")
    db.add(agent)
    db.commit()
    assert agent.organization_id == DEFAULT_ORGANIZATION_ID


def test_stamp_honours_session_org(db) -> None:
    db.info["org_id"] = OTHER_ORG
    agent = Agent(name="a")
    explicit = Agent(name="b", organization_id=OTHER_ORG)
    db.add_all([agent, explicit])
    db.commit()
    assert agent.organization_id == OTHER_ORG
    assert explicit.organization_id == OTHER_ORG


def test_stamp_mismatch_raises(db) -> None:
    db.info["org_id"] = OTHER_ORG
    db.add(Agent(name="a", organization_id=DEFAULT_ORGANIZATION_ID))
    with pytest.raises(ValueError):
        db.flush()


def test_explicit_org_kept_without_session_org(db) -> None:
    agent = Agent(name="a", organization_id=OTHER_ORG)
    db.add(agent)
    db.commit()
    assert agent.organization_id == OTHER_ORG

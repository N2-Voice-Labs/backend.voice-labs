"""Tenancy safety net: read filtering, stamping, bypass, get_owned, link_check."""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import ForeignKey, String, create_engine, delete, func, select, text, update
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from src.platform import db as db_module
from src.platform.context import AuthContext
from src.platform.db import Base
from src.platform.models import (
    DEFAULT_ORGANIZATION_ID,
    Agent,
    Organization,
    TenantMixin,
    VoiceSession,
)
from src.platform.tenancy import (
    bypass,
    get_owned,
    get_scoped_db,
    link_check,
    not_found_detail_for,
    scoped,
    scoped_session_for,
)

ORG_A = "aaaaaaaa-0000-0000-0000-000000000001"
ORG_B = "bbbbbbbb-0000-0000-0000-000000000002"


def ctx_for(org_id: str | None, principal: str = "api_key") -> AuthContext:
    return AuthContext(org_id=org_id, principal=principal, role="admin")


# Relationship models live on their own metadata so the shared Base is not polluted.
class RelBase(DeclarativeBase):
    pass


class RelOrg(RelBase):
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)


class Parent(TenantMixin, RelBase):
    __tablename__ = "rel_parents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    children: Mapped[list["Child"]] = relationship(back_populates="parent")


class Child(TenantMixin, RelBase):
    __tablename__ = "rel_children"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_id: Mapped[str] = mapped_column(ForeignKey("rel_parents.id"))
    parent: Mapped[Parent] = relationship(back_populates="children")


@pytest.fixture
def engine(monkeypatch):
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(eng)
    factory = sessionmaker(bind=eng, autoflush=False)
    monkeypatch.setattr(db_module, "SessionLocal", factory)
    with factory() as s:
        s.add_all(
            [
                Organization(id=DEFAULT_ORGANIZATION_ID, name="d", slug="default"),
                Organization(id=ORG_A, name="A", slug="a"),
                Organization(id=ORG_B, name="B", slug="b"),
            ]
        )
        s.commit()
        for org, aid in ((ORG_A, "agent-a"), (ORG_B, "agent-b")):
            s.add(Agent(id=aid, name=aid, organization_id=org))
        s.commit()
    yield eng
    eng.dispose()


@pytest.fixture
def factory(engine):
    return db_module.SessionLocal


def test_select_filtered_for_scoped_session(factory) -> None:
    with factory() as db:
        assert {a.id for a in db.scalars(select(Agent))} == {"agent-a", "agent-b"}
    with scoped_session_for(ORG_A) as db:
        assert [a.id for a in db.scalars(select(Agent))] == ["agent-a"]
        assert db.scalars(select(Agent).where(Agent.id == "agent-b")).first() is None


def test_aliased_and_aggregate_selects_filtered(factory) -> None:
    from sqlalchemy.orm import aliased

    with scoped_session_for(ORG_A) as db:
        alias = aliased(Agent)
        assert [a.id for a in db.scalars(select(alias))] == ["agent-a"]
        # Entity-based aggregates are ORM-enabled and therefore filtered...
        assert db.scalar(select(func.count(Agent.id))) == 1
        assert db.scalar(select(func.count()).select_from(Agent)) == 1
        # ...but pure Core / text() statements are not: use scoped() or get_owned there.
        assert db.execute(text("select count(*) from agents")).scalar() == 2
        assert db.scalar(select(func.count()).select_from(Agent.__table__)) == 2


def test_session_get_identity_map_caveat(factory) -> None:
    # Cold identity map: Session.get emits SQL, so the criteria applies.
    with scoped_session_for(ORG_A) as db:
        assert db.get(Agent, "agent-b") is None
        assert db.get(Agent, "agent-a") is not None

    # Warm identity map: a row loaded under bypass is returned by Session.get without
    # SQL, so the criteria is NOT applied. get_owned must not rely on Session.get.
    with scoped_session_for(ORG_A) as db:
        with bypass(db):
            loaded = db.scalars(select(Agent)).all()  # keep refs: identity map is weak
        assert len(loaded) == 2
        assert db.get(Agent, "agent-b") is not None  # documented leak
        with pytest.raises(HTTPException) as exc:
            get_owned(db, ctx_for(ORG_A), Agent, "agent-b")
        assert exc.value.status_code == 404


def test_relationship_loads_filtered(factory, engine) -> None:
    RelBase.metadata.create_all(engine)
    with factory() as s:
        s.add(Parent(id="p1", organization_id=ORG_A))
        s.commit()
        # A child that (illegally) lives in another org under the same parent.
        s.add(Child(id="c-a", parent_id="p1", organization_id=ORG_A))
        s.add(Child(id="c-b", parent_id="p1", organization_id=ORG_B))
        s.commit()

    with scoped_session_for(ORG_A) as db:
        parent = db.scalars(select(Parent)).one()
        assert [c.id for c in parent.children] == ["c-a"]  # lazy load filtered
    with scoped_session_for(ORG_A) as db:
        parent = db.scalars(select(Parent).options(selectinload(Parent.children))).one()
        assert [c.id for c in parent.children] == ["c-a"]  # eager load filtered
    with scoped_session_for(ORG_B) as db:
        assert db.scalars(select(Parent)).first() is None
        child = db.scalars(select(Child)).one()
        assert child.id == "c-b"
        assert child.parent is None  # many-to-one lazy load into a foreign org
    with factory() as db:
        assert len(db.scalars(select(Parent)).one().children) == 2


def test_bulk_update_and_delete_filtered(factory) -> None:
    with scoped_session_for(ORG_A) as db:
        res = db.execute(update(Agent).values(name="renamed"))
        assert res.rowcount == 1
        res = db.execute(delete(Agent).where(Agent.id == "agent-b"))
        assert res.rowcount == 0
        db.commit()
    with factory() as db:
        names = {a.id: a.name for a in db.scalars(select(Agent))}
        assert names == {"agent-a": "renamed", "agent-b": "agent-b"}

    with scoped_session_for(ORG_A) as db:
        assert db.execute(delete(Agent)).rowcount == 1
        db.commit()
    with factory() as db:
        assert [a.id for a in db.scalars(select(Agent))] == ["agent-b"]


def test_orm_unit_of_work_update_delete(factory) -> None:
    with scoped_session_for(ORG_A) as db:
        agent = db.scalars(select(Agent)).one()
        agent.name = "changed"
        db.commit()
        db.delete(agent)
        db.commit()
    with factory() as db:
        assert [a.id for a in db.scalars(select(Agent))] == ["agent-b"]


def test_bypass_sees_everything_and_restores(factory) -> None:
    with scoped_session_for(ORG_A) as db:
        with bypass(db):
            assert {a.id for a in db.scalars(select(Agent))} == {"agent-a", "agent-b"}
            with bypass(db):
                pass
            assert db.info["tenant_bypass"] is True  # nested exit keeps outer bypass
        assert "tenant_bypass" not in db.info
        db.expire_all()
        assert [a.id for a in db.scalars(select(Agent))] == ["agent-a"]


def test_stamping(factory) -> None:
    with factory() as db:
        db.add(Agent(id="x0", name="x0"))
        db.commit()
        assert db.get(Agent, "x0").organization_id == DEFAULT_ORGANIZATION_ID
    with scoped_session_for(ORG_A) as db:
        db.add(Agent(id="x1", name="x1"))
        db.commit()
        assert db.get(Agent, "x1").organization_id == ORG_A
        db.add(Agent(id="x2", name="x2", organization_id=ORG_B))
        with pytest.raises(ValueError):
            db.commit()
        db.rollback()
        with bypass(db):
            db.add(Agent(id="x3", name="x3", organization_id=ORG_B))
            db.commit()
    with factory() as db:
        assert db.get(Agent, "x3").organization_id == ORG_B


def test_get_owned(factory) -> None:
    with factory() as db:
        assert get_owned(db, ctx_for(ORG_A), Agent, "agent-a").id == "agent-a"
        with pytest.raises(HTTPException) as exc:
            get_owned(db, ctx_for(ORG_A), Agent, "agent-b")
        assert (exc.value.status_code, exc.value.detail) == (404, "agent_not_found")
        with pytest.raises(HTTPException) as exc:
            get_owned(db, ctx_for(ORG_A), Agent, "missing", not_found_detail="gone")
        assert exc.value.detail == "gone"
        # No org on the ctx (superadmin w/o org) -> unscoped.
        assert get_owned(db, ctx_for(None, "runtime"), Agent, "agent-b").id == "agent-b"


def test_not_found_detail_naming() -> None:
    assert not_found_detail_for(Agent) == "agent_not_found"
    assert not_found_detail_for(VoiceSession) == "voice_session_not_found"


def test_link_check(factory) -> None:
    with factory() as db:
        a = db.get(Agent, "agent-a")
        b = db.get(Agent, "agent-b")
    link_check(ctx_for(ORG_A), a, None)
    link_check(ctx_for(None, "runtime"), b)
    with pytest.raises(HTTPException) as exc:
        link_check(ctx_for(ORG_A), a, b)
    assert (exc.value.status_code, exc.value.detail) == (404, "agent_not_found")


def test_scoped_statement(factory) -> None:
    with factory() as db:
        rows = db.scalars(scoped(db, ctx_for(ORG_B), Agent)).all()
        assert [r.id for r in rows] == ["agent-b"]
        assert len(db.scalars(scoped(db, ctx_for(None, "runtime"), Agent)).all()) == 2
        # Works for Core-level aggregates on an unfiltered session.
        stmt = scoped(db, ctx_for(ORG_A), Agent).with_only_columns(func.count(Agent.id))
        assert db.scalar(stmt) == 1


def test_get_scoped_db_dependency(factory) -> None:
    app = FastAPI()

    @app.middleware("http")
    async def set_ctx(request, call_next):
        org = request.headers.get("x-org")
        request.state.ctx = ctx_for(org) if org else ctx_for(None, "runtime")
        return await call_next(request)

    @app.get("/agents")
    def list_agents(db=Depends(get_scoped_db)):
        return sorted(a.id for a in db.scalars(select(Agent)))

    @app.get("/info")
    def info(db=Depends(get_scoped_db)):
        return {"org": db.info.get("org_id")}

    def override_db():
        yield from _gen(factory)

    app.dependency_overrides[db_module.get_db] = override_db
    client = TestClient(app)
    assert client.get("/agents", headers={"x-org": ORG_A}).json() == ["agent-a"]
    assert client.get("/info", headers={"x-org": ORG_B}).json() == {"org": ORG_B}
    assert client.get("/agents").json() == ["agent-a", "agent-b"]  # runtime: unfiltered


def _gen(factory):
    db = factory()
    try:
        yield db
    finally:
        db.close()

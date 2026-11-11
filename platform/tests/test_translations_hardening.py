"""Hardening of ``/translations``: race-free hidden agent, hidden-agent isolation through the
public agent routes, authz and limits, explicit interpreter contract, legacy compatibility."""

import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from src.platform import limits, translations
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import Agent, VoiceSession

from tenancy_fixtures import bearer, make_client, make_org_with_keys
from test_interpreter_participants import runtime  # noqa: F401  (fixture)
from test_translations import contract_for, start


@pytest.fixture
def client(monkeypatch):
    with make_client(monkeypatch) as test_client:
        yield test_client


@pytest.fixture
def org(client):
    return make_org_with_keys(client, "tr")


def hidden_agent_id(session_id: str) -> str:
    return contract_for(session_id)["agent_id"]


def system_agents(org_id: str | None = None) -> list[Agent]:
    with SessionLocal() as db:
        stmt = select(Agent).where(Agent.system.is_(True))
        if org_id:
            stmt = stmt.where(Agent.organization_id == org_id)
        return list(db.scalars(stmt).all())


# ------------------------------------------------------------- 1. concurrency


def test_system_agent_race_loser_reselects_winner(client, org, monkeypatch) -> None:
    first = client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    )
    assert first.status_code == 201, first.text
    assert len(system_agents(org.org_id)) == 1

    # The racing request looked up before the winner committed: its first lookup misses.
    real_find = translations._find_system_agent
    calls = {"n": 0}

    def stale_find(db, ctx, key):
        calls["n"] += 1
        return None if calls["n"] == 1 else real_find(db, ctx, key)

    monkeypatch.setattr(translations, "_find_system_agent", stale_find)
    second = client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    )
    assert second.status_code == 201, second.text
    assert calls["n"] == 2  # miss, insert collided, re-selected
    agents = system_agents(org.org_id)
    assert len(agents) == 1
    assert hidden_agent_id(first.json()["session_id"]) == agents[0].id
    assert hidden_agent_id(second.json()["session_id"]) == agents[0].id


def test_system_agent_unique_per_org_and_key(client, org) -> None:
    from sqlalchemy.exc import IntegrityError

    client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    )
    existing = system_agents(org.org_id)[0]
    with SessionLocal() as db:
        db.add(
            Agent(
                name="dup",
                system=True,
                system_key=existing.system_key,
                organization_id=org.org_id,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
    # Ordinary agents (NULL system_key) are unaffected by the constraint.
    for _ in range(2):
        assert client.post("/agents", json={"name": "plain"}, headers=org.admin).status_code == 201


# ------------------------------------------------- 2. hidden agent isolation

SYSTEM_ID = "{system}"


def _routes():
    return [
        ("GET", "/agents/{system}", None, 404),
        ("PATCH", "/agents/{system}", {"name": "hijack"}, 404),
        ("DELETE", "/agents/{system}", None, 404),
        ("PUT", "/agents/{system}/tools", {"tool_ids": []}, 404),
        ("GET", "/agents/{system}/tools", None, 404),
        ("PUT", "/agents/{system}/knowledge", {"document_ids": []}, 404),
        ("GET", "/agents/{system}/knowledge", None, 404),
        ("POST", "/voice/sessions", {"agent_id": "{system}"}, 404),
        ("POST", "/deployments", {"agent_id": "{system}", "name": "d", "channel": "web"}, 422),
        (
            "POST",
            "/campaigns",
            {"name": "c", "agent_id": "{system}", "deployment_id": "00000000-0000-0000-0000-000000000001"},
            422,
        ),
    ]


@pytest.mark.parametrize("method,path,body,expected", _routes())
def test_system_agent_is_invisible_through_public_routes(
    client, org, method, path, body, expected
) -> None:
    created = client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    )
    assert created.status_code == 201, created.text
    agent_id = hidden_agent_id(created.json()["session_id"])
    if body is not None:
        body = {k: (agent_id if v == SYSTEM_ID else v) for k, v in body.items()}
    response = client.request(
        method, path.format(system=agent_id), json=body, headers=org.admin
    )
    assert response.status_code == expected, response.text
    detail = response.json()["detail"]
    assert detail in ("agent_not_found", "invalid_agent_id")
    # And the missing-id response is identical, so existence is not leaked.
    missing = str(uuid4())
    twin_body = None if body is None else {k: (missing if v == agent_id else v) for k, v in body.items()}
    twin = client.request(method, path.format(system=missing), json=twin_body, headers=org.admin)
    assert (twin.status_code, twin.json()["detail"]) == (response.status_code, detail)
    # Nothing was mutated.
    with SessionLocal() as db:
        row = db.get(Agent, agent_id)
        assert row.status == "enabled" and row.name != "hijack"


def test_system_agent_not_listed_and_never_in_templates(client, org) -> None:
    client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    )
    assert client.get("/agents", headers=org.member).json() == []
    # Template creation only ever makes ordinary agents.
    templates = client.get("/agent-templates", headers=org.member).json()
    key = templates[0]["key"] if isinstance(templates, list) else next(iter(templates))
    made = client.post(f"/agent-templates/{key}/agents", json={}, headers=org.admin)
    assert made.status_code == 201, made.text
    with SessionLocal() as db:
        assert db.get(Agent, made.json()["id"]).system is False


# ---------------------------------------- 3. GET /translations/{id} scoping


def test_get_translation_rejects_ordinary_voice_sessions(client, org) -> None:
    plain = client.post(
        "/agents",
        json={
            "name": "interp",
            "mode": "interpreter",
            "interpreter": {"languages": ["en", "ko"], "direction": "one_way"},
        },
        headers=org.admin,
    ).json()
    ordinary = client.post(
        "/voice/sessions", json={"agent_id": plain["id"]}, headers=org.member
    )
    assert ordinary.status_code == 201, ordinary.text
    sid = ordinary.json()["session_id"]
    assert client.get(f"/translations/{sid}", headers=org.member).status_code == 404
    assert client.post(
        f"/translations/{sid}/invite", json={"language": "en"}, headers=org.member
    ).status_code == 404
    # The ordinary route still serves it.
    assert client.get(f"/voice/sessions/{sid}", headers=org.member).status_code == 200


def test_session_token_reads_own_translation(client, org) -> None:
    created = client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    ).json()
    sid, token = created["session_id"], created["session_token"]
    assert client.get(f"/translations/{sid}", headers=bearer(token)).status_code == 200
    invited = client.post(
        f"/translations/{sid}/invite", json={"language": "en"}, headers=bearer(token)
    )
    assert invited.status_code == 200, invited.text
    guest = bearer(invited.json()["session_token"])
    assert client.get(f"/translations/{sid}", headers=guest).status_code == 200


# --------------------------------------------------- 4. authz and limits


def test_member_key_can_create_list_get_invite(client, org) -> None:
    created = client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    )
    assert created.status_code == 201, created.text
    sid = created.json()["session_id"]
    assert sid in {i["id"] for i in client.get("/translations", headers=org.member).json()}
    assert client.get(f"/translations/{sid}", headers=org.member).status_code == 200
    assert client.post(
        f"/translations/{sid}/invite", json={"language": "en"}, headers=org.member
    ).status_code == 200


def test_no_credentials_is_401(client, org) -> None:
    sid = client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    ).json()["session_id"]
    body = {"languages": ["ko", "en"], "creator": {"language": "ko"}}
    assert client.post("/translations", json=body).status_code == 401
    assert client.get("/translations").status_code == 401
    assert client.get(f"/translations/{sid}").status_code == 401
    assert client.post(f"/translations/{sid}/invite", json={"language": "en"}).status_code == 401


def test_guest_cannot_invite_and_session_a_cannot_read_b(client, org) -> None:
    def create():
        return client.post(
            "/translations",
            json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
            headers=org.member,
        ).json()

    a, b = create(), create()
    invited = client.post(
        f"/translations/{a['session_id']}/invite", json={"language": "en"}, headers=bearer(a["session_token"])
    ).json()
    guest = bearer(invited["session_token"])
    third = client.post(
        f"/translations/{a['session_id']}/invite", json={"language": "en"}, headers=guest
    )
    assert third.status_code == 403
    assert client.get(f"/translations/{b['session_id']}", headers=bearer(a["session_token"])).status_code == 403
    assert client.get(f"/translations/{b['session_id']}", headers=guest).status_code == 403
    assert client.get("/translations", headers=bearer(a["session_token"])).status_code == 403


def test_other_org_gets_404_on_get_and_invite(client, org) -> None:
    other = make_org_with_keys(client, "other")
    sid = client.post(
        "/translations",
        json={"languages": ["ko", "en"], "creator": {"language": "ko"}},
        headers=org.member,
    ).json()["session_id"]
    for headers in (other.member, other.admin, other.owner):
        assert client.get(f"/translations/{sid}", headers=headers).status_code == 404
        assert client.post(
            f"/translations/{sid}/invite", json={"language": "en"}, headers=headers
        ).status_code == 404
        assert sid not in {i["id"] for i in client.get("/translations", headers=headers).json()}


def test_active_session_cap_returns_429(client, org, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG", "1")
    body = {"languages": ["ko", "en"], "creator": {"language": "ko"}}
    assert client.post("/translations", json=body, headers=org.member).status_code == 201
    blocked = client.post("/translations", json=body, headers=org.member)
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["code"] == "too_many_active_sessions"


def test_rate_limit_returns_429(client, org, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_SESSION_CREATE_RATE", "1")
    limits.session_create_bucket.reset()
    body = {"languages": ["ko", "en"], "creator": {"language": "ko"}}
    try:
        assert client.post("/translations", json=body, headers=org.member).status_code == 201
        limited = client.post("/translations", json=body, headers=org.member)
        assert limited.status_code == 429
        assert limited.json()["detail"]["code"] == "rate_limited"
        assert int(limited.headers["Retry-After"]) >= 1
    finally:
        limits.session_create_bucket.reset()


# ------------------------------------------------------ 5. explicit mode


def test_translation_contract_and_runtime_payload_are_explicit_interpreter(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as plain:
        response = start(plain)
        assert response.status_code == 201, response.text
        sid = response.json()["session_id"]
    stored = contract_for(sid)
    sent = [r for r in requests if r["url"].endswith("/sessions")][-1]["json"]
    for contract in (stored, sent):
        assert contract["mode"] == "interpreter"
        assert contract["system_prompt"] == ""
        assert contract["tools"] == []
        assert contract["knowledge_enabled"] is False
    assert sent == stored


# ------------------------------------------------------- 6. compatibility


def test_legacy_interpreter_agent_with_prompt_still_sends_it(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as plain:
        agent = plain.post(
            "/agents",
            json={
                "name": f"legacy-{uuid4()}",
                "mode": "interpreter",
                "system_prompt": "Keep a formal register.",
                "interpreter": {"languages": ["en", "ko"], "direction": "one_way"},
            },
        )
        assert agent.status_code == 201, agent.text
        session = plain.post("/voice/sessions", json={"agent_id": agent.json()["id"]})
        assert session.status_code == 201, session.text
        sid = session.json()["session_id"]
    assert contract_for(sid)["system_prompt"] == "Keep a formal register."
    sent = [r for r in requests if r["url"].endswith("/sessions")][-1]["json"]
    assert sent["mode"] == "interpreter"
    assert sent["system_prompt"] == "Keep a formal register."


def test_system_agent_race_real_threads_file_sqlite(tmp_path) -> None:
    """N threads all miss the lookup, then insert the same ``(org, system_key)`` at once.

    Real concurrency on a file-backed SQLite (own engine, one Session per thread, a
    barrier between lookup and insert): exactly one row, no exception, every thread gets
    the winner. NOT covered: PostgreSQL concurrency (row/unique-index locking and
    SAVEPOINT behaviour differ); that needs a Postgres integration run.
    """

    import threading

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.platform.context import AuthContext
    from src.platform.db import Base
    from src.platform.models import Organization

    engine = create_engine(
        f"sqlite:///{tmp_path / 'race.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(bind=engine)
    maker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with maker() as db:
        db.add(Organization(id="org-race", name="Race", slug="race"))
        db.commit()
    ctx = AuthContext(principal="member", role="member", org_id="org-race")
    interpreter = {"languages": ["ko", "en"], "engine": "cascaded"}
    threads_n = 8
    barrier = threading.Barrier(threads_n)
    real_find = translations._find_system_agent
    results: list[str] = []
    errors: list[BaseException] = []

    def racing_find(db, ctx, key):
        found = real_find(db, ctx, key)
        if found is None and not getattr(db, "_synced", False):
            db._synced = True  # only the first (pre-insert) lookup waits
            barrier.wait(timeout=10)  # everyone has missed; now all insert
        return found

    def worker() -> None:
        try:
            with maker() as db:
                db.info["org_id"] = "org-race"
                agent = translations.system_agent_for(db, ctx, interpreter)
                db.commit()
                results.append(agent.id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    translations_find = translations._find_system_agent
    translations._find_system_agent = racing_find
    try:
        threads = [threading.Thread(target=worker) for _ in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
    finally:
        translations._find_system_agent = translations_find
    assert not errors, errors
    assert len(results) == threads_n and len(set(results)) == 1
    with maker() as db:
        assert db.scalar(select(func.count()).select_from(Agent)) == 1
    engine.dispose()

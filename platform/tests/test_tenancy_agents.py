"""Organisation scoping of agent, provider and agent-template routes (T5a)."""

import os
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import Voice
from src.platform.organizations import ensure_default_org

SUPER_KEY = "super-key-t5a"
SUPER = {"Authorization": f"Bearer {SUPER_KEY}"}

MISSING = "00000000-0000-0000-0000-0000000000aa"

# (method, path) pairs covered by this module, for the isolation completeness check.
ISOLATION_CASES = [
    ("POST", "/agents"),
    ("GET", "/agents"),
    ("GET", "/agents/{agent_id}"),
    ("PATCH", "/agents/{agent_id}"),
    ("DELETE", "/agents/{agent_id}"),
    ("POST", "/providers"),
    ("GET", "/providers"),
    ("GET", "/agent-templates"),
    ("GET", "/agent-templates/{key}"),
    ("POST", "/agent-templates/{key}/agents"),
]


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


def make_org(client) -> dict:
    created = client.post(
        "/organizations",
        json={
            "name": "Org",
            "slug": f"t5a-{uuid4().hex[:10]}",
            "create_owner_key": True,
        },
        headers=SUPER,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    owner = body["owner_key"]["key"]
    keys = {"id": body["id"], "owner": bearer(owner)}
    for role in ("admin", "member"):
        response = client.post(
            "/api-keys",
            json={"name": role, "role": role},
            headers=bearer(owner),
        )
        assert response.status_code == 201, response.text
        keys[role] = bearer(response.json()["key"])
    return keys


@pytest.fixture
def org_a(client):
    return make_org(client)


@pytest.fixture
def org_b(client):
    return make_org(client)


def make_provider(client, headers, kind="llm") -> dict:
    response = client.post(
        "/providers",
        json={"kind": kind, "provider": "openai", "configuration": {}},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_agent(client, headers, **extra) -> dict:
    response = client.post("/agents", json={"name": "A", **extra}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def make_voice(org_id: str, **fields) -> str:
    with SessionLocal() as db:
        voice = Voice(
            name="v",
            source=fields.pop("source", "clone"),
            status=fields.pop("status", "ready"),
            organization_id=org_id,
            **fields,
        )
        db.add(voice)
        db.commit()
        return voice.id


# --- agents -------------------------------------------------------------------------


def test_agent_crud_same_org(client, org_a):
    agent = make_agent(client, org_a["admin"])
    got = client.get(f"/agents/{agent['id']}", headers=org_a["member"])
    assert got.status_code == 200 and got.json()["id"] == agent["id"]
    patched = client.patch(
        f"/agents/{agent['id']}", json={"name": "B"}, headers=org_a["admin"]
    )
    assert patched.status_code == 200 and patched.json()["name"] == "B"
    deleted = client.delete(f"/agents/{agent['id']}", headers=org_a["admin"])
    assert deleted.status_code == 204


def test_agents_hidden_from_other_org(client, org_a, org_b):
    agent = make_agent(client, org_a["admin"])
    mine = make_agent(client, org_b["admin"])
    listed = client.get("/agents", headers=org_b["member"])
    assert listed.status_code == 200
    ids = {item["id"] for item in listed.json()}
    assert agent["id"] not in ids and mine["id"] in ids
    assert agent["id"] in {
        item["id"] for item in client.get("/agents", headers=org_a["member"]).json()
    }

    path = f"/agents/{agent['id']}"
    for response in (
        client.get(path, headers=org_b["admin"]),
        client.patch(path, json={"name": "x"}, headers=org_b["admin"]),
        client.delete(path, headers=org_b["admin"]),
    ):
        assert response.status_code == 404
        assert response.json()["detail"] == "agent_not_found"
    still = client.get(path, headers=org_a["member"]).json()
    assert still["name"] == "A" and still["status"] == "enabled"


def test_agent_create_stamps_caller_org(client, org_a):
    agent = make_agent(client, org_a["owner"])
    with SessionLocal() as db:
        from src.platform.models import Agent

        assert db.get(Agent, agent["id"]).organization_id == org_a["id"]


def test_agent_cross_org_provider_and_voice_profile_links_indistinguishable_from_missing(client, org_a, org_b):
    foreign_llm = make_provider(client, org_a["admin"], "llm")
    foreign_stt = make_provider(client, org_a["admin"], "stt")
    foreign_tts = make_provider(client, org_a["admin"], "tts")
    foreign_voice = make_voice(org_a["id"], source="library")
    own = make_agent(client, org_b["admin"])
    for field, value in (
        ("llm_provider_id", foreign_llm["id"]),
        ("stt_provider_id", foreign_stt["id"]),
        ("tts_provider_id", foreign_tts["id"]),
        ("voice_profile_id", foreign_voice),
    ):
        created = client.post(
            "/agents", json={"name": "x", field: value}, headers=org_b["admin"]
        )
        assert created.status_code == 422, (field, created.text)
        patched = client.patch(
            f"/agents/{own['id']}", json={field: value}, headers=org_b["admin"]
        )
        assert patched.status_code == 422, (field, patched.text)
        # Same response as a random, nonexistent id (no existence oracle).
        ghost = str(uuid4())
        created_ghost = client.post(
            "/agents", json={"name": "x", field: ghost}, headers=org_b["admin"]
        )
        patched_ghost = client.patch(
            f"/agents/{own['id']}", json={field: ghost}, headers=org_b["admin"]
        )
        assert (created_ghost.status_code, created_ghost.json()) == (
            created.status_code,
            created.json(),
        )
        assert (patched_ghost.status_code, patched_ghost.json()) == (
            patched.status_code,
            patched.json(),
        )
    assert len(client.get("/agents", headers=org_b["member"]).json()) == 1


def test_agent_own_links_still_work_and_missing_is_422(client, org_a):
    llm = make_provider(client, org_a["admin"], "llm")
    voice = make_voice(org_a["id"], source="library")
    agent = make_agent(
        client, org_a["admin"], llm_provider_id=llm["id"], voice_profile_id=voice
    )
    assert agent["llm_provider_id"] == llm["id"]
    missing = client.post(
        "/agents", json={"name": "x", "llm_provider_id": MISSING}, headers=org_a["admin"]
    )
    assert missing.status_code == 422
    assert missing.json()["detail"] == "invalid_llm_provider_id"


def test_agent_voice_id_of_foreign_clone_is_409(client, org_a, org_b):
    make_voice(org_a["id"], provider_voice_id="el-clone-123")
    own = make_agent(client, org_b["admin"])
    created = client.post(
        "/agents",
        json={"name": "x", "voice_id": "el-clone-123"},
        headers=org_b["admin"],
    )
    assert created.status_code == 409
    assert created.json()["detail"] == {"code": "voice_not_owned"}
    patched = client.patch(
        f"/agents/{own['id']}", json={"voice_id": "el-clone-123"}, headers=org_b["admin"]
    )
    assert patched.status_code == 409
    # The owner may use it; unrelated raw ids stay accepted.
    assert (
        make_agent(client, org_a["admin"], voice_id="el-clone-123")["voice_id"]
        == "el-clone-123"
    )
    assert make_agent(client, org_b["admin"], voice_id="stock-voice")["voice_id"]


def test_agent_writes_need_admin(client, org_a):
    agent = make_agent(client, org_a["admin"])
    member = org_a["member"]
    assert client.post("/agents", json={"name": "x"}, headers=member).status_code == 403
    assert (
        client.patch(
            f"/agents/{agent['id']}", json={"name": "x"}, headers=member
        ).status_code
        == 403
    )
    assert client.delete(f"/agents/{agent['id']}", headers=member).status_code == 403
    assert client.get("/agents", headers=member).status_code == 200


# --- providers ----------------------------------------------------------------------


def test_providers_listed_per_org_and_admin_to_create(client, org_a, org_b):
    provider = make_provider(client, org_a["admin"])
    ids_b = {p["id"] for p in client.get("/providers", headers=org_b["member"]).json()}
    assert provider["id"] not in ids_b
    ids_a = {p["id"] for p in client.get("/providers", headers=org_a["member"]).json()}
    assert provider["id"] in ids_a
    denied = client.post(
        "/providers",
        json={"kind": "llm", "provider": "openai"},
        headers=org_a["member"],
    )
    assert denied.status_code == 403
    assert denied.json()["detail"] == {"code": "insufficient_role"}


# --- templates ----------------------------------------------------------------------


def test_templates_are_global_but_agents_land_in_caller_org(client, org_a, org_b):
    for org in (org_a, org_b):
        listed = client.get("/agent-templates", headers=org["member"])
        assert listed.status_code == 200 and listed.json()
        one = client.get("/agent-templates/sales", headers=org["member"])
        assert one.status_code == 200 and one.json()["key"] == "sales"
    created = client.post(
        "/agent-templates/sales/agents", json={}, headers=org_a["admin"]
    )
    assert created.status_code == 201
    agent_id = created.json()["id"]
    assert agent_id in {a["id"] for a in client.get("/agents", headers=org_a["member"]).json()}
    assert agent_id not in {
        a["id"] for a in client.get("/agents", headers=org_b["member"]).json()
    }
    assert client.get(f"/agents/{agent_id}", headers=org_b["admin"]).status_code == 404


def test_template_agent_creation_needs_admin_and_unknown_key_404(client, org_a):
    member = client.post(
        "/agent-templates/sales/agents", json={}, headers=org_a["member"]
    )
    assert member.status_code == 403
    unknown = client.post(
        "/agent-templates/nope/agents", json={}, headers=org_a["admin"]
    )
    assert unknown.status_code == 404
    assert (
        client.get("/agent-templates/nope", headers=org_a["member"]).status_code == 404
    )

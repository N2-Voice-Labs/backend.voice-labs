"""Organisation isolation for knowledge documents, tools and agent links (T5c)."""

import os
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import AgentKnowledgeDocument, KnowledgeDocument
from src.platform.organizations import ensure_default_org

SUPER_KEY = "super-key-t5c"
SUPER = {"Authorization": f"Bearer {SUPER_KEY}"}

ISOLATION_CASES = [
    ("POST", "/knowledge/documents"),
    ("GET", "/knowledge/documents"),
    ("GET", "/knowledge/documents/{document_id}"),
    ("PATCH", "/knowledge/documents/{document_id}"),
    ("DELETE", "/knowledge/documents/{document_id}"),
    ("PUT", "/agents/{agent_id}/knowledge"),
    ("GET", "/agents/{agent_id}/knowledge"),
    ("POST", "/runtime/agents/{agent_id}/knowledge/search"),
    ("POST", "/tools"),
    ("GET", "/tools"),
    ("GET", "/tools/{tool_id}"),
    ("PATCH", "/tools/{tool_id}"),
    ("DELETE", "/tools/{tool_id}"),
    ("PUT", "/agents/{agent_id}/tools"),
    ("GET", "/agents/{agent_id}/tools"),
]


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER_KEY)
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "rt-secret")
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    with TestClient(app) as test_client:
        yield test_client


def make_org(client) -> dict:
    slug = f"org-t5c-{os.urandom(4).hex()}"
    created = client.post(
        "/organizations",
        json={"name": "Org", "slug": slug, "create_owner_key": True},
        headers=SUPER,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    owner = body["owner_key"]["key"]
    member = client.post(
        "/api-keys", json={"name": "m", "role": "member"}, headers=bearer(owner)
    )
    assert member.status_code == 201, member.text
    admin_h = bearer(owner)
    agent = client.post("/agents", json={"name": "agent"}, headers=admin_h)
    assert agent.status_code == 201, agent.text
    doc = client.post(
        "/knowledge/documents",
        json={"title": "Refund policy", "content": "Refunds take five days."},
        headers=admin_h,
    )
    assert doc.status_code == 201, doc.text
    tool = client.post(
        "/tools",
        json={"name": "lookup", "description": "Look things up"},
        headers=admin_h,
    )
    assert tool.status_code == 201, tool.text
    return {
        "id": body["id"],
        "admin": admin_h,
        "member": bearer(member.json()["key"]),
        "agent": agent.json()["id"],
        "doc": doc.json()["id"],
        "tool": tool.json()["id"],
    }


@pytest.fixture
def orgs(client):
    return make_org(client), make_org(client)


# ----------------------------------------------------------------- knowledge


def test_document_routes_hide_other_org(client, orgs) -> None:
    a, b = orgs
    url = f"/knowledge/documents/{a['doc']}"
    assert client.get(url, headers=b["admin"]).status_code == 404
    assert client.get(url, headers=b["admin"]).json()["detail"] == (
        "knowledge_document_not_found"
    )
    assert (
        client.patch(url, json={"title": "x"}, headers=b["admin"]).status_code == 404
    )
    assert client.delete(url, headers=b["admin"]).status_code == 404
    # Untouched for the owner.
    got = client.get(url, headers=a["admin"])
    assert got.status_code == 200 and got.json()["title"] == "Refund policy"
    assert got.json()["enabled"] is True


def test_document_lists_exclude_other_org(client, orgs) -> None:
    a, b = orgs
    ids_b = [d["id"] for d in client.get("/knowledge/documents", headers=b["admin"]).json()]
    assert ids_b == [b["doc"]]
    ids_a = [d["id"] for d in client.get("/knowledge/documents", headers=a["member"]).json()]
    assert ids_a == [a["doc"]]
    assert client.get("/knowledge/documents?enabled=true", headers=b["member"]).json()[0][
        "id"
    ] == b["doc"]


def test_document_roles(client, orgs) -> None:
    a, _ = orgs
    body = {"title": "t", "content": "c"}
    url = f"/knowledge/documents/{a['doc']}"
    assert client.post("/knowledge/documents", json=body, headers=a["member"]).status_code == 403
    assert client.patch(url, json={"title": "x"}, headers=a["member"]).status_code == 403
    assert client.delete(url, headers=a["member"]).status_code == 403
    assert client.get(url, headers=a["member"]).status_code == 200
    assert client.get("/knowledge/documents", headers=a["member"]).status_code == 200
    assert client.post("/knowledge/documents", json=body, headers=a["admin"]).status_code == 201


def test_agent_knowledge_cross_org(client, orgs) -> None:
    a, b = orgs
    url_a = f"/agents/{a['agent']}/knowledge"
    # Other org's agent -> 404 for both verbs.
    assert client.get(url_a, headers=b["admin"]).status_code == 404
    assert client.get(url_a, headers=b["admin"]).json()["detail"] == "agent_not_found"
    put = client.put(url_a, json={"document_ids": []}, headers=b["admin"])
    assert put.status_code == 404 and put.json()["detail"] == "agent_not_found"
    # Own agent but other org's document -> 422, same as an unknown id.
    url_b = f"/agents/{b['agent']}/knowledge"
    cross = client.put(url_b, json={"document_ids": [a["doc"]]}, headers=b["admin"])
    assert cross.status_code == 422
    assert cross.json()["detail"] == "invalid_knowledge_document_id"
    mixed = client.put(
        url_b, json={"document_ids": [b["doc"], a["doc"]]}, headers=b["admin"]
    )
    assert mixed.status_code == 422
    assert client.get(url_b, headers=b["admin"]).json() == []
    # Unknown id keeps the pre-existing 422.
    unknown = client.put(url_b, json={"document_ids": [str(uuid4())]}, headers=b["admin"])
    assert unknown.status_code == 422
    assert unknown.json()["detail"] == "invalid_knowledge_document_id"
    assert (unknown.status_code, unknown.json()) == (cross.status_code, cross.json())
    # Same-org link works.
    ok = client.put(url_b, json={"document_ids": [b["doc"]]}, headers=b["admin"])
    assert ok.status_code == 200 and [d["id"] for d in ok.json()] == [b["doc"]]
    assert [d["id"] for d in client.get(url_b, headers=b["member"]).json()] == [b["doc"]]


def test_agent_knowledge_roles(client, orgs) -> None:
    a, _ = orgs
    url = f"/agents/{a['agent']}/knowledge"
    assert client.put(url, json={"document_ids": []}, headers=a["member"]).status_code == 403
    assert client.get(url, headers=a["member"]).status_code == 200


def test_runtime_search_ignores_stale_cross_org_link(client, orgs) -> None:
    a, b = orgs
    own = client.post(
        "/knowledge/documents",
        json={"title": "Own", "content": "zebra crossing rules apply"},
        headers=a["admin"],
    ).json()
    with SessionLocal() as db:
        foreign = KnowledgeDocument(
            title="Foreign",
            content="zebra secrets of the other tenant",
            organization_id=b["id"],
        )
        db.add(foreign)
        db.flush()
        foreign_id = foreign.id
        db.add_all(
            [
                AgentKnowledgeDocument(agent_id=a["agent"], document_id=foreign_id),
                AgentKnowledgeDocument(agent_id=a["agent"], document_id=own["id"]),
            ]
        )
        db.commit()
    resp = client.post(
        f"/runtime/agents/{a['agent']}/knowledge/search",
        json={"query": "zebra"},
        headers={"x-runtime-token": "rt-secret"},
    )
    assert resp.status_code == 200, resp.text
    ids = [r["document_id"] for r in resp.json()["results"]]
    assert ids == [own["id"]]
    assert foreign_id not in ids
    missing = client.post(
        f"/runtime/agents/{uuid4()}/knowledge/search",
        json={"query": "zebra"},
        headers={"x-runtime-token": "rt-secret"},
    )
    assert missing.status_code == 404 and missing.json()["detail"] == "agent_not_found"


# --------------------------------------------------------------------- tools


def test_tool_routes_hide_other_org(client, orgs) -> None:
    a, b = orgs
    url = f"/tools/{a['tool']}"
    for resp in (
        client.get(url, headers=b["admin"]),
        client.patch(url, json={"description": "x"}, headers=b["admin"]),
        client.delete(url, headers=b["admin"]),
    ):
        assert resp.status_code == 404
        assert resp.json()["detail"] == "tool_not_found"
    got = client.get(url, headers=a["admin"]).json()
    assert got["enabled"] is True and got["description"] == "Look things up"


def test_tool_lists_exclude_other_org(client, orgs) -> None:
    a, b = orgs
    assert [t["id"] for t in client.get("/tools", headers=b["admin"]).json()] == [b["tool"]]
    assert [t["id"] for t in client.get("/tools", headers=a["member"]).json()] == [a["tool"]]


def test_tool_roles(client, orgs) -> None:
    a, _ = orgs
    body = {"name": "another", "description": "d"}
    url = f"/tools/{a['tool']}"
    assert client.post("/tools", json=body, headers=a["member"]).status_code == 403
    assert client.patch(url, json={"description": "x"}, headers=a["member"]).status_code == 403
    assert client.delete(url, headers=a["member"]).status_code == 403
    assert client.get(url, headers=a["member"]).status_code == 200
    assert client.get("/tools", headers=a["member"]).status_code == 200
    assert client.post("/tools", json=body, headers=a["admin"]).status_code == 201


def test_tool_names_unique_per_org(client, orgs) -> None:
    a, b = orgs
    # Both orgs already hold "lookup" (fixture) -> allowed across orgs.
    dup = client.post(
        "/tools", json={"name": "lookup", "description": "d"}, headers=a["admin"]
    )
    assert dup.status_code == 409 and dup.json()["detail"] == "tool_name_conflict"
    other = client.post(
        "/tools", json={"name": "second", "description": "d"}, headers=a["admin"]
    ).json()
    clash = client.patch(
        f"/tools/{other['id']}", json={"name": "lookup"}, headers=a["admin"]
    )
    assert clash.status_code == 409 and clash.json()["detail"] == "tool_name_conflict"
    # Same name in org B is independent.
    assert client.post(
        "/tools", json={"name": "second", "description": "d"}, headers=b["admin"]
    ).status_code == 201


def test_agent_tools_cross_org(client, orgs) -> None:
    a, b = orgs
    url_a = f"/agents/{a['agent']}/tools"
    assert client.get(url_a, headers=b["admin"]).status_code == 404
    put = client.put(url_a, json={"tool_ids": []}, headers=b["admin"])
    assert put.status_code == 404 and put.json()["detail"] == "agent_not_found"
    url_b = f"/agents/{b['agent']}/tools"
    cross = client.put(url_b, json={"tool_ids": [a["tool"]]}, headers=b["admin"])
    assert cross.status_code == 422 and cross.json()["detail"] == "invalid_tool_id"
    mixed = client.put(url_b, json={"tool_ids": [b["tool"], a["tool"]]}, headers=b["admin"])
    assert mixed.status_code == 422
    assert client.get(url_b, headers=b["admin"]).json() == []
    unknown = client.put(url_b, json={"tool_ids": [str(uuid4())]}, headers=b["admin"])
    assert unknown.status_code == 422 and unknown.json()["detail"] == "invalid_tool_id"
    assert (unknown.status_code, unknown.json()) == (cross.status_code, cross.json())
    ok = client.put(url_b, json={"tool_ids": [b["tool"]]}, headers=b["admin"])
    assert ok.status_code == 200 and [t["id"] for t in ok.json()] == [b["tool"]]
    assert [t["id"] for t in client.get(url_b, headers=b["member"]).json()] == [b["tool"]]


def test_agent_tools_roles(client, orgs) -> None:
    a, _ = orgs
    url = f"/agents/{a['agent']}/tools"
    assert client.put(url, json={"tool_ids": []}, headers=a["member"]).status_code == 403
    assert client.get(url, headers=a["member"]).status_code == 200


def test_isolation_cases_declared() -> None:
    assert len(ISOLATION_CASES) == len(set(ISOLATION_CASES)) == 15

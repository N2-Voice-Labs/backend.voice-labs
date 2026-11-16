import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import VoiceSession


def contract_for(session_id: str) -> dict:
    with SessionLocal() as db:
        return db.get(VoiceSession, session_id).runtime_contract


def make_tool(client: TestClient, **overrides) -> dict:
    body = {
        "name": f"tool_{uuid4().hex[:12]}",
        "description": "Look something up",
        "input_schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        "configuration": {"url": "https://example.test/hook", "method": "GET"},
        **overrides,
    }
    response = client.post("/tools", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def start_session(client: TestClient, agent_id: str) -> dict:
    response = client.post("/voice/sessions", json={"agent_id": agent_id})
    assert response.status_code == 201, response.text
    return contract_for(response.json()["session_id"])


def test_contract_without_tools_or_knowledge_has_stable_empty_keys() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Bare"}).json()
        contract = start_session(client, agent["id"])

    assert contract["tools"] == []
    assert contract["knowledge_enabled"] is False


def test_small_knowledge_is_inlined_in_system_prompt(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_INLINE_KNOWLEDGE_MAX_CHARS", "1000")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Inline", "system_prompt": "Be helpful."}).json()
        doc = client.post("/knowledge/documents", json={"title": "Facts", "content": "Applications open in March."}).json()
        client.put(f"/agents/{agent['id']}/knowledge", json={"document_ids": [doc["id"]]})
        contract = start_session(client, agent["id"])
    assert contract["knowledge_enabled"] is False
    assert "Be helpful.\n\n## Knowledge base" in contract["system_prompt"]
    assert "### Facts\nApplications open in March." in contract["system_prompt"]


def test_inline_knowledge_threshold_zero_uses_search(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_INLINE_KNOWLEDGE_MAX_CHARS", "0")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search fallback"}).json()
        doc = client.post("/knowledge/documents", json={"title": "Facts", "content": "Verified."}).json()
        client.put(f"/agents/{agent['id']}/knowledge", json={"document_ids": [doc["id"]]})
        contract = start_session(client, agent["id"])
    assert contract["knowledge_enabled"] is True
    assert "## Knowledge base" not in contract["system_prompt"]


def test_contract_carries_assigned_enabled_tools_and_knowledge() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Equipped"}).json()
        http_tool = make_tool(client)
        builtin = make_tool(
            client, name=f"transfer_to_human_{uuid4().hex[:6]}", tool_type="builtin", configuration={}
        )
        retired = make_tool(client)
        unassigned = make_tool(client)
        assigned = client.put(
            f"/agents/{agent['id']}/tools",
            json={"tool_ids": [http_tool["id"], builtin["id"], retired["id"]]},
        )
        assert assigned.status_code == 200
        assert client.delete(f"/tools/{retired['id']}").status_code == 204

        document = client.post(
            "/knowledge/documents", json={"title": "FAQ", "content": "Open 9 to 5."}
        ).json()
        assert client.put(
            f"/agents/{agent['id']}/knowledge", json={"document_ids": [document["id"]]}
        ).status_code == 200

        contract = start_session(client, agent["id"])

        assert contract["knowledge_enabled"] is False
        assert "### FAQ\nOpen 9 to 5." in contract["system_prompt"]
        assert contract["tools"] == sorted(
            [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "tool_type": tool["tool_type"],
                    "input_schema": tool["input_schema"],
                    "configuration": tool["configuration"],
                }
                for tool in (http_tool, builtin)
            ],
            key=lambda tool: tool["name"],
        )
        assert unassigned["name"] not in [tool["name"] for tool in contract["tools"]]
        assert retired["name"] not in [tool["name"] for tool in contract["tools"]]

        client.delete(f"/knowledge/documents/{document['id']}")
        assert start_session(client, agent["id"])["knowledge_enabled"] is False

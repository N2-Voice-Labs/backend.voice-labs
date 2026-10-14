import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app


def make_tool(client: TestClient, **overrides) -> dict:
    body = {
        "name": f"tool_{uuid4().hex[:12]}",
        "description": "Look something up",
        "input_schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        "configuration": {"url": "https://example.test/hook"},
        **overrides,
    }
    response = client.post("/tools", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_tool_crud_and_soft_delete() -> None:
    with TestClient(app) as client:
        tool = make_tool(client, tool_type="builtin")
        assert tool["enabled"] is True and tool["tool_type"] == "builtin"
        assert client.get(f"/tools/{tool['id']}").json() == tool

        patched = client.patch(
            f"/tools/{tool['id']}", json={"description": "Updated", "enabled": True}
        )
        assert patched.status_code == 200 and patched.json()["description"] == "Updated"

        assert client.delete(f"/tools/{tool['id']}").status_code == 204
        assert client.get(f"/tools/{tool['id']}").json()["enabled"] is False
        enabled_ids = {t["id"] for t in client.get("/tools?enabled=true").json()}
        disabled_ids = {t["id"] for t in client.get("/tools?enabled=false").json()}
        assert tool["id"] not in enabled_ids and tool["id"] in disabled_ids
        assert tool["id"] in {t["id"] for t in client.get("/tools").json()}


def test_tool_validation_errors() -> None:
    with TestClient(app) as client:
        base = {"name": "valid_name", "description": "d"}
        for bad in (
            {"name": "Bad-Name"},
            {"name": "1abc"},
            {"name": "a" * 65},
            {"description": ""},
            {"description": "x" * 2001},
            {"tool_type": "grpc"},
            {"input_schema": {"type": "string"}},
            {"input_schema": {"type": "object", "properties": []}},
            {"configuration": {"api_key": "sk-123"}},
        ):
            assert client.post("/tools", json={**base, **bad}).status_code == 422, bad


def test_tool_name_conflict_on_create_and_patch() -> None:
    with TestClient(app) as client:
        first, second = make_tool(client), make_tool(client)
        duplicate = client.post(
            "/tools", json={"name": first["name"], "description": "dup"}
        )
        assert duplicate.status_code == 409
        assert duplicate.json()["detail"] == "tool_name_conflict"
        renamed = client.patch(f"/tools/{second['id']}", json={"name": first["name"]})
        assert renamed.status_code == 409
        assert renamed.json()["detail"] == "tool_name_conflict"
        fresh = f"renamed_{uuid4().hex[:8]}"
        ok = client.patch(f"/tools/{second['id']}", json={"name": fresh})
        assert ok.status_code == 200 and ok.json()["name"] == fresh


def test_tool_patch_validation_and_not_found() -> None:
    with TestClient(app) as client:
        tool = make_tool(client)
        for bad in (
            {"name": None},
            {"input_schema": {"type": "array"}},
            {"configuration": {"password": "x"}},
        ):
            assert client.patch(f"/tools/{tool['id']}", json=bad).status_code == 422
        missing = str(uuid4())
        for response in (
            client.get(f"/tools/{missing}"),
            client.patch(f"/tools/{missing}", json={"enabled": False}),
            client.delete(f"/tools/{missing}"),
        ):
            assert response.status_code == 404
            assert response.json()["detail"] == "tool_not_found"


def test_agent_tools_replace_dedupe_and_errors() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Tool agent"}).json()
        url = f"/agents/{agent['id']}/tools"
        assert client.get(url).json() == []
        a, b, disabled = make_tool(client), make_tool(client), make_tool(client)
        client.delete(f"/tools/{disabled['id']}")

        put = client.put(url, json={"tool_ids": [a["id"], b["id"], a["id"]]})
        assert put.status_code == 200
        assert {t["id"] for t in put.json()} == {a["id"], b["id"]}
        assert len(put.json()) == 2
        assert client.put(url, json={"tool_ids": [b["id"]]}).status_code == 200
        assert [t["id"] for t in client.get(url).json()] == [b["id"]]

        for ids in ([str(uuid4())], [disabled["id"]], [b["id"], str(uuid4())]):
            bad = client.put(url, json={"tool_ids": ids})
            assert bad.status_code == 422 and bad.json()["detail"] == "invalid_tool_id"
        assert [t["id"] for t in client.get(url).json()] == [b["id"]]

        assert client.put(url, json={"tool_ids": []}).json() == []
        assert client.get(url).json() == []

        missing = f"/agents/{uuid4()}/tools"
        for response in (client.get(missing), client.put(missing, json={"tool_ids": []})):
            assert response.status_code == 404
            assert response.json()["detail"] == "agent_not_found"

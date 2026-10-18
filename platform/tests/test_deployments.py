import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app


def phone() -> str:
    return f"+1{uuid4().int % 10**10:010d}"


def make_agent(client: TestClient) -> dict:
    return client.post("/agents", json={"name": f"agent-{uuid4()}"}).json()


def make_deployment(client: TestClient, agent_id: str, **overrides) -> dict:
    body = {"agent_id": agent_id, "name": f"dep-{uuid4()}", "channel": "web"}
    response = client.post("/deployments", json={**body, **overrides})
    assert response.status_code == 201, response.text
    return response.json()


def test_deployment_crud_and_filters() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        number = phone()
        web = make_deployment(client, agent["id"], configuration={"theme": "dark"})
        sip = make_deployment(client, agent["id"], channel="sip", phone_number=number)
        assert web["status"] == "active"
        assert web["phone_number"] is None
        assert sip["phone_number"] == number

        assert client.get(f"/deployments/{web['id']}").json() == web
        by_agent = client.get("/deployments", params={"agent_id": agent["id"]}).json()
        assert {item["id"] for item in by_agent} == {web["id"], sip["id"]}
        by_channel = client.get(
            "/deployments", params={"agent_id": agent["id"], "channel": "sip"}
        ).json()
        assert [item["id"] for item in by_channel] == [sip["id"]]

        patched = client.patch(
            f"/deployments/{web['id']}",
            json={"name": "renamed", "status": "paused", "configuration": {"a": 1}},
        )
        assert patched.status_code == 200
        assert patched.json()["name"] == "renamed"
        assert patched.json()["status"] == "paused"
        by_status = client.get(
            "/deployments", params={"agent_id": agent["id"], "status": "paused"}
        ).json()
        assert [item["id"] for item in by_status] == [web["id"]]

        new_number = phone()
        moved = client.patch(f"/deployments/{sip['id']}", json={"phone_number": new_number})
        assert moved.json()["phone_number"] == new_number


def test_deployment_not_found() -> None:
    with TestClient(app) as client:
        missing = str(uuid4())
        for call in (client.get, client.delete):
            response = call(f"/deployments/{missing}")
            assert response.status_code == 404
            assert response.json()["detail"] == "deployment_not_found"
        response = client.patch(f"/deployments/{missing}", json={"name": "x"})
        assert response.status_code == 404
        assert response.json()["detail"] == "deployment_not_found"


def test_create_rejects_unknown_or_disabled_agent() -> None:
    with TestClient(app) as client:
        body = {"name": "d", "channel": "web"}
        response = client.post("/deployments", json={**body, "agent_id": str(uuid4())})
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid_agent_id"

        agent = make_agent(client)
        client.delete(f"/agents/{agent['id']}")
        response = client.post("/deployments", json={**body, "agent_id": agent["id"]})
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid_agent_id"


def test_phone_number_invariants() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        base = {"agent_id": agent["id"], "name": "d"}

        response = client.post("/deployments", json={**base, "channel": "sip"})
        assert response.status_code == 422
        assert response.json()["detail"] == "phone_number_required"

        response = client.post(
            "/deployments", json={**base, "channel": "sip", "phone_number": "555-1234"}
        )
        assert response.status_code == 422

        for channel in ("web", "api"):
            response = client.post(
                "/deployments",
                json={**base, "channel": channel, "phone_number": phone()},
            )
            assert response.status_code == 422
            assert response.json()["detail"] == "phone_number_only_for_sip"

        sip = make_deployment(client, agent["id"], channel="sip", phone_number=phone())
        response = client.patch(f"/deployments/{sip['id']}", json={"phone_number": None})
        assert response.status_code == 422
        assert response.json()["detail"] == "phone_number_required"
        response = client.patch(f"/deployments/{sip['id']}", json={"phone_number": "abc"})
        assert response.status_code == 422

        web = make_deployment(client, agent["id"])
        response = client.patch(f"/deployments/{web['id']}", json={"phone_number": phone()})
        assert response.status_code == 422
        assert response.json()["detail"] == "phone_number_only_for_sip"


def test_invalid_payloads_rejected() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        base = {"agent_id": agent["id"], "name": "d", "channel": "web"}
        assert client.post("/deployments", json={**base, "name": ""}).status_code == 422
        assert client.post("/deployments", json={**base, "channel": "fax"}).status_code == 422
        response = client.post(
            "/deployments", json={**base, "configuration": {"api_key": "sk-123"}}
        )
        assert response.status_code == 422
        deployment = make_deployment(client, agent["id"])
        for patch in (
            {"configuration": {"nested": {"password": "x"}}},
            {"status": "deleted"},
            {"name": None},
        ):
            response = client.patch(f"/deployments/{deployment['id']}", json=patch)
            assert response.status_code == 422


def test_delete_archives_and_archived_is_terminal() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        deployment = make_deployment(client, agent["id"])

        assert client.delete(f"/deployments/{deployment['id']}").status_code == 204
        archived = client.get(f"/deployments/{deployment['id']}").json()
        assert archived["status"] == "archived"

        for patch in ({"name": "again"}, {"status": "active"}):
            response = client.patch(f"/deployments/{deployment['id']}", json=patch)
            assert response.status_code == 409
            assert response.json()["detail"] == "deployment_archived"


VALID_TRANSFER_TO = ["+14155550123", "tel:+14155550123", "sip:support@example.com", "sips:s@example.com"]
INVALID_TRANSFER_TO = [
    "555-1234",
    "14155550123",
    "+0123",
    "",
    " +14155550123",
    "http://example.com",
    "sip:",
    "sip:has space@example.com",
    "sip:" + "a" * 252,
    12345,
    ["+14155550123"],
]


def test_transfer_to_validated_on_create_and_patch() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        for value in VALID_TRANSFER_TO:
            created = make_deployment(
                client, agent["id"], configuration={"transfer_to": value}
            )
            assert created["configuration"]["transfer_to"] == value
        # Optional: absent or null is fine.
        make_deployment(client, agent["id"], configuration={"theme": "dark"})
        make_deployment(client, agent["id"], configuration={"transfer_to": None})

        deployment = make_deployment(client, agent["id"])
        for value in INVALID_TRANSFER_TO:
            body = {"agent_id": agent["id"], "name": "x", "channel": "web",
                    "configuration": {"transfer_to": value}}
            response = client.post("/deployments", json=body)
            assert response.status_code == 422, value
            assert "invalid_transfer_to" in response.text, value
            response = client.patch(
                f"/deployments/{deployment['id']}",
                json={"configuration": {"transfer_to": value}},
            )
            assert response.status_code == 422, value
            assert "invalid_transfer_to" in response.text, value
        ok = client.patch(
            f"/deployments/{deployment['id']}",
            json={"configuration": {"transfer_to": "+442071234567"}},
        )
        assert ok.status_code == 200
        assert ok.json()["configuration"]["transfer_to"] == "+442071234567"
        # The 255-character limit is inclusive.
        boundary = "sip:" + "a" * 251
        assert len(boundary) == 255
        assert (
            client.patch(
                f"/deployments/{deployment['id']}",
                json={"configuration": {"transfer_to": boundary}},
            ).status_code
            == 200
        )

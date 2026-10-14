import os

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app


def test_list_agent_templates_returns_all_six_templates() -> None:
    with TestClient(app) as client:
        response = client.get("/agent-templates")

    assert response.status_code == 200
    templates = response.json()
    assert len(templates) == 6
    template_keys = {t["key"] for t in templates}
    assert template_keys == {
        "customer_support",
        "sales",
        "receptionist",
        "booking",
        "restaurant",
        "travel",
    }


def test_get_unknown_template_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/agent-templates/unknown_key")

    assert response.status_code == 404
    assert response.json()["detail"] == "agent_template_not_found"


def test_create_agent_from_template_with_name_override() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agent-templates/customer_support/agents",
            json={"name": "My Custom Support Agent"},
        )

    assert response.status_code == 201
    agent = response.json()
    assert agent["name"] == "My Custom Support Agent"
    assert agent["status"] == "enabled"
    assert agent["template_key"] == "customer_support"
    assert agent["system_prompt"]  # Should contain the template's system prompt
    assert agent["enabled_languages"] == ["en", "ru"]

    # Verify we can fetch the created agent
    agent_id = agent["id"]
    fetch_response = client.get(f"/agents/{agent_id}")
    assert fetch_response.status_code == 200
    fetched_agent = fetch_response.json()
    assert fetched_agent["id"] == agent_id
    assert fetched_agent["name"] == "My Custom Support Agent"
    assert fetched_agent["status"] == "enabled"

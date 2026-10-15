import os

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import VoiceSession


def contract_for(session_id: str) -> dict:
    with SessionLocal() as db:
        return db.get(VoiceSession, session_id).runtime_contract


def test_create_interpreter_agent_auto_en_ru() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "Interpreter EN-RU",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        )
        assert response.status_code == 201
        agent = response.json()
        assert agent["mode"] == "interpreter"
        assert agent["interpreter"] == {"languages": ["en", "ru"], "direction": "auto"}


def test_create_interpreter_agent_one_way_ko_en() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "Interpreter KO->EN",
                "mode": "interpreter",
                "interpreter": {"languages": ["ko", "en"], "direction": "one_way"},
            },
        )
        assert response.status_code == 201
        agent = response.json()
        assert agent["mode"] == "interpreter"
        assert agent["interpreter"] == {"languages": ["ko", "en"], "direction": "one_way"}


def test_create_interpreter_agent_auto_with_unsupported_language_ko() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "Interpreter KO-EN Auto",
                "mode": "interpreter",
                "interpreter": {"languages": ["ko", "en"], "direction": "auto"},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "code-switching" in str(err) for err in detail
        )


def test_create_interpreter_agent_duplicate_languages() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "Duplicate Languages",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "en"], "direction": "auto"},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "distinct" in str(err) for err in detail
        )


def test_create_interpreter_agent_one_language() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "One Language",
                "mode": "interpreter",
                "interpreter": {"languages": ["en"], "direction": "auto"},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "distinct" in str(err) for err in detail
        )


def test_create_interpreter_agent_three_languages() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "Three Languages",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru", "es"], "direction": "auto"},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "distinct" in str(err) for err in detail
        )


def test_create_interpreter_agent_unknown_language() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "Unknown Language",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "xyz"], "direction": "auto"},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "unknown language" in str(err) for err in detail
        )


def test_create_interpreter_agent_without_config() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={"name": "Missing Config", "mode": "interpreter"},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "interpreter_config_required" in str(err) for err in detail
        )


def test_create_agent_with_interpreter_config_not_allowed() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/agents",
            json={
                "name": "Agent With Config",
                "mode": "agent",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "interpreter_config_not_allowed" in str(err) for err in detail
        )


def test_patch_agent_to_interpreter_mode() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Starting Agent"}).json()
        assert agent["mode"] == "agent"
        assert agent["interpreter"] is None

        response = client.patch(
            f"/agents/{agent['id']}",
            json={
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        )
        assert response.status_code == 200
        updated = response.json()
        assert updated["mode"] == "interpreter"
        assert updated["interpreter"] == {"languages": ["en", "ru"], "direction": "auto"}


def test_patch_interpreter_to_agent_mode() -> None:
    with TestClient(app) as client:
        agent = client.post(
            "/agents",
            json={
                "name": "Starting Interpreter",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        ).json()
        assert agent["mode"] == "interpreter"

        response = client.patch(
            f"/agents/{agent['id']}", json={"mode": "agent", "interpreter": None}
        )
        assert response.status_code == 200
        updated = response.json()
        assert updated["mode"] == "agent"
        assert updated["interpreter"] is None


def test_patch_mode_without_interpreter_when_required() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Starting Agent"}).json()

        response = client.patch(f"/agents/{agent['id']}", json={"mode": "interpreter"})
        assert response.status_code == 422
        detail = response.json()["detail"]
        if isinstance(detail, str):
            assert "interpreter_config_required" in detail
        else:
            assert detail == "interpreter_config_required"


def test_patch_interpreter_config_invalid() -> None:
    with TestClient(app) as client:
        agent = client.post(
            "/agents",
            json={
                "name": "Existing Interpreter",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        ).json()

        response = client.patch(
            f"/agents/{agent['id']}",
            json={
                "interpreter": {"languages": ["ko", "en"], "direction": "auto"}
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and any(
            "code-switching" in str(err) for err in detail
        )


def test_session_contract_includes_mode_and_interpreter_agent() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Agent Session"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        contract = contract_for(session["session_id"])

        assert "mode" in contract
        assert contract["mode"] == "agent"
        assert "interpreter" in contract
        assert contract["interpreter"] is None


def test_session_contract_includes_mode_and_interpreter_interpreter() -> None:
    with TestClient(app) as client:
        agent = client.post(
            "/agents",
            json={
                "name": "Interpreter Session",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        contract = contract_for(session["session_id"])

        assert "mode" in contract
        assert contract["mode"] == "interpreter"
        assert "interpreter" in contract
        assert contract["interpreter"] == {"languages": ["en", "ru"], "direction": "auto"}


def test_get_agent_includes_mode_and_interpreter() -> None:
    with TestClient(app) as client:
        created = client.post(
            "/agents",
            json={
                "name": "Get Test",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        ).json()

        retrieved = client.get(f"/agents/{created['id']}").json()
        assert retrieved["mode"] == "interpreter"
        assert retrieved["interpreter"] == {"languages": ["en", "ru"], "direction": "auto"}


def test_list_agents_includes_mode_and_interpreter() -> None:
    with TestClient(app) as client:
        client.post(
            "/agents",
            json={
                "name": "Interpreter Agent",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "auto"},
            },
        )

        agents = client.get("/agents").json()
        interpreter_agents = [a for a in agents if a.get("mode") == "interpreter"]
        assert len(interpreter_agents) > 0
        assert interpreter_agents[0]["interpreter"] is not None


def test_all_supported_languages_in_interpreter_languages_constant() -> None:
    """Verify all languages from contract are available."""
    from src.platform.validation import INTERPRETER_LANGUAGES

    supported = {"en", "ru", "ko", "ja", "zh", "es", "fr", "de", "it", "pt", "nl", "hi", "tr", "pl"}
    assert INTERPRETER_LANGUAGES == supported


def test_auto_direction_languages_subset() -> None:
    """Verify AUTO_DIRECTION_LANGUAGES is a proper subset of INTERPRETER_LANGUAGES."""
    from src.platform.validation import AUTO_DIRECTION_LANGUAGES, INTERPRETER_LANGUAGES

    assert AUTO_DIRECTION_LANGUAGES.issubset(INTERPRETER_LANGUAGES)
    assert AUTO_DIRECTION_LANGUAGES == {"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"}

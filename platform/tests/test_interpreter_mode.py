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
        assert agent["interpreter"] == {"languages": ["en", "ru"], "direction": "auto", "engine": "cascaded"}


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
        assert agent["interpreter"] == {"languages": ["ko", "en"], "direction": "one_way", "engine": "cascaded"}


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
        assert updated["interpreter"] == {"languages": ["en", "ru"], "direction": "auto", "engine": "cascaded"}


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
        assert contract["interpreter"] == {"languages": ["en", "ru"], "direction": "auto", "engine": "cascaded"}


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
        assert retrieved["interpreter"] == {"languages": ["en", "ru"], "direction": "auto", "engine": "cascaded"}


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


# -- realtime engine (TR-010) ---------------------------------------------------------


def _realtime_agent(languages, direction="one_way", engine="realtime", **extra):
    return {
        "name": "Realtime Interpreter",
        "mode": "interpreter",
        "interpreter": {"languages": languages, "direction": direction, "engine": engine},
        **extra,
    }


def test_realtime_output_languages_constant_is_the_verified_list() -> None:
    from src.platform.validation import REALTIME_TRANSLATE_OUTPUT_LANGUAGES

    assert REALTIME_TRANSLATE_OUTPUT_LANGUAGES == {
        "es", "pt", "fr", "ja", "ru", "zh", "de", "ko", "hi", "id", "vi", "it", "en",
    }  # fmt: skip


def test_interpreter_engine_defaults_to_cascaded() -> None:
    with TestClient(app) as client:
        agent = client.post(
            "/agents",
            json={
                "name": "Default Engine",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ru"], "direction": "one_way"},
            },
        ).json()
        assert agent["interpreter"]["engine"] == "cascaded"


def test_stored_interpreter_without_engine_reads_back_as_cascaded() -> None:
    from src.platform.models import Agent as AgentRecord

    with TestClient(app) as client:
        agent = client.post("/agents", json=_realtime_agent(["ru", "en"], engine="cascaded")).json()
        with SessionLocal() as db:
            record = db.get(AgentRecord, agent["id"])
            record.interpreter = {"languages": ["ru", "en"], "direction": "one_way"}
            db.commit()
        assert client.get(f"/agents/{agent['id']}").json()["interpreter"]["engine"] == "cascaded"


def test_create_realtime_agent_one_way_reaches_runtime_contract() -> None:
    with TestClient(app) as client:
        response = client.post("/agents", json=_realtime_agent(["ru", "en"]))
        assert response.status_code == 201
        agent = response.json()
        assert agent["interpreter"] == {
            "languages": ["ru", "en"],
            "direction": "one_way",
            "engine": "realtime",
        }
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        assert contract_for(session["session_id"])["interpreter"]["engine"] == "realtime"


def test_realtime_agent_rejects_auto_direction() -> None:
    with TestClient(app) as client:
        response = client.post("/agents", json=_realtime_agent(["en", "ru"], direction="auto"))
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid_interpreter_config"


def test_realtime_agent_rejects_unsupported_output_language() -> None:
    with TestClient(app) as client:
        # nl and tr are valid cascaded languages but not realtime output languages.
        for target in ("nl", "tr", "pl"):
            response = client.post("/agents", json=_realtime_agent(["en", target]))
            assert response.status_code == 422, target
            assert response.json()["detail"] == "invalid_interpreter_config"
        # The source language is not limited: the model detects 70+ input languages.
        assert client.post("/agents", json=_realtime_agent(["nl", "en"])).status_code == 201


def test_patch_to_realtime_validates_engine_rules() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json=_realtime_agent(["en", "ru"], engine="cascaded")).json()
        bad = client.patch(
            f"/agents/{agent['id']}",
            json={"interpreter": {"languages": ["ru", "en"], "direction": "auto", "engine": "realtime"}},
        )
        assert bad.status_code == 422
        assert bad.json()["detail"] == "invalid_interpreter_config"
        bad_target = client.patch(
            f"/agents/{agent['id']}",
            json={"interpreter": {"languages": ["en", "nl"], "direction": "one_way", "engine": "realtime"}},
        )
        assert bad_target.status_code == 422
        good = client.patch(
            f"/agents/{agent['id']}",
            json={"interpreter": {"languages": ["ru", "en"], "direction": "one_way", "engine": "realtime"}},
        )
        assert good.status_code == 200
        assert good.json()["interpreter"]["engine"] == "realtime"


def test_unknown_engine_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/agents", json=_realtime_agent(["en", "ru"], engine="whisper"))
        assert response.status_code == 422

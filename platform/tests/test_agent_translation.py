"""Agent + translator (TR-040): the agent's ``translation`` config and session contract."""

import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient

from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import VoiceSession


def contract_for(session_id: str) -> dict:
    with SessionLocal() as db:
        return db.get(VoiceSession, session_id).runtime_contract


def make_agent(client: TestClient, translation, **extra):
    body = {"name": f"translated-{uuid4()}", "translation": translation, **extra}
    return client.post("/agents", json=body)


def test_agent_with_automatic_detection_round_trips() -> None:
    translation = {"working_language": "en", "caller_languages": ["ru", "es"]}
    with TestClient(app) as client:
        created = make_agent(client, translation)
        assert created.status_code == 201, created.text
        assert created.json()["translation"] == translation
        fetched = client.get(f"/agents/{created.json()['id']}").json()
        assert fetched["translation"] == translation


def test_fixed_caller_language_outside_the_detection_set_is_allowed() -> None:
    # ko is not in Deepgram's code-switching set: exactly one fixed caller language.
    with TestClient(app) as client:
        response = make_agent(client, {"working_language": "en", "caller_languages": ["ko"]})
        assert response.status_code == 201, response.text
        assert response.json()["translation"]["caller_languages"] == ["ko"]


@pytest.mark.parametrize(
    "translation",
    [
        {"working_language": "en", "caller_languages": ["ko", "ru"]},  # ko needs a fixed language
        {"working_language": "ko", "caller_languages": ["en", "ru"]},  # working outside the set
        {"working_language": "en", "caller_languages": []},
        {"working_language": "en", "caller_languages": ["ru", "es", "fr", "de", "it"]},  # > 4
        {"working_language": "en", "caller_languages": ["ru", "ru"]},  # duplicate
        {"working_language": "en", "caller_languages": ["en"]},  # includes working
        {"working_language": "en", "caller_languages": ["xx"]},  # unknown language
        {"working_language": "xx", "caller_languages": ["ru"]},
        {"working_language": "en", "caller_languages": "ru"},  # wrong type
        {"working_language": "en"},  # missing key
        {"working_language": "en", "caller_languages": ["ru"], "extra": 1},
    ],
)
def test_invalid_translation_config_is_rejected(translation) -> None:
    with TestClient(app) as client:
        response = make_agent(client, translation)
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid_translation_config"


def test_translation_is_not_allowed_for_interpreter_agents() -> None:
    with TestClient(app) as client:
        response = make_agent(
            client,
            {"working_language": "en", "caller_languages": ["ru"]},
            mode="interpreter",
            interpreter={"languages": ["en", "ru"], "direction": "auto"},
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "translation_not_allowed_for_interpreter"


def test_patch_validates_the_merged_state_and_can_clear_translation() -> None:
    translation = {"working_language": "en", "caller_languages": ["ru"]}
    with TestClient(app) as client:
        agent = make_agent(client, translation).json()
        url = f"/agents/{agent['id']}"

        bad = client.patch(url, json={"translation": {"working_language": "en", "caller_languages": ["en"]}})
        assert (bad.status_code, bad.json()["detail"]) == (422, "invalid_translation_config")

        # Switching to interpreter mode while a translation is set is rejected.
        switched = client.patch(
            url,
            json={"mode": "interpreter", "interpreter": {"languages": ["en", "ru"], "direction": "auto"}},
        )
        assert (switched.status_code, switched.json()["detail"]) == (
            422,
            "translation_not_allowed_for_interpreter",
        )
        assert client.get(url).json()["translation"] == translation

        updated = client.patch(url, json={"translation": {"working_language": "EN", "caller_languages": ["Ru", "es"]}})
        assert updated.status_code == 200
        assert updated.json()["translation"] == {"working_language": "en", "caller_languages": ["ru", "es"]}

        cleared = client.patch(url, json={"translation": None})
        assert cleared.status_code == 200 and cleared.json()["translation"] is None


def test_session_contract_carries_translation_only_when_set() -> None:
    translation = {"working_language": "en", "caller_languages": ["ru", "es"]}
    with TestClient(app) as client:
        plain = client.post("/agents", json={"name": "Plain"}).json()
        plain_session = client.post("/voice/sessions", json={"agent_id": plain["id"]}).json()
        assert "translation" not in contract_for(plain_session["session_id"])

        translated = make_agent(client, translation).json()
        session = client.post("/voice/sessions", json={"agent_id": translated["id"]}).json()
        contract = contract_for(session["session_id"])
        assert contract["translation"] == translation
        assert contract["mode"] == "agent" and contract["interpreter"] is None


def test_translation_model_provider_setting_reaches_the_runtime_contract() -> None:
    with TestClient(app) as client:
        provider = client.post(
            "/providers",
            json={
                "kind": "llm",
                "provider": "openai-compatible",
                "configuration": {"model": "gpt-4.1", "translation_model": "gpt-4.1-mini"},
            },
        ).json()
        agent = make_agent(
            client,
            {"working_language": "en", "caller_languages": ["ru"]},
            llm_provider_id=provider["id"],
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        llm = contract_for(session["session_id"])["provider_settings"]["llm"]
        assert llm == {"model": "gpt-4.1", "translation_model": "gpt-4.1-mini"}


_created_providers: list[str] = []


@pytest.fixture
def cleanup_providers():
    """Remove the provider (and agent) rows these tests add: other tests count providers."""

    yield
    from src.platform.models import Agent as AgentRecord, ProviderConfig

    with SessionLocal() as db:
        db.query(AgentRecord).filter(AgentRecord.stt_provider_id.in_(_created_providers)).delete(
            synchronize_session=False
        )
        db.query(ProviderConfig).filter(ProviderConfig.id.in_(_created_providers)).delete(
            synchronize_session=False
        )
        db.commit()
    _created_providers.clear()


def _stt_provider(client: TestClient, provider: str) -> str:
    response = client.post(
        "/providers",
        json={"kind": "stt", "provider": provider, "configuration": {}, "enabled": True},
    )
    assert response.status_code == 201, response.text
    _created_providers.append(response.json()["id"])
    return response.json()["id"]


@pytest.mark.usefixtures("cleanup_providers")
def test_elevenlabs_stt_detects_korean_so_several_caller_languages_are_allowed() -> None:
    translation = {"working_language": "en", "caller_languages": ["ko", "ru"]}
    with TestClient(app) as client:
        response = make_agent(client, translation, stt_provider_id=_stt_provider(client, "elevenlabs"))
        assert response.status_code == 201, response.text
        assert response.json()["translation"] == translation


@pytest.mark.usefixtures("cleanup_providers")
def test_switching_to_an_stt_provider_that_cannot_detect_the_callers_is_rejected() -> None:
    translation = {"working_language": "en", "caller_languages": ["ko", "ru"]}
    with TestClient(app) as client:
        created = make_agent(client, translation, stt_provider_id=_stt_provider(client, "elevenlabs"))
        assert created.status_code == 201, created.text
        response = client.patch(
            f"/agents/{created.json()['id']}",
            json={"stt_provider_id": _stt_provider(client, "deepgram")},
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "invalid_translation_config"


def test_reply_language_reminder_only_for_multilingual_agents() -> None:
    from src.platform.demo import reply_language_reminder

    assert reply_language_reminder(["en"]) == ""
    assert reply_language_reminder(None) == ""
    reminder = reply_language_reminder(["en", "ru", "ko"])
    assert "English, Russian, Korean" in reminder and "otherwise in English" in reminder

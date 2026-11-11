"""Translation sessions: two people, no agent to configure (``/translations``)."""

import os

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import Agent, VoiceSession

from test_interpreter_participants import runtime  # noqa: F401  (fixture)


def start(client: TestClient, **extra):
    body = {
        "languages": ["ko", "en"],
        "creator": {"language": "ko", "display_name": "Min"},
        **extra,
    }
    return client.post("/translations", json=body)


def contract_for(session_id: str) -> dict:
    with SessionLocal() as db:
        return db.get(VoiceSession, session_id).runtime_contract


def test_create_sends_an_interpreter_contract_with_no_agent_content(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        response = start(client)
        assert response.status_code == 201, response.text
        session_id = response.json()["session_id"]
    sent = requests[0]["json"]
    assert sent["mode"] == "interpreter"
    assert sent["system_prompt"] == ""
    assert sent["tools"] == []
    assert sent["knowledge_enabled"] is False
    assert sent["interpreter"]["engine"] == "cascaded"
    assert sent["interpreter"]["languages"] == ["ko", "en"]
    # Korean is outside Deepgram's code-switching set: per-seat lanes, one_way validation.
    assert sent["interpreter"]["direction"] == "one_way"
    assert list(sent["participants"].values())[0]["language"] == "ko"
    assert contract_for(session_id)["mode"] == "interpreter"


def test_hidden_agent_is_reused_and_not_listed() -> None:
    with TestClient(app) as client:
        before = {a["id"] for a in client.get("/agents").json()}
        first = start(client).json()["session_id"]
        second = start(client).json()["session_id"]
        assert {a["id"] for a in client.get("/agents").json()} == before
        with SessionLocal() as db:
            ours = {
                db.get(VoiceSession, sid).runtime_contract["agent_id"]
                for sid in (first, second)
            }
            assert len(ours) == 1  # same config: one hidden agent, reused
            agent = db.get(Agent, ours.pop())
            assert agent.system is True
            assert agent.system_prompt == ""
        assert first != second


def test_creator_language_must_be_in_the_pair() -> None:
    with TestClient(app) as client:
        response = start(client, creator={"language": "fr"})
        assert (response.status_code, response.json()["detail"]) == (
            422,
            "invalid_participant_language",
        )


def test_invalid_pair_is_rejected() -> None:
    with TestClient(app) as client:
        for languages in (["en", "en"], ["en"], ["en", "xx"]):
            response = start(client, languages=languages, creator={"language": "en"})
            assert response.status_code == 422, languages


def test_invite_seats_the_second_person_and_never_displaces_them_silently(runtime) -> None:
    with TestClient(app) as client:
        session_id = start(client).json()["session_id"]
        invited = client.post(f"/translations/{session_id}/invite", json={"language": "en"})
        assert invited.status_code == 200, invited.text
        assert invited.json()["participant_identity"].startswith("guest-")
        again = client.post(f"/translations/{session_id}/invite", json={"language": "en"})
        assert again.status_code == 409
        assert again.json()["detail"]["code"] == "guest_already_invited"
    languages = {p["language"] for p in contract_for(session_id)["participants"].values()}
    assert languages == {"ko", "en"}


def test_invite_with_the_same_language_is_rejected(runtime) -> None:
    with TestClient(app) as client:
        session_id = start(client).json()["session_id"]
        response = client.post(f"/translations/{session_id}/invite", json={"language": "ko"})
        assert (response.status_code, response.json()["detail"]) == (
            422,
            "participant_languages_must_differ",
        )


def test_get_and_list_hide_voice_ids(runtime) -> None:
    with TestClient(app) as client:
        session_id = start(client).json()["session_id"]
        view = client.get(f"/translations/{session_id}")
        assert view.status_code == 200
        body = view.json()
        assert body["languages"] == ["ko", "en"]
        assert body["participants"][0]["language"] == "ko"
        assert "voice_id" not in body["participants"][0]
        listed = client.get("/translations").json()
        assert session_id in {item["id"] for item in listed}
        assert client.get("/translations/00000000-0000-0000-0000-000000000000").status_code == 404

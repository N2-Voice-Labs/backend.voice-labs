"""Two-person interpreter sessions (TR-020): per-participant language and voice."""

import os
from uuid import uuid4

import jwt
import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import VoiceSession


class RuntimeResponse:
    def __init__(self, body: dict, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("POST", "http://runtime.test"), response=self  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return self.body


class RuntimeClient:
    def __init__(self, requests: list[dict], respond) -> None:
        self.requests = requests
        self.respond = respond

    async def __aenter__(self) -> "RuntimeClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(self, url: str, **kwargs: object) -> RuntimeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.respond(url)

    async def get(self, url: str, **kwargs: object) -> RuntimeResponse:
        # Status poll of ensure-running: the Runtime lost the session.
        return RuntimeResponse({"status": "not_running"})


@pytest.fixture
def runtime(monkeypatch):
    """Mocked Runtime HTTP: records requests; ``state['status']`` sets participant replies."""

    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    requests: list[dict] = []
    state = {"participant_status": 200}

    def respond(url: str) -> RuntimeResponse:
        if url.endswith("/participants"):
            return RuntimeResponse({"status": "updated"}, state["participant_status"])
        return RuntimeResponse({"status": "accepted"})

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: RuntimeClient(requests, respond),
    )
    return requests, state


@pytest.fixture
def livekit(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "ws://livekit.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s" * 40)


def contract_for(session_id: str) -> dict:
    with SessionLocal() as db:
        return db.get(VoiceSession, session_id).runtime_contract


def make_agent(client: TestClient, **extra) -> dict:
    body = {
        "name": f"interpreter-{uuid4()}",
        "mode": "interpreter",
        "interpreter": {"languages": ["en", "ko"], "direction": "one_way"},
        **extra,
    }
    response = client.post("/agents", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def make_ready_voice(client: TestClient, provider_voice_id: str) -> dict:
    tts = client.post("/providers", json={"kind": "tts", "provider": "cartesia"}).json()
    response = client.post(
        "/voices",
        json={
            "name": f"voice-{uuid4()}",
            "tts_provider_id": tts["id"],
            "provider_voice_id": provider_voice_id,
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "ready"
    return response.json()


def create_session(client: TestClient, agent: dict, **participant):
    body = {"agent_id": agent["id"]}
    if participant:
        body["participant"] = participant
    return client.post("/voice/sessions", json=body)


# -- creating a session with the first participant ---------------------------------------


def test_session_without_participant_has_no_participants_in_contract() -> None:
    with TestClient(app) as client:
        session = create_session(client, make_agent(client)).json()
        assert "participants" not in contract_for(session["session_id"])


def test_first_participant_is_stored_under_the_browser_identity(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        voice = make_ready_voice(client, "cartesia-alice")
        agent = make_agent(client)
        response = create_session(
            client, agent, language="en", voice_profile_id=voice["id"], display_name="Alice"
        )
        assert response.status_code == 201, response.text
        session_id = response.json()["session_id"]

    expected = {
        f"browser-{session_id}": {
            "language": "en",
            "voice_id": "cartesia-alice",
            "display_name": "Alice",
        }
    }
    assert contract_for(session_id)["participants"] == expected
    # The same contract is what the Runtime receives at session start.
    assert requests[0]["json"]["participants"] == expected


def test_first_participant_without_voice_profile_uses_agent_voice(livekit) -> None:
    with TestClient(app) as client:
        response = create_session(client, make_agent(client), language="ko")
        session_id = response.json()["session_id"]
        # The browser token identity is the participant identity in the contract.
        token = jwt.decode(
            response.json()["connection"]["token"], options={"verify_signature": False}
        )
    assert token["sub"] == f"browser-{session_id}"
    assert contract_for(session_id)["participants"] == {
        f"browser-{session_id}": {"language": "ko", "voice_id": None, "display_name": None}
    }


def test_participant_requires_a_cascaded_interpreter() -> None:
    with TestClient(app) as client:
        plain = client.post("/agents", json={"name": "plain"}).json()
        realtime = make_agent(
            client,
            interpreter={"languages": ["en", "ru"], "direction": "one_way", "engine": "realtime"},
        )
        for agent in (plain, realtime):
            response = create_session(client, agent, language="en")
            assert response.status_code == 422
            assert response.json()["detail"] == "participants_require_cascaded_interpreter"


def test_participant_language_must_be_an_agent_language() -> None:
    with TestClient(app) as client:
        response = create_session(client, make_agent(client), language="fr")
        assert (response.status_code, response.json()["detail"]) == (422, "invalid_participant_language")
        # Case and whitespace are normalised.
        assert create_session(client, make_agent(client), language=" KO ").status_code == 201


def test_participant_voice_profile_must_be_ready_with_a_provider_voice() -> None:
    with TestClient(app) as client:
        agent = make_agent(client)
        missing = create_session(client, agent, language="en", voice_profile_id=str(uuid4()))
        assert (missing.status_code, missing.json()["detail"]) == (409, "selected_voice_unavailable")

        pending = client.post(
            "/voices/design", json={"name": "pending", "design_prompt": "calm and warm"}
        ).json()
        assert pending["status"] == "pending"
        blocked = create_session(client, agent, language="en", voice_profile_id=pending["id"])
        assert (blocked.status_code, blocked.json()["detail"]) == (409, "selected_voice_unavailable")


# -- adding the second participant -----------------------------------------------------------


def start_two_person_session(client: TestClient, **first):
    voice = make_ready_voice(client, "cartesia-alice")
    agent = make_agent(client)
    created = create_session(
        client, agent, language="en", voice_profile_id=voice["id"], **first
    ).json()
    return agent, created["session_id"]


def test_add_participant_returns_guest_connection_and_notifies_runtime(runtime, livekit) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        agent, session_id = start_two_person_session(client)
        bob_voice = make_ready_voice(client, "cartesia-bob")
        response = client.post(
            f"/voice/sessions/{session_id}/participants",
            json={"language": "ko", "voice_profile_id": bob_voice["id"], "display_name": "Bob"},
        )
        assert response.status_code == 200, response.text
        body = response.json()

    identity = body["participant_identity"]
    assert identity.startswith("guest-") and identity != f"browser-{session_id}"
    assert body["connection"]["url"] == "ws://livekit.test"
    token = jwt.decode(body["connection"]["token"], options={"verify_signature": False})
    assert token["sub"] == identity
    assert token["video"]["room"] == f"voice-labs-{session_id}"

    entry = {"language": "ko", "voice_id": "cartesia-bob", "display_name": "Bob"}
    stored = contract_for(session_id)["participants"]
    assert set(stored) == {f"browser-{session_id}", identity}
    assert stored[identity] == entry

    notify = requests[-1]
    assert notify["url"] == f"http://runtime.test/sessions/{session_id}/participants"
    assert notify["headers"] == {"X-Runtime-Token": "shared-secret"}
    assert notify["json"] == {"identity": identity, **entry}


def test_ensure_running_restart_carries_all_participants(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        _, session_id = start_two_person_session(client)
        guest = client.post(
            f"/voice/sessions/{session_id}/participants", json={"language": "ko"}
        ).json()["participant_identity"]
        restarted = client.post(f"/voice/sessions/{session_id}/ensure-running")
        assert restarted.status_code == 200, restarted.text

    start = requests[-1]
    assert start["url"] == "http://runtime.test/sessions"
    assert start["json"]["participants"] == {
        f"browser-{session_id}": {"language": "en", "voice_id": "cartesia-alice", "display_name": None},
        guest: {"language": "ko", "voice_id": None, "display_name": None},
    }


def test_add_participant_without_runtime_only_stores_it(monkeypatch) -> None:
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    with TestClient(app) as client:
        _, session_id = start_two_person_session(client)
        response = client.post(
            f"/voice/sessions/{session_id}/participants", json={"language": "ko"}
        )
        assert response.status_code == 200
        assert response.json()["connection"] == {"url": "", "token": ""}
    assert len(contract_for(session_id)["participants"]) == 2


def test_third_participant_is_rejected(runtime) -> None:
    with TestClient(app) as client:
        _, session_id = start_two_person_session(client)
        url = f"/voice/sessions/{session_id}/participants"
        assert client.post(url, json={"language": "ko"}).status_code == 200
        full = client.post(url, json={"language": "ko"})
        assert (full.status_code, full.json()["detail"]) == (409, "session_full")
    assert len(contract_for(session_id)["participants"]) == 2


def test_add_participant_validations(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        agent, session_id = start_two_person_session(client)
        url = f"/voice/sessions/{session_id}/participants"
        sent = len(requests)

        wrong_language = client.post(url, json={"language": "fr"})
        assert (wrong_language.status_code, wrong_language.json()["detail"]) == (422, "invalid_participant_language")
        same_language = client.post(url, json={"language": "en"})
        assert (same_language.status_code, same_language.json()["detail"]) == (422, "participant_languages_must_differ")
        bad_voice = client.post(url, json={"language": "ko", "voice_profile_id": str(uuid4())})
        assert (bad_voice.status_code, bad_voice.json()["detail"]) == (409, "selected_voice_unavailable")
        assert len(requests) == sent  # nothing reached the Runtime
        assert len(contract_for(session_id)["participants"]) == 1

        unknown = client.post(f"/voice/sessions/{uuid4()}/participants", json={"language": "ko"})
        assert (unknown.status_code, unknown.json()["detail"]) == (404, "voice_session_not_found")


def test_add_participant_needs_an_active_two_person_session(runtime) -> None:
    with TestClient(app) as client:
        # A session created without a first participant cannot take a second one.
        plain = create_session(client, make_agent(client)).json()["session_id"]
        response = client.post(f"/voice/sessions/{plain}/participants", json={"language": "ko"})
        assert (response.status_code, response.json()["detail"]) == (409, "participants_not_enabled")

        _, session_id = start_two_person_session(client)
        assert client.post(f"/voice/sessions/{session_id}/end").status_code == 200
        ended = client.post(f"/voice/sessions/{session_id}/participants", json={"language": "ko"})
        assert (ended.status_code, ended.json()["detail"]) == (409, "session_not_active")


def test_add_participant_runtime_failure_is_not_persisted(runtime) -> None:
    _, state = runtime
    with TestClient(app) as client:
        _, session_id = start_two_person_session(client)
        url = f"/voice/sessions/{session_id}/participants"
        state["participant_status"] = 503
        failed = client.post(url, json={"language": "ko"})
        assert (failed.status_code, failed.json()["detail"]) == (503, "runtime_unavailable")
        assert len(contract_for(session_id)["participants"]) == 1

        # 404: the Runtime does not hold the session; the stored contract is enough.
        state["participant_status"] = 404
        assert client.post(url, json={"language": "ko"}).status_code == 200
    assert len(contract_for(session_id)["participants"]) == 2

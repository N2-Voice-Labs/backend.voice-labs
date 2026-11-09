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

    async def delete(self, url: str, **kwargs: object) -> RuntimeResponse:
        self.requests.append({"url": url, "method": "DELETE", **kwargs})
        return RuntimeResponse({"status": "removed"})

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


def test_connection_url_prefers_livekit_public_url(runtime, livekit, monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_PUBLIC_URL", "ws://localhost:7880")
    with TestClient(app) as client:
        _, session_id = start_two_person_session(client)
        bob_voice = make_ready_voice(client, "cartesia-bob")
        response = client.post(
            f"/voice/sessions/{session_id}/participants",
            json={"language": "ko", "voice_profile_id": bob_voice["id"], "display_name": "Bob"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["connection"]["url"] == "ws://localhost:7880"


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
        assert (failed.status_code, failed.json()["detail"]["code"]) == (503, "runtime_unavailable")
        assert len(contract_for(session_id)["participants"]) == 1

        # 404: the Runtime does not hold the session; the stored contract is enough.
        state["participant_status"] = 404
        assert client.post(url, json={"language": "ko"}).status_code == 200
    assert len(contract_for(session_id)["participants"]) == 2


# -- private listening: participants can never receive another person's microphone -----------


def grants(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False})["video"]


def test_interpreter_participants_cannot_subscribe_in_the_session_room(
    runtime, livekit, monkeypatch
) -> None:
    requests, _ = runtime
    monkeypatch.setenv("PLATFORM_PRIVATE_LISTENING", "1")
    with TestClient(app) as client:
        agent = make_agent(client)
        created = create_session(client, agent, language="en").json()
        session_id = created["session_id"]
        guest = client.post(
            f"/voice/sessions/{session_id}/participants", json={"language": "ko"}
        ).json()

    room = f"voice-labs-{session_id}"
    for identity, connection in (
        (f"browser-{session_id}", created["connection"]),
        (guest["participant_identity"], guest["connection"]),
    ):
        # Session room: publish the microphone, receive nothing (LiveKit refuses every
        # subscription, auto, client-requested or forced through the server API).
        main = grants(connection["token"])
        assert (main["room"], main["canPublish"], main["canSubscribe"]) == (room, True, False)
        # Listen room: subscribe-only, joined by this person and the Runtime alone.
        listen = connection["listen"]
        assert listen["url"] == "ws://livekit.test" == connection["url"]
        assert listen["room_name"] == f"{room}--listen--{identity}"
        assert listen["track"] == f"translation-for-{identity}"
        token = jwt.decode(listen["token"], options={"verify_signature": False})
        assert token["sub"] == identity
        assert token["video"]["room"] == listen["room_name"]
        assert (token["video"]["canSubscribe"], token["video"]["canPublish"]) == (True, False)
        assert token["video"]["canPublishData"] is False
    # The Runtime is told to publish translations into the listen rooms.
    assert requests[0]["json"]["private_listening"] is True
    assert contract_for(session_id)["private_listening"] is True


def test_hear_original_sessions_keep_one_shared_room(runtime, livekit) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        agent = make_agent(
            client,
            interpreter={"languages": ["en", "ko"], "direction": "one_way", "hear_original": True},
        )
        created = create_session(client, agent, language="en").json()
        guest = client.post(
            f"/voice/sessions/{created['session_id']}/participants", json={"language": "ko"}
        ).json()
    for connection in (created["connection"], guest["connection"]):
        assert "listen" not in connection
        assert grants(connection["token"])["canSubscribe"] is True
    assert "private_listening" not in requests[0]["json"]
    assert "private_listening" not in contract_for(created["session_id"])


def test_interpreter_without_a_participant_and_plain_agents_are_unchanged(runtime, livekit) -> None:
    with TestClient(app) as client:
        interpreter = create_session(client, make_agent(client))
        plain = client.post("/voice/sessions", json={"agent_id": client.post("/agents", json={"name": "p"}).json()["id"]})
    for response in (interpreter, plain):
        connection = response.json()["connection"]
        assert set(connection) == {"url", "token"}
        assert grants(connection["token"])["canSubscribe"] is True
        assert "private_listening" not in contract_for(response.json()["session_id"])


def test_private_listening_is_off_by_default(runtime, livekit, monkeypatch) -> None:
    requests, _ = runtime
    monkeypatch.delenv("PLATFORM_PRIVATE_LISTENING", raising=False)
    with TestClient(app) as client:
        created = create_session(client, make_agent(client), language="en").json()
    assert set(created["connection"]) == {"url", "token"}
    assert grants(created["connection"]["token"])["canSubscribe"] is True
    assert "private_listening" not in requests[0]["json"]


def test_a_guest_follows_the_privacy_mode_the_session_started_with(runtime, livekit, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_PRIVATE_LISTENING", "1")
    with TestClient(app) as client:
        session_id = create_session(client, make_agent(client), language="en").json()["session_id"]
        monkeypatch.setenv("PLATFORM_PRIVATE_LISTENING", "0")  # changed mid-call: no effect
        guest = client.post(
            f"/voice/sessions/{session_id}/participants", json={"language": "ko"}
        ).json()
    assert grants(guest["connection"]["token"])["canSubscribe"] is False
    assert "listen" in guest["connection"]


def test_listen_room_name_convention() -> None:
    from src.platform.livekit import listen_room_name

    assert listen_room_name("voice-labs-1", "guest-ab12") == "voice-labs-1--listen--guest-ab12"


# -- phone participants -----------------------------------------------------------------------


@pytest.fixture
def sip_trunk(monkeypatch):
    """A configured trunk; ``dials`` records what would be dialled (no SIP call is made)."""

    monkeypatch.setenv("LIVEKIT_URL", "ws://livekit.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s" * 40)
    monkeypatch.setenv("LIVEKIT_SIP_TRUNK_ID", "ST_test")
    dials: list[dict] = []
    hangups: list[tuple[str, str]] = []
    state = {"fail": False}

    async def dial(number, room, identity, *, display_name=None):
        dials.append({"number": number, "room": room, "identity": identity, "name": display_name})
        if state["fail"]:
            raise sip_module.SipDialError("no answer", sip_status_code=480)
        return "PA_phone"

    async def hangup(room, identity):
        hangups.append((room, identity))

    from src.platform import sip as sip_module

    monkeypatch.setattr(sip_module, "dial", dial)
    monkeypatch.setattr(sip_module, "hangup", hangup)
    return dials, hangups, state


PHONE = {"language": "ko", "phone_number": "+821012345678", "display_name": "Bob"}


def phone_url(session_id: str) -> str:
    return f"/voice/sessions/{session_id}/phone-participants"


def test_phone_participant_is_dialled_into_a_room_only_the_runtime_shares(
    runtime, livekit, sip_trunk, monkeypatch
) -> None:
    requests, _ = runtime
    dials, hangups, _ = sip_trunk
    monkeypatch.setenv("PLATFORM_PRIVATE_LISTENING", "1")
    with TestClient(app) as client:
        session_id = create_session(client, make_agent(client), language="en").json()["session_id"]
        response = client.post(phone_url(session_id), json=PHONE)
        assert response.status_code == 201, response.text
        body = response.json()

        identity = body["participant_identity"]
        phone_room = f"voice-labs-{session_id}--phone--{identity}"
        assert identity.startswith("sip-guest-")
        assert body == {
            "participant_identity": identity,
            "participant_id": "PA_phone",
            "phone_number": "+821012345678",
            "room_name": phone_room,
            "private_listening": True,
        }
        # Never the session room, so the other person's microphone is not in it at all.
        assert dials == [
            {"number": "+821012345678", "room": phone_room, "identity": identity, "name": "Bob"}
        ]
        entry = {"language": "ko", "voice_id": None, "display_name": "Bob", "phone_room": True}
        assert contract_for(session_id)["participants"][identity] == entry
        # The Runtime joined the phone room (was told) before the call was placed.
        notify = [r for r in requests if r["url"].endswith("/participants")][-1]
        assert notify["json"] == {"identity": identity, **entry}

        # Ending the session hangs the call up in the room it lives in.
        assert client.post(f"/voice/sessions/{session_id}/end").status_code == 200
    assert hangups == [(phone_room, identity)]


def test_phone_participant_without_private_listening_joins_the_session_room(
    runtime, livekit, sip_trunk, monkeypatch
) -> None:
    dials, hangups, _ = sip_trunk
    monkeypatch.delenv("PLATFORM_PRIVATE_LISTENING", raising=False)
    with TestClient(app) as client:
        session_id = create_session(client, make_agent(client), language="en").json()["session_id"]
        body = client.post(phone_url(session_id), json=PHONE).json()
        identity = body["participant_identity"]
        assert body["room_name"] == f"voice-labs-{session_id}" and not body["private_listening"]
        assert dials[0]["room"] == f"voice-labs-{session_id}"
        assert "phone_room" not in contract_for(session_id)["participants"][identity]
        client.post(f"/voice/sessions/{session_id}/end")
    assert hangups == [(f"voice-labs-{session_id}", identity)]


def test_hear_original_phone_participant_stays_in_the_shared_room(
    runtime, livekit, sip_trunk, monkeypatch
) -> None:
    dials, _, _ = sip_trunk
    monkeypatch.setenv("PLATFORM_PRIVATE_LISTENING", "1")
    with TestClient(app) as client:
        agent = make_agent(
            client, interpreter={"languages": ["en", "ko"], "direction": "one_way", "hear_original": True}
        )
        session_id = create_session(client, agent, language="en").json()["session_id"]
        body = client.post(phone_url(session_id), json=PHONE).json()
    assert body["room_name"] == f"voice-labs-{session_id}" and not body["private_listening"]
    assert dials[0]["room"] == f"voice-labs-{session_id}"


def test_unanswered_phone_call_withdraws_the_participant_and_frees_the_slot(
    runtime, livekit, sip_trunk, monkeypatch
) -> None:
    requests, _ = runtime
    _, _, state = sip_trunk
    monkeypatch.setenv("PLATFORM_PRIVATE_LISTENING", "1")
    with TestClient(app) as client:
        session_id = create_session(client, make_agent(client), language="en").json()["session_id"]
        state["fail"] = True
        failed = client.post(phone_url(session_id), json=PHONE)
        assert (failed.status_code, failed.json()["detail"]) == (502, "sip_dial_failed")
        assert len(contract_for(session_id)["participants"]) == 1
        deleted = requests[-1]
        assert deleted["method"] == "DELETE"
        assert deleted["url"].startswith(f"http://runtime.test/sessions/{session_id}/participants/sip-guest-")
        assert deleted["headers"] == {"X-Runtime-Token": "shared-secret"}
        state["fail"] = False
        assert client.post(phone_url(session_id), json=PHONE).status_code == 201


def test_phone_participant_validations(runtime, livekit, monkeypatch) -> None:
    monkeypatch.delenv("LIVEKIT_SIP_TRUNK_ID", raising=False)
    with TestClient(app) as client:
        session_id = create_session(client, make_agent(client), language="en").json()["session_id"]
        url = phone_url(session_id)
        assert client.post(url, json={**PHONE, "phone_number": "0101234"}).status_code == 422
        assert client.post(url, json={**PHONE, "language": "xx"}).json()["detail"] == "invalid_participant_language"
        assert client.post(url, json={**PHONE, "language": "en"}).json()["detail"] == "participant_languages_must_differ"
        not_configured = client.post(url, json=PHONE)
        assert (not_configured.status_code, not_configured.json()["detail"]) == (409, "sip_not_configured")
        assert client.post(phone_url(str(uuid4())), json=PHONE).status_code == 404


def test_phone_participant_needs_an_open_slot_and_a_creator_token(
    runtime, livekit, sip_trunk
) -> None:
    with TestClient(app) as client:
        session_id = create_session(client, make_agent(client), language="en").json()["session_id"]
        assert client.post(phone_url(session_id), json=PHONE).status_code == 201
        full = client.post(phone_url(session_id), json={**PHONE, "language": "en"})
        assert (full.status_code, full.json()["detail"]) == (409, "session_full")
    from src.platform import auth

    assert ("POST", "/voice/sessions/{session_id}/phone-participants") in auth.SESSION_SCOPED_ROUTES
    assert auth.SESSION_SCOPED_ROUTES[("POST", "/voice/sessions/{session_id}/phone-participants")].creator_only


def test_phone_room_name_convention() -> None:
    from src.platform.livekit import phone_room_name

    assert phone_room_name("voice-labs-1", "sip-guest-ab12") == "voice-labs-1--phone--sip-guest-ab12"

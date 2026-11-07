"""Shareable-link calls: no accounts, opt-in, bounded, and only reaching their own call."""

import os

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi.testclient import TestClient

from src.platform import calls
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import ProviderConfig, VoiceSession

from test_interpreter_participants import livekit, runtime  # noqa: F401  (fixtures)

ADMIN = "admin-key-for-tests-0123456789"


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("PLATFORM_PUBLIC_CALLS", "1")
    monkeypatch.delenv("PLATFORM_API_KEYS", raising=False)
    monkeypatch.delenv("PLATFORM_CALL_INVITE_TTL_SECS", raising=False)
    calls.reset_limits()
    yield
    calls.reset_limits()


def bearer(token: str) -> dict:
    return {"authorization": f"Bearer {token}"}


def start(client: TestClient, **extra):
    return client.post("/calls", json={"languages": ["ko", "en"], "name": "Min", **extra})


def invite_for(client: TestClient, call: dict) -> dict:
    response = client.post(f"/calls/{call['session_id']}/invite", headers=bearer(call["session_token"]))
    assert response.status_code == 200, response.text
    return response.json()


def contract_for(session_id: str) -> dict:
    with SessionLocal() as db:
        return db.get(VoiceSession, session_id).runtime_contract


def test_the_feature_is_off_unless_enabled(monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_PUBLIC_CALLS")
    with TestClient(app) as client:
        assert start(client).status_code == 404
        assert client.get("/calls/invites/x.y").status_code == 404
        assert client.post("/calls/invites/x.y/join", json={}).status_code == 404


def test_create_returns_the_creators_connection_and_a_session_token(runtime, livekit) -> None:
    with TestClient(app) as client:
        response = start(client)
        assert response.status_code == 201, response.text
        call = response.json()
    assert call["session_token"].startswith("vls_")
    assert call["connection"]["url"] == "ws://livekit.test" and call["connection"]["token"]
    contract = contract_for(call["session_id"])
    assert contract["interpreter"]["languages"] == ["ko", "en"]
    assert "dubbing" not in contract["interpreter"]  # AI-only unless chosen
    assert contract["interpreter"].get("hear_original") in (None, False)


def test_configured_providers_are_used_without_the_browser_choosing(runtime, livekit, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_CALLS_PROVIDERS", "stt=elevenlabs, llm=google-translate ,tts=elevenlabs,bogus=x")
    assert calls.configured_providers() == {"stt": "elevenlabs", "llm": "google-translate", "tts": "elevenlabs"}
    with TestClient(app) as client:
        first = start(client, providers={"stt": "deepgram"})  # an unknown field from the browser is ignored
        second = start(client)
        assert first.status_code == 201 and second.status_code == 201, first.text + second.text
    for call in (first.json(), second.json()):
        assert contract_for(call["session_id"])["providers"] == {"stt": "elevenlabs", "llm": "google-translate", "tts": "elevenlabs"}
    with SessionLocal() as db:  # the records are created once and reused
        rows = db.query(ProviderConfig).filter(ProviderConfig.provider.in_(["elevenlabs", "google-translate"])).all()
    assert sorted((r.kind, r.provider) for r in rows) == [("llm", "google-translate"), ("stt", "elevenlabs"), ("tts", "elevenlabs")]


def test_without_configuration_the_runtime_defaults_apply(runtime, livekit, monkeypatch) -> None:
    monkeypatch.delenv("PLATFORM_CALLS_PROVIDERS", raising=False)
    with TestClient(app) as client:
        call = start(client).json()
    assert "providers" not in contract_for(call["session_id"])


def test_cinematic_experience_selects_the_dubbing_profile(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client, experience="cinematic").json()
    assert contract_for(call["session_id"])["interpreter"]["dubbing"] == "cinematic"


def test_invite_join_seats_one_guest_in_the_other_language(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client).json()
        invite = invite_for(client, call)
        info = client.get(f"/calls/invites/{invite['code']}")
        assert info.status_code == 200 and info.json()["me"] == "en" and info.json()["them"] == "ko"
        assert info.json()["from"] == "Min"
        joined = client.post(f"/calls/invites/{invite['code']}/join", json={"name": "Sam"})
        assert joined.status_code == 200, joined.text
        guest = joined.json()
    assert guest["session_id"] == call["session_id"] and guest["me"] == "en"
    assert guest["session_token"].startswith("vls_") and guest["connection"]["token"]
    participants = contract_for(call["session_id"])["participants"]
    assert len(participants) == 2
    assert participants[guest["participant_identity"]]["language"] == "en"
    assert participants[guest["participant_identity"]]["display_name"] == "Sam"


def test_the_secret_is_not_stored(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client).json()
        invite = invite_for(client, call)
    secret = invite["code"].split(".", 1)[1]
    assert secret not in str(contract_for(call["session_id"]))
    assert len(secret) >= 20


def test_reopening_the_same_link_is_the_same_guest_and_nobody_else_gets_in(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client).json()
        first = invite_for(client, call)
        a = client.post(f"/calls/invites/{first['code']}/join", json={}).json()
        again = client.post(f"/calls/invites/{first['code']}/join", json={})
        assert again.status_code == 200
        assert again.json()["participant_identity"] == a["participant_identity"]
        # A new code does not displace the seated guest.
        second = invite_for(client, call)
        refused = client.post(f"/calls/invites/{second['code']}/join", json={})
        assert refused.status_code == 409
    assert len(contract_for(call["session_id"])["participants"]) == 2  # never more than two people


def test_a_new_invite_replaces_the_old_code(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client).json()
        old = invite_for(client, call)
        new = invite_for(client, call)
        assert client.get(f"/calls/invites/{old['code']}").status_code == 404
        assert client.get(f"/calls/invites/{new['code']}").status_code == 200


def test_wrong_malformed_and_foreign_codes_are_not_found(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client).json()
        code = invite_for(client, call)["code"]
        session_id, secret = code.split(".", 1)
        for bad in (f"{session_id}.{'x' * len(secret)}", f"{session_id}.short", "nonsense", f"{'0' * 8}-0000-0000-0000-{'0' * 12}.{secret}"):
            assert client.get(f"/calls/invites/{bad}").status_code == 404, bad
            assert client.post(f"/calls/invites/{bad}/join", json={}).status_code == 404, bad


def test_an_invite_expires(runtime, livekit, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_CALL_INVITE_TTL_SECS", "30")
    with TestClient(app) as client:
        call = start(client).json()
        code = invite_for(client, call)["code"]
        session_id = call["session_id"]
        with SessionLocal() as db:
            record = db.get(VoiceSession, session_id)
            contract = dict(record.runtime_contract)
            contract["call_invite"] = {**contract["call_invite"], "expires_at": 1.0}
            record.runtime_contract = contract
            db.commit()
        assert client.get(f"/calls/invites/{code}").status_code == 410
        assert client.post(f"/calls/invites/{code}/join", json={}).status_code == 410


def test_an_ended_call_cannot_be_joined(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client).json()
        code = invite_for(client, call)["code"]
        ended = client.post(f"/voice/sessions/{call['session_id']}/end", headers=bearer(call["session_token"]))
        assert ended.status_code == 200, ended.text
        assert client.post(f"/calls/invites/{code}/join", json={}).status_code == 410
        assert client.post(f"/calls/{call['session_id']}/invite", headers=bearer(call["session_token"])).status_code == 409


def test_either_person_can_hang_up_and_it_ends_for_both(runtime, livekit) -> None:
    with TestClient(app) as client:
        call = start(client).json()
        code = invite_for(client, call)["code"]
        guest = client.post(f"/calls/invites/{code}/join", json={}).json()
        hung_up = client.post(f"/calls/{call['session_id']}/end", headers=bearer(guest["session_token"]))
        assert hung_up.status_code == 200 and hung_up.json()["status"] in {"ended", "failed"}
        # Idempotent, and the creator's own hang-up afterwards is harmless.
        assert client.post(f"/calls/{call['session_id']}/end", headers=bearer(call["session_token"])).status_code == 200
        assert client.get(f"/voice/sessions/{call['session_id']}", headers=bearer(call["session_token"])).json()["status"] in {"ended", "failed"}


def test_creation_is_rate_limited_per_address(runtime, livekit, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_CALLS_PER_IP_HOUR", "2")
    with TestClient(app) as client:
        assert start(client).status_code == 201
        assert start(client).status_code == 201
        limited = start(client)
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1


def test_invitation_attempts_are_rate_limited(runtime, livekit, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_CALL_OPEN_PER_IP_MINUTE", "3")
    with TestClient(app) as client:
        codes = [client.get("/calls/invites/nonsense").status_code for _ in range(5)]
    assert codes[:3] == [404, 404, 404] and codes[3:] == [429, 429]


def test_the_active_call_cap_applies(runtime, livekit, monkeypatch) -> None:
    from src.platform import limits
    from src.platform.models import DEFAULT_ORGANIZATION_ID

    with SessionLocal() as db:  # other tests in this process leave sessions behind
        running = limits.count_active_sessions(db, DEFAULT_ORGANIZATION_ID)
    monkeypatch.setenv("PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG", str(running + 1))
    with TestClient(app) as client:
        first = start(client)
        assert first.status_code == 201
        second = start(client)
        assert second.status_code == 429 and second.json()["detail"]["code"] == "too_many_active_sessions"
        client.post(f"/voice/sessions/{first.json()['session_id']}/end", headers=bearer(first.json()["session_token"]))


class TestWithAuthentication:
    """With PLATFORM_API_KEYS set the three public routes still work without credentials, and nothing else opens."""

    @pytest.fixture(autouse=True)
    def keys(self, monkeypatch):
        monkeypatch.setenv("PLATFORM_API_KEYS", ADMIN)

    def test_public_routes_need_no_credentials_and_admin_routes_stay_closed(self, runtime, livekit) -> None:
        with TestClient(app) as client:
            call = start(client)
            assert call.status_code == 201, call.text
            call = call.json()
            assert client.get("/agents").status_code == 401
            assert client.get("/translations").status_code == 401
            assert client.post(f"/calls/{call['session_id']}/invite").status_code == 401
            assert client.post(f"/calls/{call['session_id']}/end").status_code == 401
            code = invite_for(client, call)["code"]
            assert client.get(f"/calls/invites/{code}").status_code == 200
            guest = client.post(f"/calls/invites/{code}/join", json={}).json()
            # A guest can read their own call but cannot mint invitations, create calls or list anything.
            assert client.post(f"/calls/{call['session_id']}/invite", headers=bearer(guest["session_token"])).status_code == 403
            assert client.get("/agents", headers=bearer(guest["session_token"])).status_code == 403
            # The ordinary end route stays creator-only; hanging up goes through /calls/{id}/end.
            assert client.post(f"/voice/sessions/{call['session_id']}/end", headers=bearer(guest["session_token"])).status_code == 403
            assert client.post(f"/calls/{call['session_id']}/end", headers=bearer(guest["session_token"])).status_code == 200
            assert client.get("/agents", headers=bearer(call["session_token"])).status_code == 403

    def test_a_session_token_only_works_for_its_own_call(self, runtime, livekit) -> None:
        with TestClient(app) as client:
            one, two = start(client).json(), start(client).json()
            other = client.post(f"/calls/{two['session_id']}/invite", headers=bearer(one["session_token"]))
            assert other.status_code == 403
            ended = client.post(f"/voice/sessions/{two['session_id']}/end", headers=bearer(one["session_token"]))
            assert ended.status_code == 403
            assert client.post(f"/calls/{two['session_id']}/end", headers=bearer(one["session_token"])).status_code == 403

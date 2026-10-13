import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.orm import Session as OrmSession

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app
from src.platform.db import SessionLocal
from src.platform.models import Message, ProviderConfig, VoiceSession


class RuntimeResponse:
    def __init__(self, body: dict[str, str]) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return self.body


class RuntimeClient:
    def __init__(
        self, *, response: RuntimeResponse, requests: list[dict[str, object]]
    ) -> None:
        self.response = response
        self.requests = requests

    async def __aenter__(self) -> "RuntimeClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def post(self, url: str, **kwargs: object) -> RuntimeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.response


def test_agent_and_session_flow() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Immigration assistant"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]})
        assert session.status_code == 201
        assert session.json()["conversation_id"]


def test_agent_patch_rejects_null_for_required_name() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Patch validation agent"}).json()

        response = client.patch(f"/agents/{agent['id']}", json={"name": None})

        assert response.status_code == 422


def test_session_creation_hands_off_exact_session_contract(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test/")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    requests: list[dict[str, object]] = []

    def runtime_client(*, timeout: float) -> RuntimeClient:
        assert timeout == 5.0
        return RuntimeClient(
            response=RuntimeResponse({"status": "accepted"}), requests=requests
        )

    monkeypatch.setattr("src.platform.main.httpx.AsyncClient", runtime_client)
    with TestClient(app) as client:
        stt_provider = client.post(
            "/providers",
            json={
                "kind": "stt",
                "provider": "deepgram",
                "configuration": {
                    "model": "nova-3",
                    "language": "multi",
                    "custom": {"option": "must-not-cross-boundary"},
                },
                "secret_reference": "DEEPGRAM_API_KEY",
            },
        ).json()
        llm_provider = client.post(
            "/providers",
            json={
                "kind": "llm",
                "provider": "openai-compatible",
                "configuration": {"model": "gpt-4o-mini", "temperature": 0.4},
                "secret_reference": "OPENAI_API_KEY",
            },
        ).json()
        tts_provider = client.post(
            "/providers",
            json={
                "kind": "tts",
                "provider": "cartesia",
                "configuration": {"model": "sonic-2", "language": "en"},
                "secret_reference": "CARTESIA_API_KEY",
            },
        ).json()
        agent = client.post(
            "/agents",
            json={
                "name": "Handoff agent",
                "system_prompt": "Be concise",
                "voice_id": "voice-7",
                "stt_provider_id": stt_provider["id"],
                "llm_provider_id": llm_provider["id"],
                "tts_provider_id": tts_provider["id"],
            },
        ).json()
        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})

    assert response.status_code == 201
    session = response.json()
    assert session["runtime_status"] == "accepted"
    assert session["session_status"] == "created"
    assert "runtime_joined" not in session
    assert requests == [
        {
            "url": "http://runtime.test/sessions",
            "headers": {"X-Runtime-Token": "shared-secret"},
            "json": {
                "session_id": session["session_id"],
                "conversation_id": session["conversation_id"],
                "agent_id": agent["id"],
                "room_name": session["room_name"],
                "system_prompt": "Be concise",
                "voice_id": "voice-7",
                "providers": {
                    "stt": "deepgram",
                    "llm": "openai-compatible",
                    "tts": "cartesia",
                },
                "provider_settings": {
                    "stt": {"model": "nova-3", "language": "multi"},
                    "llm": {"model": "gpt-4o-mini", "temperature": 0.4},
                    "tts": {"model": "sonic-2", "language": "en"},
                },
            },
        }
    ]


def test_runtime_state_retries_with_original_contract_after_runtime_forgets_it(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    requests: list[dict[str, object]] = []

    class RuntimeStatusClient:
        async def __aenter__(self) -> "RuntimeStatusClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> RuntimeResponse:
            requests.append({"method": "POST", "url": url, **kwargs})
            return RuntimeResponse({"status": "accepted"})

        async def get(self, url: str, **kwargs: object) -> RuntimeResponse:
            requests.append({"method": "GET", "url": url, **kwargs})
            return RuntimeResponse({"status": "not_running"})

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: RuntimeStatusClient(),
    )
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Recoverable agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        updated_agent = client.patch(
            f"/agents/{agent['id']}",
            json={
                "system_prompt": "Changed after session creation",
                "voice_id": "new-voice",
            },
        )
        state = client.get(f"/conversations/{session['conversation_id']}/runtime-state")
        assert [request["method"] for request in requests] == ["POST"]
        retry = client.post(f"/voice/sessions/{session['session_id']}/ensure-running")
        persisted_session = client.get(
            f"/voice/sessions/{session['session_id']}"
        ).json()
        conversation = next(
            item
            for item in client.get("/conversations").json()
            if item["id"] == session["conversation_id"]
        )

    assert state.status_code == 200
    assert state.json()["session_status"] == "created"
    assert retry.status_code == 200
    assert retry.json() == {"session_status": "created", "runtime_status": "accepted"}
    assert state.json()["events"] == []
    assert updated_agent.status_code == 200
    assert persisted_session["status"] == "created"
    assert conversation["status"] == "active"
    assert [request["method"] for request in requests] == ["POST", "GET", "POST"]
    assert requests[1] == {
        "method": "GET",
        "url": f"http://runtime.test/sessions/{session['session_id']}",
        "headers": {"X-Runtime-Token": "shared-secret"},
    }
    assert requests[2]["url"] == "http://runtime.test/sessions"
    assert requests[2]["headers"] == {"X-Runtime-Token": "shared-secret"}
    assert requests[2]["json"] == requests[0]["json"]


def test_runtime_state_preserves_pending_session_when_runtime_status_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")

    class UnavailableRuntimeClient:
        async def __aenter__(self) -> "UnavailableRuntimeClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, _url: str, **_kwargs: object) -> RuntimeResponse:
            return RuntimeResponse({"status": "accepted"})

        async def get(self, _url: str, **_kwargs: object) -> RuntimeResponse:
            raise httpx.ConnectError(
                "Runtime status endpoint is temporarily unavailable"
            )

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: UnavailableRuntimeClient(),
    )
    with TestClient(app) as client:
        agent = client.post(
            "/agents", json={"name": "Unavailable runtime agent"}
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        retry = client.post(f"/voice/sessions/{session['session_id']}/ensure-running")
        state = client.get(f"/conversations/{session['conversation_id']}/runtime-state")

    assert retry.status_code == 503
    assert state.status_code == 200
    assert state.json()["session_status"] == "created"


def test_runtime_state_fails_legacy_session_without_a_persisted_contract(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Legacy agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()

    # Simulate a session created before runtime contracts were persisted.
    with SessionLocal() as db:
        legacy_session = db.get(VoiceSession, session["session_id"])
        assert legacy_session is not None
        legacy_session.runtime_contract = None
        db.commit()

    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    requests: list[str] = []

    class LegacyRuntimeClient:
        async def __aenter__(self) -> "LegacyRuntimeClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, url: str, **_kwargs: object) -> RuntimeResponse:
            requests.append(url)
            return RuntimeResponse({"status": "not_running"})

        async def post(self, *_: object, **_kwargs: object) -> RuntimeResponse:
            pytest.fail("A legacy session must not be restarted without its contract")

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: LegacyRuntimeClient(),
    )
    with TestClient(app) as client:
        state = client.get(f"/conversations/{session['conversation_id']}/runtime-state")
        retry = client.post(f"/voice/sessions/{session['session_id']}/ensure-running")

    assert state.status_code == 200
    assert retry.status_code == 200
    assert retry.json()["session_status"] == "failed"
    assert requests == [f"http://runtime.test/sessions/{session['session_id']}"]


def test_session_created_without_runtime_configuration_can_handoff_later(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    monkeypatch.delenv("PLATFORM_RUNTIME_TOKEN", raising=False)
    with TestClient(app) as client:
        agent = client.post(
            "/agents",
            json={
                "name": "Deferred handoff agent",
                "system_prompt": "Resume this exact contract.",
                "voice_id": "deferred-voice",
            },
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()

    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    sent_contract: list[dict[str, object]] = []

    class RestartedRuntimeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, *_: object, **__: object) -> RuntimeResponse:
            return RuntimeResponse({"status": "not_running"})

        async def post(self, url: str, **kwargs: object) -> RuntimeResponse:
            assert url == "http://runtime.test/sessions"
            sent_contract.append(kwargs["json"])
            return RuntimeResponse({"status": "accepted"})

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: RestartedRuntimeClient(),
    )
    with TestClient(app) as client:
        retry = client.post(f"/voice/sessions/{session['session_id']}/ensure-running")

    assert retry.status_code == 200
    assert retry.json() == {
        "session_status": "created",
        "runtime_status": "accepted",
    }
    assert sent_contract == [
        {
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "room_name": session["room_name"],
            "system_prompt": "Resume this exact contract.",
            "voice_id": "deferred-voice",
        }
    ]


@pytest.mark.parametrize(
    ("runtime_status", "expected_status"),
    [("failed", "failed"), ("ended", "ended"), ("cancelled", "ended")],
)
def test_runtime_state_persists_runtime_terminal_status_when_event_was_lost(
    monkeypatch, runtime_status: str, expected_status: str
) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")

    class TerminalRuntimeClient:
        async def __aenter__(self) -> "TerminalRuntimeClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, _url: str, **_kwargs: object) -> RuntimeResponse:
            return RuntimeResponse({"status": "accepted"})

        async def get(self, _url: str, **_kwargs: object) -> RuntimeResponse:
            return RuntimeResponse({"status": runtime_status})

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: TerminalRuntimeClient(),
    )
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Terminal agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        state = client.post(f"/voice/sessions/{session['session_id']}/ensure-running")
        conversation = client.get(f"/conversations/{session['conversation_id']}").json()

    assert state.status_code == 200
    assert state.json()["session_status"] == expected_status
    assert conversation["status"] == expected_status


def test_session_creation_requires_runtime_auth_before_persisting(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.delenv("PLATFORM_RUNTIME_TOKEN", raising=False)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "No-auth agent"}).json()
        before = len(client.get("/conversations").json())
        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})
        after = len(client.get("/conversations").json())

    assert response.status_code == 503
    assert response.json()["detail"] == "runtime_auth_not_configured"
    assert after == before


def test_session_creation_marks_handoff_rejection_failed(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")

    def runtime_client(*, timeout: float) -> RuntimeClient:
        return RuntimeClient(response=RuntimeResponse({}), requests=[])

    monkeypatch.setattr("src.platform.main.httpx.AsyncClient", runtime_client)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Rejected handoff agent"}).json()
        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})

        assert response.status_code == 503
        conversations = client.get("/conversations").json()
        failed_conversation = next(
            item for item in conversations if item["agent_id"] == agent["id"]
        )
        assert failed_conversation["status"] == "failed"


def test_session_creation_keeps_active_runtime_retry_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")

    def runtime_client(*, timeout: float) -> RuntimeClient:
        return RuntimeClient(
            response=RuntimeResponse({"status": "already_running"}), requests=[]
        )

    monkeypatch.setattr("src.platform.main.httpx.AsyncClient", runtime_client)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Active retry agent"}).json()
        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})

        assert response.status_code == 201
        assert response.json()["runtime_status"] == "already_running"
        assert response.json()["session_status"] == "created"


def test_session_creation_retries_ambiguous_runtime_handoff_with_same_session(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    requests: list[dict[str, object]] = []

    class RuntimeAcceptThenLoseResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> RuntimeResponse:
            requests.append({"url": url, **kwargs})
            if len(requests) == 1:
                raise httpx.ReadTimeout("Runtime accepted but its response was lost")
            return RuntimeResponse({"status": "already_running"})

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: RuntimeAcceptThenLoseResponse(),
    )
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Ambiguous handoff agent"}).json()
        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})

    assert response.status_code == 201
    assert response.json()["runtime_status"] == "already_running"
    assert len(requests) == 2
    assert requests[0]["json"] == requests[1]["json"]


@pytest.mark.parametrize(
    ("runtime_session_status", "expected_status"),
    [("ended", "ended"), ("failed", "failed"), ("cancelled", "ended")],
)
def test_session_creation_reconciles_terminal_runtime_retry(
    monkeypatch,
    runtime_session_status: str,
    expected_status: str,
) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")

    def runtime_client(*, timeout: float) -> RuntimeClient:
        return RuntimeClient(
            response=RuntimeResponse(
                {"status": "already_finished", "session_status": runtime_session_status}
            ),
            requests=[],
        )

    monkeypatch.setattr("src.platform.main.httpx.AsyncClient", runtime_client)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Finished retry agent"}).json()
        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})

        assert response.status_code == 201
        session = response.json()
        assert session["session_status"] == expected_status
        assert session["runtime_status"] == "already_finished"
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == expected_status
        )
        conversation = client.get(f"/conversations/{session['conversation_id']}").json()
        assert conversation["status"] == expected_status


@pytest.mark.parametrize("runtime_status", ["failed", "rejected", "unknown"])
def test_session_creation_fails_closed_for_non_accepted_runtime_status(
    monkeypatch, runtime_status: str
) -> None:
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "shared-secret")
    requests: list[dict[str, object]] = []

    def runtime_client(*, timeout: float) -> RuntimeClient:
        return RuntimeClient(
            response=RuntimeResponse({"status": runtime_status}), requests=requests
        )

    monkeypatch.setattr("src.platform.main.httpx.AsyncClient", runtime_client)
    with TestClient(app) as client:
        agent = client.post(
            "/agents", json={"name": f"{runtime_status} handoff agent"}
        ).json()
        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})

        assert response.status_code == 503
        handed_off = requests[0]["json"]
        assert isinstance(handed_off, dict)
        session_state = client.get(f"/voice/sessions/{handed_off['session_id']}").json()
        conversation = client.get(
            f"/conversations/{handed_off['conversation_id']}"
        ).json()

    assert session_state["status"] == "failed"
    assert session_state["ended_at"] is not None
    assert conversation["status"] == "failed"
    assert conversation["ended_at"] is not None


def test_runtime_event_rejects_agent_that_does_not_own_session(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Session owner"}).json()
        other_agent = client.post("/agents", json={"name": "Different agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "RuntimeStarted",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": other_agent["id"],
            "payload": {"room_name": session["room_name"]},
        }

        response = client.post("/runtime/events", json=event, headers=headers)

        assert response.status_code == 409
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "created"
        )


def test_duplicate_events_are_idempotent(monkeypatch) -> None:
    token = "test-runtime-token"
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", token)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Idempotent event agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "TranscriptFinal",
            "schema_version": 1,
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
        }
        headers = {"X-Runtime-Token": token}
        assert client.post("/runtime/events", json=event, headers=headers).json() == {
            "status": "accepted"
        }
        assert client.post("/runtime/events", json=event, headers=headers).json() == {
            "status": "duplicate"
        }


def test_runtime_event_ingestion_requires_configured_authentication(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "expected-secret")
    with TestClient(app) as client:
        agent = client.post(
            "/agents", json={"name": "Authenticated event agent"}
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "RuntimeStarted",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
        }
        assert client.post("/runtime/events", json=event).status_code == 401
        assert (
            client.post(
                "/runtime/events",
                json=event,
                headers={"X-Runtime-Token": "wrong-secret"},
            ).status_code
            == 401
        )
        assert client.post(
            "/runtime/events",
            json=event,
            headers={"X-Runtime-Token": "expected-secret"},
        ).json() == {"status": "accepted"}


def test_delayed_runtime_started_event_does_not_reactivate_ended_conversation(
    monkeypatch,
) -> None:
    token = "out-of-order-event-token"
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", token)
    headers = {"X-Runtime-Token": token}
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Event ordering agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        identity = {
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
        }

        for event_type, payload in (
            ("RuntimeStarted", {}),
            ("SessionEnded", {"reason": "normal"}),
            ("RuntimeStarted", {}),
        ):
            response = client.post(
                "/runtime/events",
                json={
                    **identity,
                    "event_id": str(uuid4()),
                    "event_type": event_type,
                    "schema_version": 1,
                    "payload": payload,
                },
                headers=headers,
            )
            assert response.status_code == 202
            assert response.json() == {"status": "accepted"}

        conversation = client.get(f"/conversations/{session['conversation_id']}").json()

    assert conversation["status"] == "ended"


def test_session_with_unavailable_provider_does_not_create_orphan_conversation() -> (
    None
):
    with TestClient(app) as client:
        provider_response = client.post(
            "/providers",
            json={
                "kind": "stt",
                "provider": "deepgram",
            },
        )
        assert provider_response.status_code == 201, provider_response.text
        provider = provider_response.json()
        agent_response = client.post(
            "/agents",
            json={
                "name": "Unavailable provider agent",
                "stt_provider_id": provider["id"],
            },
        )
        assert agent_response.status_code == 201, agent_response.text
        agent = agent_response.json()
        with SessionLocal() as db:
            provider_record = db.get(ProviderConfig, provider["id"])
            assert provider_record is not None
            provider_record.enabled = False
            db.commit()
        before = client.get("/conversations").json()

        response = client.post("/voice/sessions", json={"agent_id": agent["id"]})

        after = client.get("/conversations").json()

    assert response.status_code == 409
    assert after == before


def test_runtime_events_are_rejected_when_authentication_is_unconfigured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PLATFORM_RUNTIME_TOKEN", raising=False)
    event = {"event_id": str(uuid4()), "event_type": "RuntimeStarted"}
    with TestClient(app) as client:
        response = client.post("/runtime/events", json=event)
    assert response.status_code == 503
    assert response.json()["detail"] == "runtime_auth_not_configured"


def test_runtime_lifecycle_events_update_conversation_state(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Lifecycle agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "created"
        )
        started = {
            "event_id": str(uuid4()),
            "event_type": "RuntimeStarted",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {"room_name": session["room_name"]},
        }
        ended = {**started, "event_id": str(uuid4()), "event_type": "SessionEnded"}

        assert client.post("/runtime/events", json=started, headers=headers).json() == {
            "status": "accepted"
        }
        session_state = client.get(f"/voice/sessions/{session['session_id']}")
        assert session_state.status_code == 200
        assert session_state.json()["status"] == "running"
        conversations = client.get("/conversations").json()
        assert (
            next(
                item
                for item in conversations
                if item["id"] == session["conversation_id"]
            )["status"]
            == "active"
        )
        assert client.post("/runtime/events", json=ended, headers=headers).json() == {
            "status": "accepted"
        }
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "ended"
        )
        restarted = {**started, "event_id": str(uuid4())}
        assert client.post(
            "/runtime/events", json=restarted, headers=headers
        ).json() == {"status": "accepted"}
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "ended"
        )
        conversations = client.get("/conversations").json()
        assert (
            next(
                item
                for item in conversations
                if item["id"] == session["conversation_id"]
            )["status"]
            == "ended"
        )


@pytest.mark.parametrize(
    "reason",
    [
        "Session ended by request",
        "LiveKit disconnected",
        "Browser participant disconnected",
    ],
)
def test_runtime_disconnect_reasons_end_session_normally(monkeypatch, reason) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Disconnect agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        ended = {
            "event_id": str(uuid4()),
            "event_type": "SessionEnded",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {
                "outcome": "ended",
                "reason": reason,
            },
        }

        response = client.post("/runtime/events", json=ended, headers=headers)

        assert response.status_code == 202
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "ended"
        )


def test_runtime_event_rejects_unknown_types_and_schema_versions(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Validated event agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        base_event = {
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
        }
        invalid_events = [
            {**base_event, "event_type": "UnexpectedEvent", "schema_version": 1},
            {**base_event, "event_type": "RuntimeStarted", "schema_version": 99},
        ]

        for event in invalid_events:
            response = client.post(
                "/runtime/events",
                json={"event_id": str(uuid4()), **event},
                headers=headers,
            )
            assert response.status_code == 422
            assert (
                client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
                == "created"
            )


def test_runtime_event_requires_valid_session_conversation_and_agent_ids(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Event identity agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        base_event = {
            "event_id": str(uuid4()),
            "event_type": "RuntimeStarted",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
        }
        invalid_events = [
            {**base_event, "session_id": None},
            {**base_event, "conversation_id": "not-a-uuid"},
            {**base_event, "agent_id": 123},
        ]

        for event in invalid_events:
            response = client.post("/runtime/events", json=event, headers=headers)
            assert response.status_code == 422
            assert (
                client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
                == "created"
            )


def test_invalid_transcript_event_is_not_acknowledged_as_duplicate(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post(
            "/agents", json={"name": "Transcript validation agent"}
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "TranscriptFinal",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {"role": None, "content": "must not be lost"},
        }

        response = client.post("/runtime/events", json=event, headers=headers)

        assert response.status_code == 422
        assert (
            client.get(f"/conversations/{session['conversation_id']}/messages").json()
            == []
        )


def test_transcript_event_rejects_invalid_language_metadata(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post(
            "/agents", json={"name": "Transcript language validation agent"}
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        for field, invalid_value in (
            ("primary_language", 42),
            ("detected_languages", {"en": True}),
        ):
            event = {
                "event_id": str(uuid4()),
                "event_type": "TranscriptFinal",
                "session_id": session["session_id"],
                "conversation_id": session["conversation_id"],
                "agent_id": agent["id"],
                "payload": {
                    "role": "user",
                    "content": "привет hello",
                    field: invalid_value,
                },
            }

            response = client.post("/runtime/events", json=event, headers=headers)

            assert response.status_code == 422
        assert (
            client.get(f"/conversations/{session['conversation_id']}/messages").json()
            == []
        )


def test_conversation_runtime_state_exposes_persisted_interruptions_and_disconnect(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Runtime history agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event_base = {
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
        }
        events = [
            {
                "event_type": "TranscriptFinal",
                "payload": {"role": "user", "content": "hello"},
            },
            {"event_type": "UserInterrupted", "payload": {"reason": "barge_in"}},
            {
                "event_type": "AgentSpeechStarted",
                "payload": {"text": "Hello there."},
            },
            {
                "event_type": "AgentSpeechCancelled",
                "payload": {"reason": "user_interrupted"},
            },
            {
                "event_type": "SessionEnded",
                "payload": {"reason": "participant_disconnected"},
            },
        ]
        for event in events:
            response = client.post(
                "/runtime/events",
                json={"event_id": str(uuid4()), **event_base, **event},
                headers=headers,
            )
            assert response.json() == {"status": "accepted"}

        runtime_state = client.get(
            f"/conversations/{session['conversation_id']}/runtime-state"
        )

        assert runtime_state.status_code == 200
        body = runtime_state.json()
        assert body["session_id"] == session["session_id"]
        assert body["session_status"] == "ended"
        assert {event["event_type"] for event in body["events"]} == {
            "UserInterrupted",
            "AgentSpeechStarted",
            "AgentSpeechCancelled",
            "SessionEnded",
        }
        disconnect = next(
            event for event in body["events"] if event["event_type"] == "SessionEnded"
        )
        assert disconnect["payload"]["reason"] == "participant_disconnected"


def test_failure_reason_marks_session_and_conversation_failed(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Failed agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "SessionEnded",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {"reason": "provider_unavailable"},
        }
        response = client.post(
            "/runtime/events", json=event, headers={"X-Runtime-Token": "runtime-secret"}
        )
        assert response.json() == {"status": "accepted"}
        session_state = client.get(f"/voice/sessions/{session['session_id']}").json()
        assert session_state["status"] == "failed"
        assert session_state["ended_at"] is not None
        conversations = client.get("/conversations").json()
        failed_conversation = next(
            item for item in conversations if item["id"] == session["conversation_id"]
        )
        assert failed_conversation["status"] == "failed"
        assert failed_conversation["ended_at"] == session_state["ended_at"]


def test_runtime_event_and_projections_roll_back_together_on_commit_failure(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    headers = {"X-Runtime-Token": "runtime-secret"}
    with TestClient(app, raise_server_exceptions=False) as client:
        agent = client.post("/agents", json={"name": "Atomic event agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "TranscriptFinal",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {"role": "user", "content": "hello"},
        }

        def fail_commit(_: OrmSession) -> None:
            raise RuntimeError("simulated database commit failure")

        sqlalchemy_event.listen(OrmSession, "before_commit", fail_commit)
        try:
            failed = client.post("/runtime/events", json=event, headers=headers)
        finally:
            sqlalchemy_event.remove(OrmSession, "before_commit", fail_commit)
        assert failed.status_code == 500
        assert (
            client.get(f"/conversations/{session['conversation_id']}/messages").json()
            == []
        )

        assert client.post("/runtime/events", json=event, headers=headers).json() == {
            "status": "accepted"
        }
        messages = client.get(
            f"/conversations/{session['conversation_id']}/messages"
        ).json()
        assert [message["content"] for message in messages] == ["hello"]

        end_event = {
            **event,
            "event_id": str(uuid4()),
            "event_type": "SessionEnded",
            "payload": {},
        }
        sqlalchemy_event.listen(OrmSession, "before_commit", fail_commit)
        try:
            failed = client.post("/runtime/events", json=end_event, headers=headers)
        finally:
            sqlalchemy_event.remove(OrmSession, "before_commit", fail_commit)
        assert failed.status_code == 500
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "created"
        )
        assert (
            client.get(f"/conversations/{session['conversation_id']}").json()["status"]
            == "active"
        )

        assert client.post(
            "/runtime/events", json=end_event, headers=headers
        ).json() == {"status": "accepted"}
        assert client.post(
            "/runtime/events", json=end_event, headers=headers
        ).json() == {"status": "duplicate"}
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "ended"
        )
        assert (
            client.get(f"/conversations/{session['conversation_id']}").json()["status"]
            == "ended"
        )


def test_explicit_failure_can_upgrade_an_ended_session(monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Late failure agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "SessionFailed",
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {"reason": "pipeline crashed"},
        }
        headers = {"X-Runtime-Token": "runtime-secret"}
        event["event_type"] = "SessionEnded"
        assert client.post("/runtime/events", json=event, headers=headers).json() == {
            "status": "accepted"
        }
        event["event_id"] = str(uuid4())
        event["event_type"] = "SessionFailed"
        assert client.post("/runtime/events", json=event, headers=headers).json() == {
            "status": "accepted"
        }
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "failed"
        )


def test_transcript_event_is_persisted(monkeypatch) -> None:
    token = "test-runtime-token"
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", token)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Transcript agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        event = {
            "event_id": str(uuid4()),
            "event_type": "TranscriptFinal",
            "schema_version": 1,
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {
                "role": "user",
                "content": "Мне нужна appointment",
                "primary_language": "ru",
                "detected_languages": ["ru", "en"],
            },
        }
        assert client.post(
            "/runtime/events", json=event, headers={"X-Runtime-Token": token}
        ).json() == {"status": "accepted"}
        assert client.post(
            "/runtime/events", json=event, headers={"X-Runtime-Token": token}
        ).json() == {"status": "duplicate"}
        messages = client.get(
            f"/conversations/{session['conversation_id']}/messages"
        ).json()
        assert len(messages) == 1
        assert messages[0]["detected_languages"] == ["ru", "en"]


def test_messages_are_returned_in_transcript_order() -> None:
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Ordered transcript agent"}).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        conversation_id = session["conversation_id"]
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with SessionLocal() as db:
            db.add_all(
                [
                    Message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content="Later response",
                        created_at=timestamp + timedelta(seconds=1),
                    ),
                    Message(
                        conversation_id=conversation_id,
                        role="user",
                        content="Earlier question",
                        created_at=timestamp,
                    ),
                ]
            )
            db.commit()

        response = client.get(f"/conversations/{conversation_id}/messages")

        assert [message["content"] for message in response.json()] == [
            "Earlier question",
            "Later response",
        ]


def test_provider_secrets_are_rejected_from_configuration() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/providers",
            json={
                "kind": "stt",
                "provider": "example",
                "configuration": {"api_key": "secret"},
            },
        )
        assert response.status_code == 422


def test_provider_secrets_are_rejected_when_nested_in_configuration() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/providers",
            json={
                "kind": "stt",
                "provider": "example",
                "configuration": {"options": [{"auth": {"api_key": "secret"}}]},
            },
        )
        assert response.status_code == 422


def test_enabled_provider_can_be_attached_to_matching_agent_slot() -> None:
    with TestClient(app) as client:
        provider = client.post(
            "/providers", json={"kind": "stt", "provider": "faster-whisper"}
        ).json()
        response = client.post(
            "/agents",
            json={"name": "Provider agent", "stt_provider_id": provider["id"]},
        )
        assert response.status_code == 201


def test_agent_provider_assignments_can_be_updated_and_cleared() -> None:
    with TestClient(app) as client:
        stt_provider = client.post(
            "/providers", json={"kind": "stt", "provider": "deepgram"}
        ).json()
        replacement_stt = client.post(
            "/providers", json={"kind": "stt", "provider": "faster-whisper"}
        ).json()
        llm_provider = client.post(
            "/providers", json={"kind": "llm", "provider": "openai"}
        ).json()
        agent = client.post(
            "/agents",
            json={"name": "Configurable agent", "stt_provider_id": stt_provider["id"]},
        ).json()

        updated = client.patch(
            f"/agents/{agent['id']}",
            json={
                "stt_provider_id": replacement_stt["id"],
                "llm_provider_id": llm_provider["id"],
            },
        )
        assert updated.status_code == 200
        assert updated.json()["stt_provider_id"] == replacement_stt["id"]
        assert updated.json()["llm_provider_id"] == llm_provider["id"]

        cleared = client.patch(
            f"/agents/{agent['id']}",
            json={"stt_provider_id": None, "llm_provider_id": None},
        )
        assert cleared.status_code == 200
        assert cleared.json()["stt_provider_id"] is None
        assert cleared.json()["llm_provider_id"] is None


def test_agent_provider_update_rejects_wrong_kind_without_changing_assignment() -> None:
    with TestClient(app) as client:
        stt_provider = client.post(
            "/providers", json={"kind": "stt", "provider": "deepgram"}
        ).json()
        llm_provider = client.post(
            "/providers", json={"kind": "llm", "provider": "openai"}
        ).json()
        agent = client.post(
            "/agents",
            json={"name": "Configurable agent", "stt_provider_id": stt_provider["id"]},
        ).json()

        rejected = client.patch(
            f"/agents/{agent['id']}", json={"stt_provider_id": llm_provider["id"]}
        )
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == "invalid_stt_provider_id"
        current = client.get(f"/agents/{agent['id']}")
        assert current.status_code == 200
        assert current.json()["stt_provider_id"] == stt_provider["id"]


def test_ensure_running_recovers_runtime_process_loss_after_session_started(
    monkeypatch,
) -> None:
    token = "running-session-recovery-token"
    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", token)
    start_contracts: list[dict[str, object]] = []

    class RuntimeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get(self, *_: object, **__: object) -> RuntimeResponse:
            return RuntimeResponse({"status": "not_running"})

        async def post(self, url: str, **kwargs: object) -> RuntimeResponse:
            if url.endswith("/sessions"):
                start_contracts.append(kwargs["json"])
            return RuntimeResponse({"status": "accepted"})

    monkeypatch.setattr(
        "src.platform.main.httpx.AsyncClient",
        lambda *, timeout: RuntimeClient(),
    )
    with TestClient(app) as client:
        agent = client.post(
            "/agents",
            json={
                "name": "Running session recovery",
                "system_prompt": "Resume after Runtime restart.",
                "voice_id": "recovery-voice",
            },
        ).json()
        session_response = client.post(
            "/voice/sessions", json={"agent_id": agent["id"]}
        )
        assert session_response.status_code == 201
        session = session_response.json()

        event = {
            "event_id": str(uuid4()),
            "event_type": "RuntimeStarted",
            "schema_version": 1,
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": {"room_name": session["room_name"]},
        }
        started = client.post(
            "/runtime/events",
            json=event,
            headers={"X-Runtime-Token": token},
        )
        assert started.status_code == 202
        assert (
            client.get(f"/voice/sessions/{session['session_id']}").json()["status"]
            == "running"
        )

        recovered = client.post(
            f"/voice/sessions/{session['session_id']}/ensure-running"
        )

    assert recovered.status_code == 200
    assert recovered.json() == {
        "session_status": "running",
        "runtime_status": "accepted",
    }
    assert len(start_contracts) == 2
    assert start_contracts[0] == start_contracts[1]

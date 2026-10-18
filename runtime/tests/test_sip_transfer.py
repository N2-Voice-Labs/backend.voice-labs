"""LiveKit SIP transfer handler with livekit-api mocked (no network)."""

import asyncio
from uuid import uuid4

import pytest

api = pytest.importorskip("livekit.api")
from livekit.protocol.models import ParticipantInfo

from src.runtime import api as runtime_api
from src.runtime import sip_transfer, tool_runtime
from src.runtime.session import RuntimeSession
from src.runtime.sip_transfer import TransferError
from src.runtime.tool_runtime import build_tool_runtime


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("LIVEKIT_SIP_TRUNK_ID", "SIP_TRANSFER_TO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_URL", "ws://lk.test:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setattr(tool_runtime, "transfer_handler", None)


def participant(identity: str, kind=ParticipantInfo.Kind.STANDARD) -> ParticipantInfo:
    return ParticipantInfo(identity=identity, kind=kind)


class FakeLiveKit:
    def __init__(self, participants=(), *, list_error=None, transfer_error=None, status=None):
        self.participants = list(participants)
        self.list_error = list_error
        self.transfer_error = transfer_error
        self.status = api.SIPTransferStatus.STS_TRANSFER_SUCCESSFUL if status is None else status
        self.listed: list = []
        self.transfers: list = []
        outer = self

        class Room:
            async def list_participants(self, request):
                outer.listed.append(request)
                if outer.list_error:
                    raise outer.list_error
                return api.ListParticipantsResponse(participants=outer.participants)

        class Sip:
            async def transfer_sip_participant(self, request):
                outer.transfers.append(request)
                if outer.transfer_error:
                    raise outer.transfer_error
                return api.TransferSIPParticipantResponse(
                    status=outer.status, reason=api.SIPTransferReason.STR_REJECTED
                )

        self.room = Room()
        self.sip = Sip()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def install(monkeypatch, fake):
    monkeypatch.setattr(sip_transfer, "_new_client", lambda: fake)


def make_session() -> RuntimeSession:
    return RuntimeSession(uuid4(), uuid4(), uuid4())


def run_transfer(session, reason="wants a person"):
    return asyncio.run(sip_transfer.transfer(session, reason))


def test_transfers_sip_participant_found_by_identity_prefix(monkeypatch):
    monkeypatch.setenv("SIP_TRANSFER_TO", "+14155550199")
    fake = FakeLiveKit([participant("voice-labs-agent-x"), participant("sip-123")])
    install(monkeypatch, fake)
    session = make_session()

    assert run_transfer(session) == "transferred"

    assert fake.listed[0].room == f"voice-labs-{session.session_id}"
    request = fake.transfers[0]
    assert request.participant_identity == "sip-123"
    assert request.room_name == f"voice-labs-{session.session_id}"
    assert request.transfer_to == "tel:+14155550199"


def test_sip_participant_found_by_kind(monkeypatch):
    monkeypatch.setenv("SIP_TRANSFER_TO", "sip:support@example.com")
    fake = FakeLiveKit([participant("phone-leg", ParticipantInfo.Kind.SIP)])
    install(monkeypatch, fake)
    assert run_transfer(make_session()) == "transferred"
    assert fake.transfers[0].participant_identity == "phone-leg"
    assert fake.transfers[0].transfer_to == "sip:support@example.com"


def test_contract_supplies_room_and_destination(monkeypatch):
    monkeypatch.setenv("SIP_TRANSFER_TO", "+10000000000")
    session = make_session()
    key = str(session.session_id)
    monkeypatch.setitem(
        runtime_api._session_contracts,
        key,
        {"room_name": "custom-room", "transfer_to": "+442071234567"},
    )
    fake = FakeLiveKit([participant("sip-1")])
    install(monkeypatch, fake)

    run_transfer(session)

    assert fake.transfers[0].room_name == "custom-room"
    assert fake.transfers[0].transfer_to == "tel:+442071234567"  # contract beats env


def test_web_call_without_sip_participant_only_records_handoff(monkeypatch):
    monkeypatch.setenv("SIP_TRANSFER_TO", "+14155550199")
    fake = FakeLiveKit([participant("browser-1"), participant("voice-labs-agent-x")])
    install(monkeypatch, fake)

    assert run_transfer(make_session()) == "no_sip_participant"
    assert fake.transfers == []


def test_api_errors_raise_transfer_error(monkeypatch):
    monkeypatch.setenv("SIP_TRANSFER_TO", "+14155550199")
    install(monkeypatch, FakeLiveKit(list_error=RuntimeError("connection reset")))
    with pytest.raises(TransferError, match="connection reset"):
        run_transfer(make_session())

    install(
        monkeypatch,
        FakeLiveKit([participant("sip-1")], transfer_error=api.ServerError("unavailable", "no route", status=503)),
    )
    with pytest.raises(TransferError):
        run_transfer(make_session())


def test_failed_transfer_status_raises(monkeypatch):
    monkeypatch.setenv("SIP_TRANSFER_TO", "+14155550199")
    install(
        monkeypatch,
        FakeLiveKit([participant("sip-1")], status=api.SIPTransferStatus.STS_TRANSFER_FAILED),
    )
    with pytest.raises(TransferError, match="transfer_failed"):
        run_transfer(make_session())


def test_missing_or_invalid_destination(monkeypatch):
    fake = FakeLiveKit([participant("sip-1")])
    install(monkeypatch, fake)
    with pytest.raises(TransferError, match="transfer_not_configured"):
        run_transfer(make_session())
    monkeypatch.setenv("SIP_TRANSFER_TO", "555-1234")
    with pytest.raises(TransferError, match="invalid_transfer_destination"):
        run_transfer(make_session())
    assert fake.transfers == []


def test_install_whenever_livekit_credentials_are_set(monkeypatch):
    # No trunk id and no SIP_TRANSFER_TO: the destination may come per deployment.
    assert sip_transfer.install_if_configured() is True
    assert tool_runtime.transfer_handler is sip_transfer.transfer

    monkeypatch.setattr(tool_runtime, "transfer_handler", None)
    monkeypatch.delenv("LIVEKIT_API_SECRET")
    assert sip_transfer.install_if_configured() is False
    assert tool_runtime.transfer_handler is None

    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.delenv("LIVEKIT_URL")
    assert sip_transfer.install_if_configured() is False


def test_api_lifespan_installs_handler(monkeypatch):
    from fastapi.testclient import TestClient

    # Neither SIP_TRANSFER_TO nor a trunk id is set (clean_env): LiveKit creds suffice.
    with TestClient(runtime_api.app):
        assert tool_runtime.transfer_handler is sip_transfer.transfer


def test_session_contract_accepts_transfer_to():
    request = runtime_api.SessionStart(
        session_id=uuid4(),
        conversation_id=uuid4(),
        agent_id=uuid4(),
        room_name="r",
        transfer_to="+14155550199",
    )
    assert request.model_dump(mode="json")["transfer_to"] == "+14155550199"


def test_transfer_to_human_tool_end_to_end(monkeypatch):
    monkeypatch.setenv("SIP_TRANSFER_TO", "+14155550199")
    fake = FakeLiveKit([participant("sip-9")])
    install(monkeypatch, fake)
    sip_transfer.install_if_configured()
    session = make_session()
    events: list = []
    session.event_client = object()
    session.enqueue_event = events.append  # type: ignore[method-assign]
    runtime = build_tool_runtime(
        [{"name": "transfer_to_human", "description": "Escalate", "tool_type": "builtin"}],
        False,
        session,
    )

    result = asyncio.run(runtime.invoke("transfer_to_human", {"reason": "billing"}))

    assert result["status"] == "ok"
    assert fake.transfers[0].transfer_to == "tel:+14155550199"
    assert "HandoffRequested" in [event.event_type for event in events]

    fake.transfer_error = RuntimeError("boom")
    result = asyncio.run(runtime.invoke("transfer_to_human", {}))
    assert result == {"status": "error", "error": "transfer_failed"}


def test_transfer_tool_reports_not_configured_when_no_destination(monkeypatch):
    """SIP call, no per-deployment destination and no env fallback: the LLM sees an error."""
    fake = FakeLiveKit([participant("sip-9")])
    install(monkeypatch, fake)
    assert sip_transfer.install_if_configured() is True
    session = make_session()
    events: list = []
    session.event_client = object()
    session.enqueue_event = events.append  # type: ignore[method-assign]
    runtime = build_tool_runtime(
        [{"name": "transfer_to_human", "description": "Escalate", "tool_type": "builtin"}],
        False,
        session,
    )

    result = asyncio.run(runtime.invoke("transfer_to_human", {"reason": "billing"}))

    assert result["status"] == "error"
    assert result["error"] == "transfer_not_configured"
    assert "no human" in result["data"]
    assert fake.transfers == []


def test_web_call_without_destination_still_records_handoff(monkeypatch):
    fake = FakeLiveKit([participant("browser-1")])
    install(monkeypatch, fake)
    assert run_transfer(make_session()) == "no_sip_participant"

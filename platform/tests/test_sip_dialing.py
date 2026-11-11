"""Outbound SIP: dial helper, campaign dispatcher, dial-next, and interpreter bridge.

livekit-api and the in-process session service are mocked; nothing touches the network.
"""

import asyncio
import json
import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from livekit import api
from livekit.protocol.sip import SIPParticipantInfo

from src.platform import campaigns, sessions, sip
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import Campaign, CampaignTarget, Conversation


def phone() -> str:
    return f"+1{uuid4().int % 10**10:010d}"


@pytest.fixture
def sip_env(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", "wss://lk.example.test")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("LIVEKIT_SIP_TRUNK_ID", "ST_trunk")
    monkeypatch.setattr(campaigns, "POLL_INTERVAL_S", 0)
    # Tests drive the dispatcher explicitly; moving a campaign to running must not
    # start a background one.
    started: list = []
    monkeypatch.setattr(campaigns, "start_dispatcher", lambda *a: started.append(a))
    return started


@pytest.fixture
def no_sip_env(monkeypatch):
    monkeypatch.delenv("LIVEKIT_SIP_TRUNK_ID", raising=False)


class FakeLiveKit:
    """Stands in for ``livekit.api.LiveKitAPI`` and records every call."""

    def __init__(self, behavior=None):
        self.created: list = []
        self.removed: list = []
        self.urls: list = []
        self.behavior = behavior or {}  # phone number -> exception to raise
        self.on_create = None
        outer = self

        class Sip:
            async def create_sip_participant(self, request):
                outer.created.append(request)
                if outer.on_create:
                    outer.on_create(request)
                error = outer.behavior.get(request.sip_call_to)
                if error:
                    raise error
                return SIPParticipantInfo(
                    participant_id=f"PA_{len(outer.created)}",
                    participant_identity=request.participant_identity,
                    room_name=request.room_name,
                )

        class Room:
            async def remove_participant(self, request):
                outer.removed.append((request.room, request.identity))

        self.sip = Sip()
        self.room = Room()

    def factory(self):
        def new_client():
            return self

        return new_client

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def sip_call_error(code: int, reason: str) -> api.SipCallError:
    return api.SipCallError(
        "unavailable",
        reason,
        status=503,
        metadata={"sip_status_code": str(code), "sip_status": reason},
    )


class FakePlatform:
    """Stands in for the in-process session service the SIP code calls.

    Records one ``(method, path, content)`` tuple per call in ``requests`` (the same
    shape the HTTP mock used) plus the organisation and principal of every context.
    """

    def __init__(self, session_status="ended", fail_create=False, create_errors=None):
        self.session_status = session_status
        self.fail_create = fail_create
        # Scripted HTTPExceptions raised by successive open_voice_session calls (then ok).
        self.create_errors = list(create_errors or [])
        self.create_calls = 0
        self.on_create_call = None
        self.requests: list[tuple[str, str, bytes]] = []
        self.contexts: list = []
        self.sessions = 0

    async def open_voice_session(self, db, ctx, *, agent_id, deployment_id=None, participant=None):
        self.contexts.append(ctx)
        body = json.dumps(
            {"agent_id": str(agent_id), "deployment_id": deployment_id and str(deployment_id)}
        ).encode()
        self.requests.append(("POST", "/voice/sessions", body))
        self.create_calls += 1
        if self.on_create_call is not None:
            self.on_create_call(self.create_calls)
        if self.create_errors:
            raise self.create_errors.pop(0)
        if self.fail_create:
            raise HTTPException(status_code=404, detail="agent_not_found")
        self.sessions += 1
        sid = str(uuid4())
        # A real row: the real service creates one, and PostgreSQL enforces the foreign key
        # from campaign_targets.conversation_id (SQLite does not).
        conversation = Conversation(agent_id=str(agent_id), organization_id=ctx.org_id)
        db.add(conversation)
        db.flush()
        return {
            "session_id": sid,
            "conversation_id": conversation.id,
            "room_name": f"voice-labs-{sid}",
        }

    async def get_session_status(self, db, ctx, session_id):
        self.contexts.append(ctx)
        self.requests.append(("GET", f"/voice/sessions/{session_id}", b""))
        return self.session_status

    async def end_session(self, db, ctx, session_id):
        self.contexts.append(ctx)
        self.requests.append(("POST", f"/voice/sessions/{session_id}/end", b""))
        return {"status": "ended"}

    def install(self, monkeypatch):
        monkeypatch.setattr(sessions, "open_voice_session", self.open_voice_session)
        monkeypatch.setattr(sessions, "get_session_status", self.get_session_status)
        monkeypatch.setattr(sessions, "end_session", self.end_session)

    def ended(self) -> int:
        return sum(1 for _, path, _ in self.requests if path.endswith("/end"))


def make_agent(client, **extra) -> dict:
    response = client.post("/agents", json={"name": f"agent-{uuid4()}", **extra})
    assert response.status_code == 201, response.text
    return response.json()


def make_deployment(client, agent_id: str, channel="sip") -> dict:
    body = {"agent_id": agent_id, "name": f"dep-{uuid4()}", "channel": channel}
    if channel == "sip":
        body["phone_number"] = phone()
    response = client.post("/deployments", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def make_running_campaign(client, targets=1, **overrides) -> tuple[dict, list[str]]:
    """Create a campaign and move it to running; the dispatcher is not started."""
    agent = make_agent(client)
    deployment = make_deployment(client, agent["id"])
    campaign = client.post(
        "/campaigns",
        json={
            "name": f"camp-{uuid4()}",
            "agent_id": agent["id"],
            "deployment_id": deployment["id"],
            **overrides,
        },
    ).json()
    numbers = [phone() for _ in range(targets)]
    client.post(
        f"/campaigns/{campaign['id']}/targets",
        json={"targets": [{"phone_number": n, "name": f"Person {i}"} for i, n in enumerate(numbers)]},
    )
    for step in ("scheduled", "running"):
        response = client.post(f"/campaigns/{campaign['id']}/status", json={"status": step})
        assert response.status_code == 200, response.text
    return campaign, numbers


def targets_of(campaign_id: str) -> list[CampaignTarget]:
    with SessionLocal() as db:
        rows = (
            db.query(CampaignTarget)
            .filter(CampaignTarget.campaign_id == campaign_id)
            .order_by(CampaignTarget.created_at, CampaignTarget.id)
            .all()
        )
        db.expunge_all()
        return rows


def campaign_status(campaign_id: str) -> str:
    with SessionLocal() as db:
        return db.get(Campaign, campaign_id).status


# -- sip.dial -----------------------------------------------------------------------


def test_dial_builds_livekit_request(sip_env, monkeypatch):
    fake = FakeLiveKit()
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    participant = asyncio.run(
        sip.dial("+14155550100", "room-1", "sip-abc", display_name="Ada")
    )
    assert participant == "PA_1"
    request = fake.created[0]
    assert request.sip_trunk_id == "ST_trunk"
    assert request.sip_call_to == "+14155550100"
    assert request.room_name == "room-1"
    assert request.participant_identity == "sip-abc"
    assert request.participant_name == "Ada"
    assert request.wait_until_answered is True
    assert request.krisp_enabled is False


def test_dial_ringing_and_krisp_options(sip_env, monkeypatch):
    monkeypatch.setenv("LIVEKIT_SIP_RINGING_TIMEOUT_S", "20")
    monkeypatch.setenv("LIVEKIT_SIP_KRISP", "1")
    fake = FakeLiveKit()
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    asyncio.run(sip.dial("+14155550100", "r", "sip-1"))
    assert fake.created[0].ringing_timeout.seconds == 20
    assert fake.created[0].krisp_enabled is True


def test_dial_client_uses_http_url(sip_env, monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, url, key, secret):
            seen.update(url=url, key=key, secret=secret)

        async def __aenter__(self):
            return FakeLiveKit()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(api, "LiveKitAPI", Recorder)
    asyncio.run(sip.dial("+14155550100", "r", "sip-1"))
    assert seen == {"url": "https://lk.example.test", "key": "key", "secret": "secret"}
    assert sip.api_url("ws://localhost:7880") == "http://localhost:7880"


def test_dial_failure_is_typed(sip_env, monkeypatch):
    fake = FakeLiveKit({"+1busy": sip_call_error(486, "Busy Here")})
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with pytest.raises(sip.SipDialError) as exc:
        asyncio.run(sip.dial("+1busy", "r", "sip-1"))
    assert exc.value.sip_status_code == 486 and exc.value.no_answer

    fake.behavior["+1bad"] = sip_call_error(404, "Not Found")
    with pytest.raises(sip.SipDialError) as exc:
        asyncio.run(sip.dial("+1bad", "r", "sip-1"))
    assert not exc.value.no_answer

    fake.behavior["+1boom"] = RuntimeError("socket closed")
    with pytest.raises(sip.SipDialError):
        asyncio.run(sip.dial("+1boom", "r", "sip-1"))


def test_dial_requires_configuration(no_sip_env):
    with pytest.raises(sip.SipError, match="sip_not_configured"):
        asyncio.run(sip.dial("+14155550100", "r", "sip-1"))


# -- disabled behaviour -------------------------------------------------------------


def test_sip_disabled_leaves_targets_pending(no_sip_env, monkeypatch):
    started = []
    monkeypatch.setattr(campaigns, "start_dispatcher", lambda *a: started.append(a))
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client, targets=2)
        detail = client.get(f"/campaigns/{campaign['id']}").json()
        assert detail["dialing"] == "disabled"
        assert detail["status"] == "running"
        assert detail["target_counts"] == {"pending": 2}
        response = client.post(f"/campaigns/{campaign['id']}/dial-next")
        assert response.status_code == 409
        assert response.json()["detail"] == "sip_not_configured"
    assert started == []


def test_status_running_starts_dispatcher_when_configured(sip_env):
    started = sip_env
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(
            client, configuration={"max_concurrent_calls": 3}
        )
        assert started == [(campaign["id"], 3)]
        assert client.get(f"/campaigns/{campaign['id']}").json()["dialing"] == "active"


def test_dialing_needs_livekit_credentials(sip_env, monkeypatch):
    monkeypatch.delenv("LIVEKIT_API_SECRET")
    assert not sip.sip_configured()


# -- dispatcher state machine -------------------------------------------------------


def run_batch(campaign_id: str, **kwargs) -> None:
    asyncio.run(campaigns.run_campaign_batch(campaign_id, **kwargs))


def test_batch_dials_all_targets_and_completes_campaign(sip_env, monkeypatch):
    fake = FakeLiveKit()
    platform = FakePlatform()
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    platform.install(monkeypatch)
    with TestClient(app) as client:
        campaign, numbers = make_running_campaign(client, targets=2)
        run_batch(campaign["id"])
        detail = client.get(f"/campaigns/{campaign['id']}").json()
    assert [r.sip_call_to for r in fake.created] == numbers
    rooms = {r.room_name for r in fake.created}
    assert len(rooms) == 2 and all(r.startswith("voice-labs-") for r in rooms)
    assert all(r.participant_identity.startswith("sip-") for r in fake.created)
    rows = targets_of(campaign["id"])
    assert [t.status for t in rows] == ["completed", "completed"]
    assert all(t.attempts == 1 and t.conversation_id for t in rows)
    assert detail["status"] == "completed"
    assert detail["target_counts"] == {"completed": 2}
    # The session request carried the campaign's agent and deployment.
    posts = [c for m, p, c in platform.requests if m == "POST" and p == "/voice/sessions"]
    assert len(posts) == 2 and campaign["deployment_id"].encode() in posts[0]


def test_batch_records_failure_and_no_answer(sip_env, monkeypatch):
    fake = FakeLiveKit()
    platform = FakePlatform()
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    platform.install(monkeypatch)
    with TestClient(app) as client:
        campaign, numbers = make_running_campaign(client, targets=3)
        fake.behavior[numbers[0]] = sip_call_error(404, "Not Found")
        fake.behavior[numbers[1]] = sip_call_error(480, "Temporarily Unavailable")
        run_batch(campaign["id"])
    assert [t.status for t in targets_of(campaign["id"])] == [
        "failed",
        "no_answer",
        "completed",
    ]
    assert platform.ended() == 2  # sessions of unconnected calls are closed
    assert campaign_status(campaign["id"]) == "completed"


def test_batch_retries_until_max_attempts(sip_env, monkeypatch):
    fake = FakeLiveKit()
    platform = FakePlatform()
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    platform.install(monkeypatch)
    with TestClient(app) as client:
        campaign, numbers = make_running_campaign(client, targets=2, max_attempts=3)
        fake.behavior[numbers[0]] = sip_call_error(486, "Busy Here")
        run_batch(campaign["id"])
    first, second = targets_of(campaign["id"])
    assert (first.status, first.attempts) == ("no_answer", 3)
    assert (second.status, second.attempts) == ("completed", 1)
    # Fresh targets are dialed before retries.
    assert [r.sip_call_to for r in fake.created] == [
        numbers[0],
        numbers[1],
        numbers[0],
        numbers[0],
    ]


def test_batch_session_create_failure_marks_failed_without_dialing(sip_env, monkeypatch):
    fake = FakeLiveKit()
    FakePlatform(fail_create=True).install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client)
        run_batch(campaign["id"])
    assert fake.created == []
    (target,) = targets_of(campaign["id"])
    assert (target.status, target.attempts) == ("failed", 1)


DEFERRABLE = [
    HTTPException(429, {"code": "too_many_active_sessions", "message": "cap"}),
    HTTPException(429, {"code": "rate_limited", "message": "slow down"}),
    HTTPException(503, {"code": "runtime_at_capacity", "message": "busy"}),
    HTTPException(503, {"code": "provider_unavailable", "message": "down"}),
    HTTPException(503, {"code": "runtime_unavailable", "message": "down"}),
    HTTPException(503, "runtime_unavailable"),
]


@pytest.mark.parametrize("error", DEFERRABLE, ids=lambda e: str(e.detail)[:40])
def test_batch_defers_targets_on_capacity_errors_without_burning_attempts(
    sip_env, monkeypatch, error
):
    """Admission/capacity refusals return the target to pending (attempt refunded) and
    the worker backs off and retries instead of failing every target."""

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(campaigns, "_defer_sleep", fake_sleep)
    fake = FakeLiveKit()
    # Three refusals in a row, then capacity frees up.
    platform = FakePlatform(create_errors=[error, error, error])
    platform.install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        campaign, numbers = make_running_campaign(client, targets=2, max_attempts=1)
        run_batch(campaign["id"])
    rows = targets_of(campaign["id"])
    assert [t.status for t in rows] == ["completed", "completed"]
    assert [t.attempts for t in rows] == [1, 1]  # refusals did not consume attempts
    assert platform.create_calls == 5
    assert len(sleeps) == 3 and sleeps == sorted(sleeps) and max(sleeps) <= 60
    assert campaign_status(campaign["id"]) == "completed"


def test_campaign_is_not_completed_while_targets_are_deferred(sip_env, monkeypatch):
    fake = FakeLiveKit()
    refusal = HTTPException(429, {"code": "too_many_active_sessions"})
    platform = FakePlatform(create_errors=[refusal] * 50)
    platform.install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())

    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(campaigns, "_defer_sleep", fake_sleep)

    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client, targets=3, max_attempts=2)

        def pause_after_three(call: int) -> None:
            if call == 3:
                client.post(f"/campaigns/{campaign['id']}/status", json={"status": "paused"})

        platform.on_create_call = pause_after_three
        run_batch(campaign["id"], max_concurrent=2)
    rows = targets_of(campaign["id"])
    assert [t.status for t in rows] == ["pending"] * 3
    assert [t.attempts for t in rows] == [0, 0, 0]
    assert campaign_status(campaign["id"]) == "paused"
    assert fake.created == []


def test_defer_backoff_is_exponential_and_capped(monkeypatch):
    monkeypatch.setattr(campaigns, "DEFER_BACKOFF_BASE_S", 2.0)
    monkeypatch.setattr(campaigns, "DEFER_BACKOFF_MAX_S", 60.0)
    delays = [campaigns.defer_delay(n) for n in range(1, 9)]
    assert delays[:4] == [2.0, 4.0, 8.0, 16.0]
    assert delays[-1] == 60.0 and max(delays) == 60.0


def test_batch_hard_session_errors_still_fail_the_target(sip_env, monkeypatch):
    for error in (HTTPException(404, "agent_not_found"), HTTPException(409, "x")):
        fake = FakeLiveKit()
        FakePlatform(create_errors=[error]).install(monkeypatch)
        monkeypatch.setattr(sip, "_new_client", fake.factory())
        with TestClient(app) as client:
            campaign, _ = make_running_campaign(client, max_attempts=1)
            run_batch(campaign["id"])
        (target,) = targets_of(campaign["id"])
        assert (target.status, target.attempts) == ("failed", 1)


def test_dial_next_defers_on_capacity_error(sip_env, monkeypatch):
    refusal = HTTPException(429, {"code": "too_many_active_sessions"})
    FakePlatform(create_errors=[refusal]).install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", FakeLiveKit().factory())
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client, max_attempts=1)
        response = client.post(f"/campaigns/{campaign['id']}/dial-next")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["result"] == "deferred"
        assert (body["target"]["status"], body["target"]["attempts"]) == ("pending", 0)
        assert client.get(f"/campaigns/{campaign['id']}").json()["status"] == "running"


def test_batch_stops_when_campaign_paused(sip_env, monkeypatch):
    fake = FakeLiveKit()
    FakePlatform().install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client, targets=3)

        def pause(_request):
            with SessionLocal() as db:
                db.get(Campaign, campaign["id"]).status = "paused"
                db.commit()

        fake.on_create = pause
        run_batch(campaign["id"])
    assert len(fake.created) == 1
    assert [t.status for t in targets_of(campaign["id"])] == [
        "completed",
        "pending",
        "pending",
    ]
    assert campaign_status(campaign["id"]) == "paused"


def test_batch_waits_for_call_to_end(sip_env, monkeypatch):
    fake = FakeLiveKit()
    platform = FakePlatform(session_status="running")
    platform.install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    monkeypatch.setattr(campaigns, "CALL_WAIT_TIMEOUT_S", 0.5)
    monkeypatch.setattr(campaigns, "POLL_INTERVAL_S", 0.1)
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client)
        run_batch(campaign["id"])
    polls = [1 for m, p, _ in platform.requests if m == "GET"]
    assert len(polls) >= 4  # polled until the wait timeout


def test_batch_concurrency_respects_max_concurrent(sip_env, monkeypatch):
    fake = FakeLiveKit()
    FakePlatform(session_status="running").install(monkeypatch)
    in_flight = peak = 0
    original = sip.dial

    async def counting_dial(*args, **kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.02)
            return await original(*args, **kwargs)
        finally:
            in_flight -= 1

    monkeypatch.setattr(sip, "dial", counting_dial)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    monkeypatch.setattr(campaigns, "CALL_WAIT_TIMEOUT_S", 0)
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client, targets=4)
        run_batch(campaign["id"], max_concurrent=2)
    assert peak == 2
    assert [t.status for t in targets_of(campaign["id"])] == ["completed"] * 4
    assert sorted(t.attempts for t in targets_of(campaign["id"])) == [1, 1, 1, 1]


def test_batch_does_nothing_unless_running(sip_env, monkeypatch):
    fake = FakeLiveKit()
    FakePlatform().install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client)
        client.post(f"/campaigns/{campaign['id']}/status", json={"status": "paused"})
        run_batch(campaign["id"])
    assert fake.created == []
    assert [t.status for t in targets_of(campaign["id"])] == ["pending"]


# -- dial-next ----------------------------------------------------------------------


def test_dial_next_processes_exactly_one_target(sip_env, monkeypatch):
    fake = FakeLiveKit()
    FakePlatform(session_status="running").install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        campaign, numbers = make_running_campaign(client, targets=2)
        response = client.post(f"/campaigns/{campaign['id']}/dial-next")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["result"] == "completed"
        assert body["target"]["phone_number"] == numbers[0]
        assert body["target"]["status"] == "completed"
        assert body["target"]["attempts"] == 1
        assert body["target"]["conversation_id"]
        assert len(fake.created) == 1
        detail = client.get(f"/campaigns/{campaign['id']}").json()
        assert detail["target_counts"] == {"completed": 1, "pending": 1}
        assert detail["status"] == "running"

        assert client.post(f"/campaigns/{campaign['id']}/dial-next").status_code == 200
        assert client.get(f"/campaigns/{campaign['id']}").json()["status"] == "completed"
        again = client.post(f"/campaigns/{campaign['id']}/dial-next")
        assert again.status_code == 409  # campaign no longer running


def test_dial_next_failure_and_state_errors(sip_env, monkeypatch):
    fake = FakeLiveKit()
    FakePlatform().install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        campaign, numbers = make_running_campaign(client, max_attempts=2)
        fake.behavior[numbers[0]] = sip_call_error(486, "Busy Here")
        body = client.post(f"/campaigns/{campaign['id']}/dial-next").json()
        assert body["result"] == "no_answer"
        assert body["target"]["status"] == "pending"  # one retry left
        assert body["target"]["attempts"] == 1
        body = client.post(f"/campaigns/{campaign['id']}/dial-next").json()
        assert body["target"]["status"] == "no_answer"

        client.post(f"/campaigns/{campaign['id']}/status", json={"status": "cancelled"})
        assert client.post(f"/campaigns/{campaign['id']}/dial-next").json()["detail"] == "campaign_not_running"
        assert client.post(f"/campaigns/{uuid4()}/dial-next").status_code == 404


def test_dial_next_without_pending_targets(sip_env, monkeypatch):
    with TestClient(app) as client:
        campaign, _ = make_running_campaign(client)
        with SessionLocal() as db:
            db.query(CampaignTarget).filter(
                CampaignTarget.campaign_id == campaign["id"]
            ).update({"status": "completed"})
            db.commit()
        response = client.post(f"/campaigns/{campaign['id']}/dial-next")
        assert response.status_code == 409
        assert response.json()["detail"] == "no_pending_targets"


# -- interpreter bridge -------------------------------------------------------------


def make_interpreter_deployment(client, channel="sip") -> dict:
    agent = make_agent(
        client,
        mode="interpreter",
        interpreter={"languages": ["en", "ru"], "direction": "auto"},
    )
    return make_deployment(client, agent["id"], channel=channel)


def bridge(client, deployment_id, a="+14155550100", b="+74955550100"):
    return client.post(
        f"/deployments/{deployment_id}/bridge-call",
        json={"from_number": a, "to_number": b},
    )


def test_bridge_call_dials_both_numbers_into_one_room(sip_env, monkeypatch):
    fake = FakeLiveKit()
    platform = FakePlatform()
    platform.install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        deployment = make_interpreter_deployment(client)
        response = bridge(client, deployment["id"])
        assert response.status_code == 201, response.text
        body = response.json()
    assert platform.sessions == 1
    assert [r.sip_call_to for r in fake.created] == ["+14155550100", "+74955550100"]
    assert {r.room_name for r in fake.created} == {body["room_name"]}
    identities = [r.participant_identity for r in fake.created]
    assert len(set(identities)) == 2 and all(i.startswith("sip-") for i in identities)
    assert [p["role"] for p in body["participants"]] == ["from", "to"]
    assert [p["participant_id"] for p in body["participants"]] == ["PA_1", "PA_2"]
    assert body["session_id"] and body["conversation_id"]
    posts = [c for m, p, c in platform.requests if m == "POST" and p == "/voice/sessions"]
    assert deployment["agent_id"].encode() in posts[0]


def test_bridge_call_validation(sip_env, monkeypatch):
    FakePlatform().install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", FakeLiveKit().factory())
    with TestClient(app) as client:
        plain = make_deployment(client, make_agent(client)["id"])
        r = bridge(client, plain["id"])
        assert (r.status_code, r.json()["detail"]) == (422, "agent_not_interpreter")

        web = make_interpreter_deployment(client, channel="web")
        r = bridge(client, web["id"])
        assert (r.status_code, r.json()["detail"]) == (422, "deployment_not_sip")

        interpreter = make_interpreter_deployment(client)
        assert bridge(client, interpreter["id"], a="12345").status_code == 422
        r = bridge(client, interpreter["id"], a="+14155550100", b="+14155550100")
        assert r.json()["detail"] == "bridge_numbers_must_differ"
        assert bridge(client, str(uuid4())).status_code == 404

        client.patch(f"/deployments/{interpreter['id']}", json={"status": "paused"})
        r = bridge(client, interpreter["id"])
        assert (r.status_code, r.json()["detail"]) == (409, "deployment_not_active")


@pytest.mark.parametrize(
    "interpreter",
    [
        {"languages": ["ko", "en"], "direction": "one_way"},
        {"languages": ["en", "es"], "direction": "one_way", "engine": "realtime"},
    ],
)
def test_bridge_call_requires_two_way_cascaded_interpreter(sip_env, monkeypatch, interpreter):
    """Bridge sessions carry no participants, so one-direction interpreters are rejected
    before any session is created or number dialed."""
    fake = FakeLiveKit()
    platform = FakePlatform()
    platform.install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        agent = make_agent(client, mode="interpreter", interpreter=interpreter)
        deployment = make_deployment(client, agent["id"])
        response = bridge(client, deployment["id"])
    assert response.status_code == 422
    assert response.json()["detail"] == "bridge_requires_two_way_interpreter"
    assert platform.sessions == 0
    assert fake.created == []


def test_bridge_call_requires_sip_configuration(no_sip_env):
    with TestClient(app) as client:
        deployment = make_interpreter_deployment(client)
        response = bridge(client, deployment["id"])
        assert response.status_code == 409
        assert response.json()["detail"] == "sip_not_configured"


def test_bridge_call_cleans_up_when_second_leg_fails(sip_env, monkeypatch):
    fake = FakeLiveKit({"+74955550100": sip_call_error(486, "Busy Here")})
    platform = FakePlatform()
    platform.install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        deployment = make_interpreter_deployment(client)
        response = bridge(client, deployment["id"])
    assert response.status_code == 502
    assert response.json()["detail"] == "sip_dial_failed"
    assert len(fake.created) == 2
    assert fake.removed == [(fake.created[0].room_name, fake.created[0].participant_identity)]
    assert platform.ended() == 1


def test_bridge_call_session_failure(sip_env, monkeypatch):
    fake = FakeLiveKit()
    FakePlatform(fail_create=True).install(monkeypatch)
    monkeypatch.setattr(sip, "_new_client", fake.factory())
    with TestClient(app) as client:
        deployment = make_interpreter_deployment(client)
        response = bridge(client, deployment["id"])
    assert response.status_code == 502
    assert response.json()["detail"] == "session_create_failed"
    assert fake.created == []


# -- dispatcher registry ------------------------------------------------------------


def test_start_dispatcher_runs_one_task_per_campaign(monkeypatch):
    calls = []

    async def fake_batch(campaign_id, max_concurrent=1):
        calls.append((campaign_id, max_concurrent))
        await asyncio.sleep(0.05)

    monkeypatch.setattr(campaigns, "run_campaign_batch", fake_batch)

    async def scenario():
        assert campaigns.start_dispatcher("c1", 2) is True
        assert campaigns.start_dispatcher("c1", 2) is False  # already running
        await asyncio.sleep(0.1)
        assert "c1" not in campaigns._dispatchers  # cleaned up when done
        assert campaigns.start_dispatcher("c1", 1) is True  # can resume
        await asyncio.sleep(0.1)

    asyncio.run(scenario())
    assert calls == [("c1", 2), ("c1", 1)]


def test_max_concurrent_calls_is_clamped():
    def record(value):
        return Campaign(configuration={"max_concurrent_calls": value})

    assert campaigns.max_concurrent_calls(Campaign(configuration={})) == 1
    assert campaigns.max_concurrent_calls(record(4)) == 4
    assert campaigns.max_concurrent_calls(record(500)) == campaigns.MAX_CONCURRENT_CALLS
    assert campaigns.max_concurrent_calls(record(0)) == 1
    assert campaigns.max_concurrent_calls(record("many")) == 1

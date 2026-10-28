"""Shared world builder for the cross-organisation isolation harness (organisations plan, T8).

``build_org`` populates one organisation, through the public API wherever possible, with one of
every tenant resource (provider, agent, voice clone/library/design, knowledge document, tool,
agent links, web + SIP deployments, campaign + target, voice session + conversation + messages
+ runtime events). ``OrgWorld.ids`` is the set of every identifier belonging to the org, so a
test can scan a foreign response for leaks.

The module has no pytest dependency: ``make_client`` is a context manager and the tests wrap it
in a fixture.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.db import Base, SessionLocal, engine
from src.platform.main import app
from src.platform.models import (
    CampaignTarget,
    Message,
    RuntimeEvent,
    Voice,
)
from src.platform.organizations import ensure_default_org
from src.platform.tenancy import bypass

SUPER_KEY = "tenancy-isolation-superadmin"
RUNTIME_TOKEN = "tenancy-isolation-runtime-token"
SUPER = {"Authorization": f"Bearer {SUPER_KEY}"}
RUNTIME = {"X-Runtime-Token": RUNTIME_TOKEN}

CONSENT = {
    "confirmed": True,
    "subject_name": "Ada",
    "statement": "I consent to cloning my voice.",
    "method": "self",
}

_LIVE_ENV = (
    "RUNTIME_API_URL",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_SIP_TRUNK_ID",
)


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def phone() -> str:
    return f"+1{uuid4().int % 10**10:010d}"


@contextlib.contextmanager
def make_client(monkeypatch):
    """TestClient with a superadmin key and runtime token, and no live services configured."""

    monkeypatch.setenv("PLATFORM_API_KEYS", SUPER_KEY)
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", RUNTIME_TOKEN)
    for name in _LIVE_ENV:
        monkeypatch.delenv(name, raising=False)
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    with TestClient(app) as client:
        yield client


@dataclass
class OrgWorld:
    """One organisation plus the keys and ids of everything created inside it."""

    org_id: str
    slug: str
    owner: dict[str, str]
    admin: dict[str, str]
    member: dict[str, str]
    # Ids by resource kind; ``extra`` holds identifiers that are not path ids
    # (phone numbers, names, key ids, session tokens, ...).
    ids: dict[str, str] = field(default_factory=dict)
    extra: dict[str, str] = field(default_factory=dict)
    all_ids: set[str] = field(default_factory=set)

    def __getitem__(self, name: str) -> str:
        return self.ids[name]

    @property
    def leak_markers(self) -> set[str]:
        """Every string that must never appear in another org's responses."""

        return set(self.all_ids) | set(self.extra.values())


def _ok(response, status: int, what: str):
    assert response.status_code == status, f"{what}: {response.status_code} {response.text}"
    return response.json()


def make_org_with_keys(client, label: str) -> OrgWorld:
    slug = f"iso-{label}-{uuid4().hex[:10]}"
    created = _ok(
        client.post(
            "/organizations",
            json={"name": f"Org {label}", "slug": slug, "create_owner_key": True},
            headers=SUPER,
        ),
        201,
        "create org",
    )
    owner = bearer(created["owner_key"]["key"])
    world = OrgWorld(
        org_id=created["id"], slug=slug, owner=owner, admin={}, member={}
    )
    world.ids["owner_key"] = created["owner_key"]["id"]
    for role in ("admin", "member"):
        key = _ok(
            client.post(
                "/api-keys", json={"name": f"{label}-{role}", "role": role}, headers=owner
            ),
            201,
            f"create {role} key",
        )
        setattr(world, role, bearer(key["key"]))
        world.ids[f"{role}_key"] = key["id"]
    return world


def build_org(client, label: str) -> OrgWorld:
    """Create an organisation with one of every resource. ``label`` only names things."""

    world = make_org_with_keys(client, label)
    admin, member = world.admin, world.member

    # Providers (llm + tts) and an agent wired to them.
    llm = _ok(
        client.post(
            "/providers",
            json={"kind": "llm", "provider": "openai", "configuration": {}},
            headers=admin,
        ),
        201,
        "llm provider",
    )
    tts = _ok(
        client.post(
            "/providers", json={"kind": "tts", "provider": "test"}, headers=admin
        ),
        201,
        "tts provider",
    )
    stt = _ok(
        client.post(
            "/providers", json={"kind": "stt", "provider": "test"}, headers=admin
        ),
        201,
        "stt provider",
    )
    world.ids["llm_provider"] = llm["id"]
    world.ids["tts_provider"] = tts["id"]
    world.ids["stt_provider"] = stt["id"]

    agent = _ok(
        client.post(
            "/agents",
            json={
                "name": f"agent-{label}-{uuid4().hex[:8]}",
                "llm_provider_id": llm["id"],
                "tts_provider_id": tts["id"],
            },
            headers=admin,
        ),
        201,
        "agent",
    )
    world.ids["agent"] = agent["id"]
    world.extra["agent_name"] = agent["name"]

    # Voices: clone (made ready), library and design.
    clone = _ok(
        client.post(
            "/voices/clone",
            json={
                "name": f"clone-{label}-{uuid4().hex[:8]}",
                "sample_reference": f"https://example.com/{label}-{uuid4().hex}.wav",
                "consent": CONSENT,
            },
            headers=admin,
        ),
        201,
        "voice clone",
    )
    library = _ok(
        client.post(
            "/voices",
            json={
                "name": f"library-{label}",
                "tts_provider_id": tts["id"],
                "provider_voice_id": f"pv-lib-{uuid4().hex[:10]}",
            },
            headers=admin,
        ),
        201,
        "voice library",
    )
    design = _ok(
        client.post(
            "/voices/design",
            json={"name": f"design-{label}", "design_prompt": "calm and warm"},
            headers=admin,
        ),
        201,
        "voice design",
    )
    clone_provider_id = f"pv-clone-{uuid4().hex[:10]}"
    with SessionLocal() as db, bypass(db):
        voice = db.get(Voice, clone["id"])
        voice.status = "ready"
        voice.provider_voice_id = clone_provider_id
        db.commit()
    world.ids.update(
        voice_clone=clone["id"], voice_library=library["id"], voice_design=design["id"]
    )
    world.extra["clone_provider_voice_id"] = clone_provider_id
    world.extra["clone_sample_reference"] = clone["sample_reference"]

    interpreter = _ok(
        client.post(
            "/agents",
            json={
                "name": f"interpreter-{label}-{uuid4().hex[:8]}",
                "mode": "interpreter",
                "interpreter": {
                    "languages": ["en", "ru"],
                    "direction": "one_way",
                    "engine": "cascaded",
                },
            },
            headers=admin,
        ),
        201,
        "interpreter agent",
    )
    world.ids["interpreter_agent"] = interpreter["id"]

    # Knowledge document + tool, linked to the agent.
    doc = _ok(
        client.post(
            "/knowledge/documents",
            json={
                "title": f"doc-{label}-{uuid4().hex[:8]}",
                "content": f"secret refund policy of {label} {uuid4().hex}",
            },
            headers=admin,
        ),
        201,
        "document",
    )
    tool = _ok(
        client.post(
            "/tools",
            json={"name": f"tool_{label}_{uuid4().hex[:8]}", "description": "Look things up"},
            headers=admin,
        ),
        201,
        "tool",
    )
    world.ids.update(document=doc["id"], tool=tool["id"])
    world.extra["document_title"] = doc["title"]
    world.extra["tool_name"] = tool["name"]
    _ok(
        client.put(
            f"/agents/{agent['id']}/knowledge",
            json={"document_ids": [doc["id"]]},
            headers=admin,
        ),
        200,
        "link documents",
    )
    _ok(
        client.put(
            f"/agents/{agent['id']}/tools", json={"tool_ids": [tool["id"]]}, headers=admin
        ),
        200,
        "link tools",
    )

    # Deployments: web and SIP (with a phone number).
    web = _ok(
        client.post(
            "/deployments",
            json={"agent_id": agent["id"], "name": f"web-{label}", "channel": "web"},
            headers=admin,
        ),
        201,
        "web deployment",
    )
    sip_number = phone()
    sip = _ok(
        client.post(
            "/deployments",
            json={
                "agent_id": agent["id"],
                "name": f"sip-{label}",
                "channel": "sip",
                "phone_number": sip_number,
            },
            headers=admin,
        ),
        201,
        "sip deployment",
    )
    world.ids.update(deployment_web=web["id"], deployment=sip["id"])
    world.extra["sip_phone_number"] = sip_number

    # Campaign with one target.
    campaign = _ok(
        client.post(
            "/campaigns",
            json={
                "name": f"campaign-{label}-{uuid4().hex[:8]}",
                "agent_id": agent["id"],
                "deployment_id": sip["id"],
            },
            headers=admin,
        ),
        201,
        "campaign",
    )
    target_number = phone()
    _ok(
        client.post(
            f"/campaigns/{campaign['id']}/targets",
            json={"targets": [{"phone_number": target_number}]},
            headers=admin,
        ),
        200,
        "campaign target",
    )
    with SessionLocal() as db, bypass(db):
        target = (
            db.query(CampaignTarget)
            .filter(CampaignTarget.campaign_id == campaign["id"])
            .one()
        )
        world.ids["campaign_target"] = target.id
    world.ids["campaign"] = campaign["id"]
    world.extra["campaign_name"] = campaign["name"]
    world.extra["target_phone_number"] = target_number

    # Voice session -> conversation, then messages and runtime events via the ingest route.
    session = _ok(
        client.post(
            "/voice/sessions",
            json={"agent_id": agent["id"], "deployment_id": web["id"]},
            headers=member,
        ),
        201,
        "voice session",
    )
    world.ids.update(session=session["session_id"], conversation=session["conversation_id"])
    for key in ("session_token", "room_name"):
        if session.get(key):
            world.extra[key] = session[key]
    transcript = f"hello from {label} {uuid4().hex}"
    world.extra["transcript"] = transcript
    for event_type, payload in (
        ("TranscriptFinal", {"role": "user", "content": transcript}),
        ("TranscriptFinal", {"role": "assistant", "content": f"reply to {label}"}),
        ("LatencyMeasured", {"metric": "llm_ttfb", "value_ms": 321}),
    ):
        event = {
            "event_id": str(uuid4()),
            "event_type": event_type,
            "session_id": session["session_id"],
            "conversation_id": session["conversation_id"],
            "agent_id": agent["id"],
            "payload": payload,
        }
        _ok(client.post("/runtime/events", json=event, headers=RUNTIME), 202, "runtime event")
        world.all_ids.add(event["event_id"])
    with SessionLocal() as db, bypass(db):
        for row in db.query(Message).filter(
            Message.conversation_id == session["conversation_id"]
        ):
            world.all_ids.add(str(row.id))
        for row in db.query(RuntimeEvent).filter(
            RuntimeEvent.conversation_id == session["conversation_id"]
        ):
            world.all_ids.add(str(row.id))

    note = f"problem seen by {label} {uuid4().hex}"
    report = _ok(
        client.post(
            f"/conversations/{session['conversation_id']}/reports",
            json={"note": note},
            headers=member,
        ),
        201,
        "problem report",
    )
    world.ids["report"] = report["id"]
    world.extra["report_note"] = note

    world.all_ids.update(world.ids.values())
    world.all_ids.add(world.org_id)
    world.all_ids.add(world.slug)
    return world

"""No orphan/unbounded hidden agents from ``/translations``; voice/TTS provider matching."""

import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from src.platform import limits
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import Agent, Voice, VoiceSession
from src.platform.tenancy import bypass

from tenancy_fixtures import make_client, make_org_with_keys
from test_interpreter_participants import (  # noqa: F401  (fixtures + helpers)
    create_session,
    make_agent,
    make_ready_voice,
    runtime,
)

BODY = {"languages": ["ko", "en"], "creator": {"language": "ko"}}


@pytest.fixture
def client(monkeypatch):
    with make_client(monkeypatch) as test_client:
        yield test_client


@pytest.fixture
def org(client):
    return make_org_with_keys(client, "ts")


def row_counts(org_id: str) -> tuple[int, int]:
    with SessionLocal() as db, bypass(db):
        agents = db.scalar(
            select(func.count()).select_from(Agent).where(
                Agent.organization_id == org_id, Agent.system.is_(True)
            )
        )
        sessions = db.scalar(
            select(func.count()).select_from(VoiceSession).where(
                VoiceSession.organization_id == org_id
            )
        )
    return agents, sessions


def post(client, org, **extra):
    return client.post("/translations", json={**BODY, **extra}, headers=org.member)


# ---------------------------------------------------------------- orphan rows


def test_refused_by_session_cap_leaves_no_rows(client, org, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG", "1")
    assert post(client, org).status_code == 201
    before = row_counts(org.org_id)
    # A NEW config under a full cap: must not leave its hidden agent behind.
    refused = post(client, org, hear_original=True)
    assert refused.status_code == 429
    assert refused.json()["detail"]["code"] == "too_many_active_sessions"
    assert row_counts(org.org_id) == before


def test_invalid_voice_profile_leaves_no_rows(client, org) -> None:
    before = row_counts(org.org_id)
    response = client.post(
        "/translations",
        json={**BODY, "hear_original": True,
              "creator": {"language": "ko", "voice_profile_id": str(uuid4())}},
        headers=org.member,
    )
    assert (response.status_code, response.json()["detail"]) == (409, "selected_voice_unavailable")
    assert row_counts(org.org_id) == before


def test_preflight_down_leaves_no_rows(client, org, runtime, monkeypatch) -> None:
    from src.platform import health

    async def down(required):
        raise health.HTTPException(status_code=503, detail={"code": "provider_unavailable"})

    monkeypatch.setattr(health, "preflight_providers", down)
    before = row_counts(org.org_id)
    assert post(client, org, hear_original=True).status_code == 503
    assert row_counts(org.org_id) == before


def test_runtime_failure_persists_agent_and_failed_session_consistently(
    client, org, runtime, monkeypatch
) -> None:
    class Dead:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, **kwargs):
            raise httpx.ConnectError("down")

    monkeypatch.setattr("src.platform.main.httpx.AsyncClient", lambda *, timeout: Dead())
    response = post(client, org, hear_original=True)
    assert response.status_code >= 500
    # The handoff happens after the single commit: agent + session committed together, the
    # session marked failed (never an agent without its session, or a session without agent).
    assert row_counts(org.org_id) == (1, 1)
    with SessionLocal() as db, bypass(db):
        session = db.scalars(
            select(VoiceSession).where(VoiceSession.organization_id == org.org_id)
        ).one()
        assert session.status == "failed"


# ---------------------------------------------------------- canonical + bounded


def test_reordered_glossary_shares_one_agent(client, org) -> None:
    g1 = [{"source": "Acme", "keep": True}, {"source": "Zed", "keep": True}]
    for glossary in (g1, list(reversed(g1))):
        response = post(client, org, glossary=glossary)
        assert response.status_code == 201, response.text
    assert row_counts(org.org_id)[0] == 1
    with SessionLocal() as db, bypass(db):  # stored order is the caller's, first one wins
        agent = db.scalars(select(Agent).where(Agent.organization_id == org.org_id)).one()
        assert [e["source"] for e in agent.interpreter["glossary"]] == ["Acme", "Zed"]


def test_per_org_bound_on_translation_configs(client, org, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_MAX_TRANSLATION_CONFIGS", "2")
    assert post(client, org).status_code == 201
    assert post(client, org, hear_original=True).status_code == 201
    over = post(client, org, latency_mode="instant")
    assert over.status_code == 429, over.text
    assert over.json()["detail"]["code"] == "too_many_translation_configs"
    assert row_counts(org.org_id)[0] == 2
    # Reusing an existing config always works.
    assert post(client, org).status_code == 201
    assert post(client, org, hear_original=True).status_code == 201
    assert row_counts(org.org_id)[0] == 2


def test_bound_is_per_org_and_zero_disables(client, org, monkeypatch) -> None:
    other = make_org_with_keys(client, "tsb")
    monkeypatch.setenv("PLATFORM_MAX_TRANSLATION_CONFIGS", "1")
    assert post(client, org).status_code == 201
    assert post(client, org, hear_original=True).status_code == 429
    assert client.post(
        "/translations", json={**BODY, "hear_original": True}, headers=other.member
    ).status_code == 201
    monkeypatch.setenv("PLATFORM_MAX_TRANSLATION_CONFIGS", "0")
    assert post(client, org, hear_original=True).status_code == 201


def test_default_bound_constant() -> None:
    assert limits.NEW_TRANSLATION_CONFIGS_PER_DAY == 50
    assert limits.TRANSLATION_CONFIG_WINDOW_SECONDS == 24 * 3600


def age_system_agents(org_id: str, hours: float) -> None:
    from datetime import timedelta

    with SessionLocal() as db, bypass(db):
        for agent in db.scalars(
            select(Agent).where(Agent.organization_id == org_id, Agent.system.is_(True))
        ):
            agent.created_at = agent.created_at - timedelta(hours=hours)
        db.commit()


def test_bound_is_a_rolling_24h_window(client, org, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_MAX_TRANSLATION_CONFIGS", "2")
    assert post(client, org).status_code == 201
    assert post(client, org, hear_original=True).status_code == 201
    assert post(client, org, latency_mode="instant").status_code == 429
    # Rows just inside the window still count.
    age_system_agents(org.org_id, 23)
    assert post(client, org, latency_mode="instant").status_code == 429
    # Past the window they no longer count: new configs are allowed again.
    age_system_agents(org.org_id, 2)
    assert post(client, org, latency_mode="instant").status_code == 201
    g = [{"source": "Acme", "keep": True}]
    assert post(client, org, glossary=g).status_code == 201  # 2nd fresh config
    assert post(client, org, latency_mode="instant", hear_original=True).status_code == 429
    # Old rows can still be reused.
    assert post(client, org).status_code == 201


def test_old_rows_do_not_count_and_window_is_per_org(client, org, monkeypatch) -> None:
    other = make_org_with_keys(client, "tsw")
    monkeypatch.setenv("PLATFORM_MAX_TRANSLATION_CONFIGS", "1")
    assert post(client, org).status_code == 201
    age_system_agents(org.org_id, 25)
    assert post(client, org, hear_original=True).status_code == 201  # old row ignored
    assert post(client, org, latency_mode="instant").status_code == 429  # fresh one counts
    # The other org's budget is untouched by this org's rows.
    assert client.post("/translations", json=BODY, headers=other.member).status_code == 201


# ------------------------------------------------------ voice/provider mismatch


def provider(client, name: str) -> dict:
    response = client.post("/providers", json={"kind": "tts", "provider": name})
    assert response.status_code == 201, response.text
    return response.json()


def voice_for(client, tts: dict | None, pvid: str) -> dict:
    body = {"name": f"v-{uuid4()}", "provider_voice_id": pvid,
            "tts_provider_id": (tts or provider(client, "cartesia"))["id"]}
    response = client.post("/voices", json=body)
    assert response.status_code == 201, response.text
    created = response.json()
    if tts is None:  # a voice with no provider recorded (legacy rows)
        with SessionLocal() as db, bypass(db):
            db.get(Voice, created["id"]).tts_provider_id = None
            db.commit()
    return created


def test_creator_voice_must_match_agent_tts_provider(runtime) -> None:
    with TestClient(app) as client:
        eleven, cart = provider(client, "elevenlabs"), provider(client, "cartesia")
        agent = make_agent(client, tts_provider_id=eleven["id"])
        bad = voice_for(client, cart, "c1")
        good = voice_for(client, eleven, "e1")
        r = create_session(client, agent, language="en", voice_profile_id=bad["id"])
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        assert create_session(client, agent, language="en", voice_profile_id=good["id"]).status_code == 201


def test_agent_voice_profile_must_match_agent_tts_provider(runtime) -> None:
    with TestClient(app) as client:
        eleven, cart = provider(client, "elevenlabs"), provider(client, "cartesia")
        bad = voice_for(client, cart, "c2")
        agent = client.post(
            "/agents",
            json={"name": "a", "tts_provider_id": eleven["id"], "voice_profile_id": bad["id"]},
        ).json()
        r = client.post("/voice/sessions", json={"agent_id": agent["id"]})
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")


def test_unset_provider_is_compared_against_runtime_default(runtime) -> None:
    from src.platform.health import DEFAULT_PROVIDERS

    assert DEFAULT_PROVIDERS["tts"] == "cartesia"
    with TestClient(app) as client:
        eleven, cart = provider(client, "elevenlabs"), provider(client, "cartesia")
        agent = make_agent(client)  # no tts provider -> Runtime default (cartesia)
        bad = voice_for(client, eleven, "e2")
        r = create_session(client, agent, language="en", voice_profile_id=bad["id"])
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        ok = voice_for(client, cart, "c2b")
        assert create_session(client, agent, language="en", voice_profile_id=ok["id"]).status_code == 201
        # Agent voice profile registered under elevenlabs, agent without provider.
        elevenlabs_agent = client.post(
            "/agents", json={"name": "a", "voice_profile_id": bad["id"]}
        ).json()
        r = client.post("/voice/sessions", json={"agent_id": elevenlabs_agent["id"]})
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        # Guest on a default session: elevenlabs voice rejected, cartesia accepted.
        sid = create_session(client, agent, language="en").json()["session_id"]
        r = client.post(f"/voice/sessions/{sid}/participants",
                        json={"language": "ko", "voice_profile_id": bad["id"]})
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        # /translations without tts_provider_id, creator and invite.
        r = client.post("/translations", json={
            **BODY, "creator": {"language": "ko", "voice_profile_id": bad["id"]}})
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        t = client.post("/translations", json={
            **BODY, "creator": {"language": "ko", "voice_profile_id": ok["id"]}})
        assert t.status_code == 201, t.text
        r = client.post(f"/translations/{t.json()['session_id']}/invite",
                        json={"language": "en", "voice_profile_id": bad["id"]})
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        # Voice without tts_provider_id: skipped on a default session too.
        free = voice_for(client, None, "x0")
        assert create_session(client, agent, language="en", voice_profile_id=free["id"]).status_code == 201
        assert client.post("/translations", json={
            **BODY, "creator": {"language": "ko", "voice_profile_id": free["id"]}}).status_code == 201
        # Voice without tts_provider_id: skipped even with a pinned agent provider.
        pinned = make_agent(client, tts_provider_id=eleven["id"])
        free = voice_for(client, None, "x1")
        assert create_session(client, pinned, language="en", voice_profile_id=free["id"]).status_code == 201


def test_guest_voice_checked_against_contract_providers(runtime) -> None:
    with TestClient(app) as client:
        eleven, cart = provider(client, "elevenlabs"), provider(client, "cartesia")
        agent = make_agent(client, tts_provider_id=eleven["id"])
        sid = create_session(client, agent, language="en").json()["session_id"]
        bad = voice_for(client, cart, "c3")
        r = client.post(
            f"/voice/sessions/{sid}/participants",
            json={"language": "ko", "voice_profile_id": bad["id"]},
        )
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        good = voice_for(client, eleven, "e3")
        ok = client.post(
            f"/voice/sessions/{sid}/participants",
            json={"language": "ko", "voice_profile_id": good["id"]},
        )
        assert ok.status_code == 200, ok.text


def test_guest_voice_without_contract_tts_uses_default(runtime) -> None:
    with TestClient(app) as client:
        cart = provider(client, "cartesia")
        sid = create_session(client, make_agent(client), language="en").json()["session_id"]
        voice = voice_for(client, cart, "c4")
        r = client.post(
            f"/voice/sessions/{sid}/participants",
            json={"language": "ko", "voice_profile_id": voice["id"]},
        )
        assert r.status_code == 200, r.text


def test_provider_lookup_is_org_scoped(client, org, runtime) -> None:
    """A voice pointing at another org's provider row is treated as unknown (skipped)."""

    other = make_org_with_keys(client, "tsc")
    foreign = client.post(
        "/providers", json={"kind": "tts", "provider": "cartesia"}, headers=other.owner
    ).json()
    mine = client.post(
        "/providers", json={"kind": "tts", "provider": "elevenlabs"}, headers=org.owner
    ).json()
    voice = client.post(
        "/voices",
        json={"name": "v", "provider_voice_id": "p1", "tts_provider_id": mine["id"]},
        headers=org.owner,
    ).json()
    with SessionLocal() as db, bypass(db):
        db.get(Voice, voice["id"]).tts_provider_id = foreign["id"]
        db.commit()
    agent = client.post(
        "/agents",
        json={"name": "a", "mode": "interpreter",
              "interpreter": {"languages": ["en", "ko"], "direction": "one_way"},
              "tts_provider_id": mine["id"]},
        headers=org.owner,
    ).json()
    r = client.post(
        "/voice/sessions",
        json={"agent_id": agent["id"],
              "participant": {"language": "en", "voice_profile_id": voice["id"]}},
        headers=org.owner,
    )
    assert r.status_code == 201, r.text


# ------------------------------------------------- selectable TTS provider


def eleven_with_settings(client, headers=None, **kw):
    response = client.post(
        "/providers",
        json={"kind": "tts", "provider": "elevenlabs",
              "configuration": {"model": "eleven_flash_v2_5", "language": "ko"}, **kw},
        headers=headers or {},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_tts_provider_is_sent_to_runtime(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        eleven = eleven_with_settings(client)
        r = client.post("/translations", json={**BODY, "tts_provider_id": eleven["id"]})
        assert r.status_code == 201, r.text
        sent = requests[0]["json"]
        assert sent["mode"] == "interpreter"
        assert sent["providers"] == {"tts": "elevenlabs"}
        assert sent["provider_settings"] == {"tts": {"model": "eleven_flash_v2_5", "language": "ko"}}
        # Without a provider the payload is unchanged: no providers block.
        requests.clear()
        assert client.post("/translations", json=BODY).status_code == 201
        assert "providers" not in requests[0]["json"]


def test_different_providers_get_different_hidden_agents_and_none_is_stable(client, org) -> None:
    from src.platform.translations import interpreter_config, system_agent_for, TranslationIn
    import hashlib, json

    eleven = eleven_with_settings(client, org.owner)
    assert post(client, org).status_code == 201
    assert post(client, org, tts_provider_id=eleven["id"]).status_code == 201
    assert post(client, org, tts_provider_id=eleven["id"]).status_code == 201
    assert row_counts(org.org_id)[0] == 2
    # Stability: a config without a provider keeps the pre-existing key derivation.
    interpreter = interpreter_config(TranslationIn(**BODY))
    expected = "translator:" + hashlib.sha256(
        json.dumps(interpreter, sort_keys=True).encode()
    ).hexdigest()[:32]
    with SessionLocal() as db, bypass(db):
        keys = {a.system_key for a in db.scalars(select(Agent).where(Agent.system.is_(True), Agent.organization_id == org.org_id))}
        assert expected in keys


def test_invalid_tts_provider_ids_are_indistinguishable(client, org) -> None:
    other = make_org_with_keys(client, "tsd")
    foreign = eleven_with_settings(client, other.owner)
    llm = client.post("/providers", json={"kind": "llm", "provider": "openai"}, headers=org.owner).json()
    disabled = eleven_with_settings(client, org.owner)
    from src.platform.models import ProviderConfig

    with SessionLocal() as db, bypass(db):
        db.get(ProviderConfig, disabled["id"]).enabled = False
        db.commit()
    before = row_counts(org.org_id)
    answers = set()
    for pid in (foreign["id"], str(uuid4()), llm["id"], disabled["id"]):
        r = post(client, org, tts_provider_id=pid)
        answers.add((r.status_code, r.text))
    assert answers == {(422, '{"detail":"invalid_tts_provider_id"}')}
    assert row_counts(org.org_id) == before


def test_translation_voice_must_match_selected_provider(runtime) -> None:
    with TestClient(app) as client:
        eleven, cart = provider(client, "elevenlabs"), provider(client, "cartesia")
        bad, good = voice_for(client, cart, "c5"), voice_for(client, eleven, "e5")
        r = client.post("/translations", json={
            **BODY, "tts_provider_id": eleven["id"],
            "creator": {"language": "ko", "voice_profile_id": bad["id"]}})
        assert (r.status_code, r.json()["detail"]) == (409, "voice_provider_mismatch")
        ok = client.post("/translations", json={
            **BODY, "tts_provider_id": eleven["id"],
            "creator": {"language": "ko", "voice_profile_id": good["id"]}})
        assert ok.status_code == 201, ok.text
        sid = ok.json()["session_id"]
        guest = client.post(f"/translations/{sid}/invite",
                            json={"language": "en", "voice_profile_id": bad["id"]})
        assert (guest.status_code, guest.json()["detail"]) == (409, "voice_provider_mismatch")
        assert client.post(f"/translations/{sid}/invite",
                           json={"language": "en", "voice_profile_id": good["id"]}).status_code == 200


def test_provider_configs_count_toward_the_bound(client, org, monkeypatch) -> None:
    monkeypatch.setenv("PLATFORM_MAX_TRANSLATION_CONFIGS", "1")
    eleven = eleven_with_settings(client, org.owner)
    assert post(client, org).status_code == 201
    r = post(client, org, tts_provider_id=eleven["id"])
    assert (r.status_code, r.json()["detail"]["code"]) == (429, "too_many_translation_configs")


# ---------------------------------------------- ElevenLabs STT + Gemini translation + voice


def record(client, kind: str, name: str, headers=None, **kw) -> dict:
    response = client.post(
        "/providers", json={"kind": kind, "provider": name, **kw}, headers=headers or {}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_stt_llm_and_tts_providers_are_sent_to_runtime(runtime) -> None:
    requests, _ = runtime
    with TestClient(app) as client:
        stt = record(client, "stt", "elevenlabs", configuration={"language": "ko"})
        llm = record(client, "llm", "gemini", configuration={"model": "my-model"})
        tts = record(client, "tts", "elevenlabs")
        r = client.post("/translations", json={
            **BODY, "stt_provider_id": stt["id"], "llm_provider_id": llm["id"],
            "tts_provider_id": tts["id"],
        })
        assert r.status_code == 201, r.text
        sent = requests[0]["json"]
        assert sent["providers"] == {"stt": "elevenlabs", "llm": "gemini", "tts": "elevenlabs"}
        assert sent["provider_settings"]["llm"] == {"model": "my-model"}
        assert sent["provider_settings"]["stt"] == {"language": "ko"}


def test_each_provider_slot_gets_its_own_hidden_agent_and_old_keys_are_stable(client, org) -> None:
    stt = record(client, "stt", "elevenlabs", org.owner)
    llm = record(client, "llm", "gemini", org.owner)
    assert post(client, org).status_code == 201
    assert post(client, org, stt_provider_id=stt["id"]).status_code == 201
    assert post(client, org, llm_provider_id=llm["id"]).status_code == 201
    assert post(client, org, stt_provider_id=stt["id"], llm_provider_id=llm["id"]).status_code == 201
    assert post(client, org, stt_provider_id=stt["id"], llm_provider_id=llm["id"]).status_code == 201
    assert row_counts(org.org_id)[0] == 4  # none, stt, llm, stt+llm (the repeat reused one)


@pytest.mark.parametrize("kind", ["stt", "llm"])
def test_invalid_stt_and_llm_provider_ids_are_indistinguishable(client, org, kind) -> None:
    other = make_org_with_keys(client, f"x{kind}")
    foreign = record(client, kind, "elevenlabs" if kind == "stt" else "gemini", other.owner)
    wrong_kind = record(client, "tts", "elevenlabs", org.owner)
    disabled = record(client, kind, "elevenlabs" if kind == "stt" else "gemini", org.owner)
    from src.platform.models import ProviderConfig

    with SessionLocal() as db, bypass(db):
        db.get(ProviderConfig, disabled["id"]).enabled = False
        db.commit()
    before = row_counts(org.org_id)
    answers = set()
    for pid in (foreign["id"], str(uuid4()), wrong_kind["id"], disabled["id"]):
        r = post(client, org, **{f"{kind}_provider_id": pid})
        answers.add((r.status_code, r.text))
    assert answers == {(422, f'{{"detail":"invalid_{kind}_provider_id"}}')}
    assert row_counts(org.org_id) == before


def test_preflight_checks_the_selected_stt_and_llm_providers(client, org, monkeypatch) -> None:
    seen = []

    async def fake_preflight(required):
        seen.append(required)

    monkeypatch.setenv("RUNTIME_API_URL", "http://runtime.test")
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "t")
    monkeypatch.setattr("src.platform.health.preflight_providers", fake_preflight)

    class Boom(Exception):
        pass

    async def stop(*args, **kwargs):
        raise Boom

    monkeypatch.setattr("src.platform.translations.open_voice_session", stop)
    stt = record(client, "stt", "elevenlabs", org.owner)
    llm = record(client, "llm", "gemini", org.owner)
    with pytest.raises(Boom):
        post(client, org, stt_provider_id=stt["id"], llm_provider_id=llm["id"])
    assert seen == [[("stt", "elevenlabs"), ("llm", "gemini"), ("tts", "cartesia")]]

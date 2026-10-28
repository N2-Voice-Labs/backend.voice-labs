"""Platform API keys and session tokens (RM-004)."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from src.platform import auth
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import VoiceSession

KEY = "admin-key-1"
KEY2 = "admin-key-2"


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


ADMIN = bearer(KEY)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", f" {KEY} , {KEY2},,")
    monkeypatch.delenv("RUNTIME_API_URL", raising=False)
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "runtime-secret")
    with TestClient(app) as test_client:
        yield test_client


def make_interpreter_agent(client: TestClient) -> dict:
    response = client.post(
        "/agents",
        headers=ADMIN,
        json={
            "name": f"interp-{uuid4()}",
            "mode": "interpreter",
            "interpreter": {"languages": ["en", "ko"], "direction": "one_way"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def make_session(client: TestClient, agent: dict | None = None) -> dict:
    agent = agent or make_interpreter_agent(client)
    response = client.post(
        "/voice/sessions",
        headers=ADMIN,
        json={"agent_id": agent["id"], "participant": {"language": "en"}},
    )
    assert response.status_code == 201, response.text
    return response.json()


def scoped_requests(session: dict) -> list[tuple[str, str]]:
    sid, cid = session["session_id"], session["conversation_id"]
    return [
        ("GET", f"/voice/sessions/{sid}"),
        ("GET", f"/conversations/{cid}"),
        ("GET", f"/conversations/{cid}/messages"),
        ("GET", f"/conversations/{cid}/runtime-state"),
        ("POST", f"/voice/sessions/{sid}/ensure-running"),
    ]


# -- admin API keys ----------------------------------------------------------------------


def test_missing_key_is_401_authentication_required(client) -> None:
    response = client.get("/agents")
    assert response.status_code == 401
    assert response.json() == {"detail": "authentication_required"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header", ["Bearer wrong", "Basic abc", f"Bearer", KEY, f"Token {KEY}"]
)
def test_wrong_or_malformed_key_is_401_invalid_api_key(client, header) -> None:
    response = client.get("/agents", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_api_key"}


def test_every_configured_key_works_and_scheme_is_case_insensitive(client) -> None:
    assert client.get("/agents", headers=ADMIN).status_code == 200
    assert client.get("/agents", headers=bearer(KEY2)).status_code == 200
    assert client.get("/agents", headers={"Authorization": f"bearer {KEY}"}).status_code == 200


def test_health_is_public(client) -> None:
    assert client.get("/health").status_code == 200


def test_runtime_token_routes_do_not_need_the_admin_key(client) -> None:
    # Reaches the route's own runtime-token check instead of the admin check.
    response = client.post("/runtime/events", json={})
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_runtime_token"}
    response = client.post(
        f"/runtime/agents/{uuid4()}/knowledge/search", json={"query": "x"}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_runtime_token"}
    # The runtime token is not an admin credential.
    response = client.get("/agents", headers={"X-Runtime-Token": "runtime-secret"})
    assert response.json() == {"detail": "authentication_required"}


def test_admin_key_does_not_replace_runtime_token(client) -> None:
    response = client.post("/runtime/events", json={}, headers=ADMIN)
    assert response.json() == {"detail": "invalid_runtime_token"}


def test_all_other_routes_require_authentication(client) -> None:
    exempt = auth.PUBLIC_ROUTES | auth.RUNTIME_TOKEN_ROUTES
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            if (method, route.path) in exempt:
                continue
            path = route.path.replace("{", "").replace("}", "")  # placeholder segments
            response = client.request(method, path)
            assert response.status_code == 401, (method, route.path, response.text)
            assert response.json() == {"detail": "authentication_required"}


def test_preflight_allows_authorization_header(client) -> None:
    response = client.options(
        "/agents",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 200
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


@pytest.mark.parametrize("value", [None, "", " , "])
def test_auth_disabled_when_keys_unset_or_empty(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("PLATFORM_API_KEYS", raising=False)
    else:
        monkeypatch.setenv("PLATFORM_API_KEYS", value)
    with TestClient(app) as client:
        assert client.get("/agents").status_code == 200
        agent = client.post(
            "/agents",
            json={
                "name": f"interp-{uuid4()}",
                "mode": "interpreter",
                "interpreter": {"languages": ["en", "ko"], "direction": "one_way"},
            },
        ).json()
        session = client.post("/voice/sessions", json={"agent_id": agent["id"]}).json()
        # Tokens are still issued but never required.
        assert session["session_token"].startswith(auth.SESSION_TOKEN_PREFIX)
        for method, path in scoped_requests(session):
            assert client.request(method, path).status_code == 200


def test_startup_warns_once_when_disabled(monkeypatch, caplog) -> None:
    monkeypatch.delenv("PLATFORM_API_KEYS", raising=False)
    with caplog.at_level("WARNING", logger="src.platform.auth"):
        with TestClient(app):
            pass
    assert sum("PLATFORM_API_KEYS is not set" in r.message for r in caplog.records) == 1


# -- session tokens ----------------------------------------------------------------------


def test_session_creation_requires_admin_key_and_returns_token(client) -> None:
    agent = make_interpreter_agent(client)
    assert client.post("/voice/sessions", json={"agent_id": agent["id"]}).status_code == 401
    session = make_session(client, agent)
    token = session["session_token"]
    assert token.startswith(auth.SESSION_TOKEN_PREFIX) and len(token) > 40
    # A session token cannot create sessions or call admin routes.
    for response in (
        client.post("/voice/sessions", json={"agent_id": agent["id"]}, headers=bearer(token)),
        client.get("/agents", headers=bearer(token)),
        client.get("/conversations", headers=bearer(token)),
    ):
        assert response.status_code == 403
        assert response.json() == {"detail": "forbidden"}


def test_only_the_token_hash_is_stored(client) -> None:
    session = make_session(client)
    with SessionLocal() as db:
        row = db.get(VoiceSession, session["session_id"])
        assert row.session_token_hash == auth.hash_token(session["session_token"])
        assert session["session_token"] not in str(row.runtime_contract)
        assert session["session_token"] not in (row.session_token_hash or "")


def test_session_token_works_for_its_own_session_only(client) -> None:
    mine, other = make_session(client), make_session(client)
    headers = bearer(mine["session_token"])
    for method, path in scoped_requests(mine):
        assert client.request(method, path, headers=headers).status_code == 200, path
    assert client.get(
        f"/voice/sessions/{mine['session_id']}", headers=headers
    ).json()["id"] == mine["session_id"]
    for method, path in scoped_requests(other):
        response = client.request(method, path, headers=headers)
        assert response.status_code == 403, path
        assert response.json() == {"detail": "forbidden"}
    # Unknown session ids do not leak existence to a session token.
    response = client.get(f"/voice/sessions/{uuid4()}", headers=headers)
    assert response.status_code == 403


def test_session_token_can_end_its_session(client) -> None:
    session = make_session(client)
    other = make_session(client)
    headers = bearer(session["session_token"])
    assert client.post(f"/voice/sessions/{other['session_id']}/end", headers=headers).status_code == 403
    response = client.post(f"/voice/sessions/{session['session_id']}/end", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "ended"}


def test_session_token_is_rejected_on_unscoped_routes(client) -> None:
    session = make_session(client)
    headers = bearer(session["session_token"])
    cid = session["conversation_id"]
    for method, path in [
        ("PUT", f"/conversations/{cid}/outcome"),
        ("GET", "/analytics/summary"),
        ("GET", "/voices"),
        ("POST", "/voices/clone"),
    ]:
        assert client.request(method, path, headers=headers).status_code == 403, path


def test_admin_key_still_works_on_scoped_routes(client) -> None:
    session = make_session(client)
    for method, path in scoped_requests(session):
        assert client.request(method, path, headers=ADMIN).status_code == 200, path
    assert client.get(f"/voice/sessions/{uuid4()}", headers=ADMIN).status_code == 404


def test_tampered_session_token_is_forbidden(client) -> None:
    session = make_session(client)
    bad = session["session_token"][:-2] + "xx"
    response = client.get(f"/voice/sessions/{session['session_id']}", headers=bearer(bad))
    assert response.status_code == 403


def test_guest_tokens(client) -> None:
    session = make_session(client)
    sid = session["session_id"]
    creator = bearer(session["session_token"])

    # The creator's token adds the guest; admin can too.
    added = client.post(
        f"/voice/sessions/{sid}/participants", json={"language": "ko"}, headers=creator
    )
    assert added.status_code == 200, added.text
    guest_token = added.json()["session_token"]
    assert guest_token.startswith(auth.SESSION_TOKEN_PREFIX)
    assert guest_token != session["session_token"]

    with SessionLocal() as db:
        row = db.get(VoiceSession, sid)
        assert row.guest_token_hashes == [auth.hash_token(guest_token)]
        assert guest_token not in str(row.runtime_contract)

    guest = bearer(guest_token)
    # The guest can use the scoped routes of its own session ...
    assert client.get(f"/voice/sessions/{sid}", headers=guest).status_code == 200
    assert (
        client.get(f"/conversations/{session['conversation_id']}/messages", headers=guest).status_code
        == 200
    )
    # ... but cannot add participants, and not touch another session.
    response = client.post(
        f"/voice/sessions/{sid}/participants", json={"language": "en"}, headers=guest
    )
    assert response.status_code == 403
    assert response.json() == {"detail": "forbidden"}
    other = make_session(client)
    assert client.get(f"/voice/sessions/{other['session_id']}", headers=guest).status_code == 403
    # Wrong-session creator token cannot add participants either.
    response = client.post(
        f"/voice/sessions/{other['session_id']}/participants",
        json={"language": "ko"},
        headers=creator,
    )
    assert response.status_code == 403


def test_admin_adding_participant_gets_a_guest_token(client) -> None:
    session = make_session(client)
    sid = session["session_id"]
    added = client.post(
        f"/voice/sessions/{sid}/participants", json={"language": "ko"}, headers=ADMIN
    )
    assert added.status_code == 200
    assert client.get(
        f"/voice/sessions/{sid}", headers=bearer(added.json()["session_token"])
    ).status_code == 200


# -- migration 0007 ----------------------------------------------------------------------


def run_alembic(db_url: str, *args: str) -> None:
    platform_dir = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PLATFORM_DATABASE_URL": db_url,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("PLATFORM_API_KEYS", None)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=platform_dir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_migration_0007_up_and_down_on_sqlite() -> None:
    cache = Path.home() / ".cache" / "voice-labs-test"
    cache.mkdir(parents=True, exist_ok=True)
    db_path = cache / f"auth-migration-{uuid4().hex}.db"
    url = f"sqlite:///{db_path}"
    try:
        run_alembic(url, "upgrade", "0006_agent_translation")
        engine = create_engine(url)
        columns = {c["name"] for c in inspect(engine).get_columns("voice_sessions")}
        assert not {"session_token_hash", "guest_token_hashes"} & columns

        run_alembic(url, "upgrade", "head")
        columns = {c["name"] for c in inspect(engine).get_columns("voice_sessions")}
        assert {"session_token_hash", "guest_token_hashes"} <= columns

        run_alembic(url, "downgrade", "0006_agent_translation")
        columns = {c["name"] for c in inspect(engine).get_columns("voice_sessions")}
        assert not {"session_token_hash", "guest_token_hashes"} & columns
        engine.dispose()
    finally:
        db_path.unlink(missing_ok=True)


def test_unknown_vlk_key_is_generic_401(client) -> None:
    response = client.get("/agents", headers=bearer("vlk_does-not-exist"))
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_api_key"}
    assert response.headers["www-authenticate"] == "Bearer"


def test_admin_key_with_unknown_org_header_is_404(client) -> None:
    response = client.get("/agents", headers={**ADMIN, "X-Organization-Id": str(uuid4())})
    assert response.status_code == 404
    assert response.json() == {"detail": {"code": "organization_not_found"}}


def test_guest_cannot_end_the_creators_session(client) -> None:
    session = make_session(client)
    sid = session["session_id"]
    added = client.post(
        f"/voice/sessions/{sid}/participants",
        json={"language": "ko"},
        headers=bearer(session["session_token"]),
    )
    guest = bearer(added.json()["session_token"])
    response = client.post(f"/voice/sessions/{sid}/end", headers=guest)
    assert response.status_code == 403
    assert response.json() == {"detail": "forbidden"}
    # The creator (and admin) still can.
    assert client.post(f"/voice/sessions/{sid}/end", headers=bearer(session["session_token"])).status_code == 200


def test_cors_preflight_allows_org_and_request_id_headers(client) -> None:
    response = client.options(
        "/agents",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-organization-id, x-request-id, authorization",
        },
    )
    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"].lower()
    assert "x-organization-id" in allowed and "x-request-id" in allowed


def test_request_id_is_echoed_as_response_header(client) -> None:
    echoed = client.get("/agents", headers={**ADMIN, "X-Request-Id": "req-abc"})
    assert echoed.headers["x-request-id"] == "req-abc"
    generated = client.get("/agents", headers=ADMIN)
    assert len(generated.headers["x-request-id"]) == 36
    rejected = client.get("/agents", headers={"X-Request-Id": "req-rej"})
    assert rejected.status_code == 401 and rejected.headers["x-request-id"] == "req-rej"
    cors = client.get(
        "/agents", headers={**ADMIN, "Origin": "http://localhost:3000"}
    )
    assert "x-request-id" in cors.headers["access-control-expose-headers"].lower()

"""AuthContext, organisation API keys, and role guards (organisations plan, T3)."""

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from src.platform import auth
from src.platform.context import AuthContext, get_ctx, require, role_at_least
from src.platform.db import Base, SessionLocal, engine
from src.platform.organizations import ensure_default_org
from src.platform.models import (
    DEFAULT_ORGANIZATION_ID,
    ApiKey,
    Organization,
    VoiceSession,
)

ADMIN_KEY = "admin-key-1"
ADMIN = {"Authorization": f"Bearer {ADMIN_KEY}"}

# A tiny router so the tests can observe the context and exercise require().
_router = APIRouter(prefix="/__test_ctx")


@_router.get("/whoami")
def _whoami(ctx: AuthContext = Depends(get_ctx)) -> dict:
    return {
        "org_id": ctx.org_id,
        "principal": ctx.principal,
        "role": ctx.role,
        "key_id": ctx.key_id,
        "session_id": ctx.session_id,
        "is_guest": ctx.is_guest,
        "request_id": ctx.request_id,
    }


@_router.get("/admin-only")
def _admin_only(ctx: AuthContext = Depends(require("admin"))) -> dict:
    return {"ok": True}


@_router.get("/owner-only")
def _owner_only(ctx: AuthContext = Depends(require("owner"))) -> dict:
    return {"ok": True}


@_router.get("/sessions/{session_id}")
def _session_probe(session_id: str, ctx: AuthContext = Depends(get_ctx)) -> dict:
    return {"org_id": ctx.org_id, "principal": ctx.principal, "is_guest": ctx.is_guest}


# A private app: the real app's route table stays untouched for other test modules.
app = FastAPI(dependencies=[Depends(auth.authenticate)])
app.include_router(_router)

WHOAMI = "/__test_ctx/whoami"


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def schema(monkeypatch):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_default_org(db)
    # SESSION_SCOPED_ROUTES is keyed by route path; let session tokens reach the probe.
    monkeypatch.setitem(
        auth.SESSION_SCOPED_ROUTES,
        ("GET", "/__test_ctx/sessions/{session_id}"),
        auth.SessionScope("session_id", "session"),
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEYS", ADMIN_KEY)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def dev_client(monkeypatch):
    monkeypatch.delenv("PLATFORM_API_KEYS", raising=False)
    with TestClient(app) as test_client:
        yield test_client


def make_org(status: str = "active") -> str:
    org_id = str(uuid4())
    with SessionLocal() as db:
        db.add(Organization(id=org_id, name="Org", slug=f"org-{org_id}", status=status))
        db.commit()
    return org_id


def make_key(
    org_id: str = DEFAULT_ORGANIZATION_ID,
    role: str = "admin",
    *,
    revoked: bool = False,
    expires_in: timedelta | None = None,
    last_used_at: datetime | None = None,
) -> tuple[str, str]:
    plaintext, prefix, digest = auth.generate_api_key()
    key_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        db.add(
            ApiKey(
                id=key_id,
                organization_id=org_id,
                name="test",
                key_prefix=prefix,
                key_hash=digest,
                role=role,
                revoked_at=now if revoked else None,
                expires_at=now + expires_in if expires_in is not None else None,
                last_used_at=last_used_at,
            )
        )
        db.commit()
    return plaintext, key_id


def last_used(key_id: str) -> datetime | None:
    with SessionLocal() as db:
        return db.get(ApiKey, key_id).last_used_at


# -- helpers -----------------------------------------------------------------------------


def test_generate_api_key_format() -> None:
    plaintext, prefix, digest = auth.generate_api_key()
    assert plaintext.startswith("vlk_") and len(plaintext) > 40
    assert plaintext.startswith(prefix)
    assert digest == auth.hash_token(plaintext)
    assert auth.generate_api_key()[0] != plaintext


def test_role_ordering() -> None:
    assert role_at_least("owner", "admin") and role_at_least("admin", "admin")
    assert role_at_least("admin", "member") and not role_at_least("member", "admin")
    assert not role_at_least(None, "member") and not role_at_least("bogus", "member")
    with pytest.raises(ValueError):
        role_at_least("owner", "root")
    with pytest.raises(ValueError):
        require("root")


# -- superadmin --------------------------------------------------------------------------


def test_superadmin_defaults_to_default_org(client) -> None:
    body = client.get(WHOAMI, headers=ADMIN).json()
    assert body["principal"] == "superadmin"
    assert body["org_id"] == DEFAULT_ORGANIZATION_ID


def test_superadmin_can_switch_org_by_header(client) -> None:
    org_id = make_org()
    body = client.get(WHOAMI, headers={**ADMIN, "X-Organization-Id": org_id}).json()
    assert body["org_id"] == org_id


def test_superadmin_unknown_org_is_404(client) -> None:
    response = client.get(WHOAMI, headers={**ADMIN, "X-Organization-Id": str(uuid4())})
    assert response.status_code == 404
    assert response.json() == {"detail": {"code": "organization_not_found"}}


def test_request_id_echoed_or_generated(client) -> None:
    assert (
        client.get(WHOAMI, headers={**ADMIN, "X-Request-Id": "req-123"}).json()["request_id"]
        == "req-123"
    )
    assert len(client.get(WHOAMI, headers=ADMIN).json()["request_id"]) == 36


def test_runtime_route_gets_runtime_ctx_without_org(client, monkeypatch) -> None:
    monkeypatch.setattr(
        auth, "RUNTIME_TOKEN_ROUTES", auth.RUNTIME_TOKEN_ROUTES | {("GET", WHOAMI)}
    )
    body = client.get(WHOAMI).json()  # no bearer needed on runtime-token routes
    assert (body["principal"], body["org_id"], body["role"]) == ("runtime", None, None)


def test_public_route_has_no_ctx(client, monkeypatch) -> None:
    monkeypatch.setattr(auth, "PUBLIC_ROUTES", auth.PUBLIC_ROUTES | {("GET", WHOAMI)})
    response = client.get(WHOAMI)
    assert response.status_code == 401  # get_ctx refuses: public routes carry no context


# -- vlk_ API keys -----------------------------------------------------------------------


def test_api_key_valid(client) -> None:
    org_id = make_org()
    key, key_id = make_key(org_id, "member")
    body = client.get(WHOAMI, headers=bearer(key)).json()
    assert body["principal"] == "api_key"
    assert body["org_id"] == org_id
    assert body["role"] == "member"
    assert body["key_id"] == key_id


def test_api_key_matching_header_ok_and_mismatch_403(client) -> None:
    org_id = make_org()
    key, _ = make_key(org_id)
    ok = client.get(WHOAMI, headers={**bearer(key), "X-Organization-Id": org_id})
    assert ok.status_code == 200
    bad = client.get(WHOAMI, headers={**bearer(key), "X-Organization-Id": make_org()})
    assert bad.status_code == 403
    assert bad.json() == {"detail": {"code": "organization_mismatch"}}


def _generic_401(response) -> None:
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid_api_key"}
    assert response.headers["www-authenticate"] == "Bearer"


def test_api_key_rejections_are_indistinguishable(client) -> None:
    revoked, _ = make_key(revoked=True)
    expired, _ = make_key(expires_in=timedelta(seconds=-5))
    suspended, _ = make_key(make_org(status="suspended"))
    unknown = auth.generate_api_key()[0]
    for token in (revoked, expired, suspended, unknown):
        _generic_401(client.get(WHOAMI, headers=bearer(token)))


def test_unexpired_key_works(client) -> None:
    key, _ = make_key(expires_in=timedelta(days=1))
    assert client.get(WHOAMI, headers=bearer(key)).status_code == 200


def test_last_used_is_throttled_to_once_a_minute(client) -> None:
    key, key_id = make_key()
    assert last_used(key_id) is None
    client.get(WHOAMI, headers=bearer(key))
    first = last_used(key_id)
    assert first is not None
    client.get(WHOAMI, headers=bearer(key))
    assert last_used(key_id) == first  # within a minute: untouched

    stale = datetime.now(timezone.utc) - timedelta(minutes=5)
    with SessionLocal() as db:
        db.get(ApiKey, key_id).last_used_at = stale
        db.commit()
    client.get(WHOAMI, headers=bearer(key))
    refreshed = last_used(key_id)
    assert refreshed is not None
    assert refreshed.replace(tzinfo=timezone.utc) > stale + timedelta(minutes=4)


# -- dev mode ----------------------------------------------------------------------------


def test_dev_mode_context(dev_client) -> None:
    body = dev_client.get(WHOAMI).json()
    assert body["principal"] == "dev"
    assert body["org_id"] == DEFAULT_ORGANIZATION_ID
    assert body["role"] == "owner"


def test_dev_mode_honours_valid_vlk_key(dev_client) -> None:
    org_id = make_org()
    key, _ = make_key(org_id, "member")
    body = dev_client.get(WHOAMI, headers=bearer(key)).json()
    assert (body["principal"], body["org_id"], body["role"]) == ("api_key", org_id, "member")


def test_dev_mode_rejects_invalid_vlk_key(dev_client) -> None:
    revoked, _ = make_key(revoked=True)
    _generic_401(dev_client.get(WHOAMI, headers=bearer(revoked)))
    _generic_401(dev_client.get(WHOAMI, headers=bearer(auth.generate_api_key()[0])))


def test_dev_mode_ignores_other_bearer_tokens(dev_client) -> None:
    assert dev_client.get(WHOAMI, headers=bearer("whatever")).json()["principal"] == "dev"


# -- require() ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "admin_status", "owner_status"),
    [("member", 403, 403), ("admin", 200, 403), ("owner", 200, 200)],
)
def test_require_by_role(client, role, admin_status, owner_status) -> None:
    key, _ = make_key(role=role)
    headers = bearer(key)
    admin_response = client.get("/__test_ctx/admin-only", headers=headers)
    assert admin_response.status_code == admin_status
    if admin_status == 403:
        assert admin_response.json() == {"detail": {"code": "insufficient_role"}}
    assert client.get("/__test_ctx/owner-only", headers=headers).status_code == owner_status


def test_require_passes_superadmin_and_dev(client, dev_client) -> None:
    assert client.get("/__test_ctx/owner-only", headers=ADMIN).status_code == 200
    assert dev_client.get("/__test_ctx/owner-only").status_code == 200


@pytest.mark.parametrize("principal", ["session", "runtime"])
def test_require_never_passes_session_or_runtime(principal) -> None:
    ctx = AuthContext(org_id=DEFAULT_ORGANIZATION_ID, principal=principal, role="owner")

    class _Req:
        class state:  # noqa: N801 - mimics request.state
            pass

    request = _Req()
    request.state.ctx = ctx
    with pytest.raises(Exception) as excinfo:
        require("member")(request)  # type: ignore[arg-type]
    assert excinfo.value.status_code == 403


# -- session tokens ----------------------------------------------------------------------


def test_session_token_ctx_carries_session_org(client) -> None:
    org_id = make_org()
    token, digest = auth.new_session_token()
    guest, guest_digest = auth.new_session_token()
    session_id = str(uuid4())
    with SessionLocal() as db:
        db.add(
            VoiceSession(
                id=session_id,
                conversation_id=str(uuid4()),
                room_name=f"room-{session_id}",
                session_token_hash=digest,
                guest_token_hashes=[guest_digest],
                organization_id=org_id,
            )
        )
        db.commit()

    path = f"/__test_ctx/sessions/{session_id}"
    body = client.get(path, headers=bearer(token)).json()
    assert body == {"org_id": org_id, "principal": "session", "is_guest": False}
    assert client.get(path, headers=bearer(guest)).json()["is_guest"] is True
    # A session token cannot reach non-scoped routes.
    assert client.get(WHOAMI, headers=bearer(token)).status_code == 403


def test_session_token_rejected_when_org_suspended(client) -> None:
    org_id = make_org()
    token, digest = auth.new_session_token()
    guest, guest_digest = auth.new_session_token()
    session_id = str(uuid4())
    with SessionLocal() as db:
        db.add(
            VoiceSession(
                id=session_id,
                conversation_id=str(uuid4()),
                room_name=f"room-{session_id}",
                session_token_hash=digest,
                guest_token_hashes=[guest_digest],
                organization_id=org_id,
            )
        )
        db.commit()
    path = f"/__test_ctx/sessions/{session_id}"
    assert client.get(path, headers=bearer(token)).status_code == 200

    with SessionLocal() as db:
        db.get(Organization, org_id).status = "suspended"
        db.commit()
    for tok in (token, guest):
        response = client.get(path, headers=bearer(tok))
        assert response.status_code == 401
        assert response.json() == {"detail": "invalid_api_key"}

    with SessionLocal() as db:
        db.get(Organization, org_id).status = "active"
        db.commit()
    assert client.get(path, headers=bearer(token)).status_code == 200

"""Route-table-driven cross-organisation isolation harness (organisations plan, T8).

The module-level tests in ``test_tenancy_<module>.py`` prove each router in isolation. This file
is the net underneath them: it enumerates *every* route the API exposes (from the OpenAPI schema,
which also includes routers that ``app.routes`` hides) and fails when a route is neither covered
here, covered by a module's ``ISOLATION_CASES``, nor deliberately listed as public/global.

World: two organisations, each with one of everything (see ``tenancy_fixtures``). Attacker is
always org B; victim is always org A.

a. completeness (route table == ROUTE_CASES | module ISOLATION_CASES | PUBLIC | GLOBAL)
b. every route that takes a tenant id: B's keys + A's id -> 404, indistinguishable from a
   non-existent id, and A's data is unchanged afterwards
c. every collection route: B's keys never see any of A's identifiers
d. body references: B referencing A's ids when creating/patching -> rejected (with a positive
   control proving the same request with B's own ids succeeds); mass-assignment of organization_id
e. superadmin + X-Organization-Id: sees exactly the selected organisation
f. audit log of B never mentions A, and B's attacks do not leak into A's log
"""

import json
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.routing import APIRoute

import test_tenancy_agents
import test_tenancy_conversations
import test_tenancy_knowledge_tools
import test_tenancy_sessions
import test_tenancy_voices
from tenancy_fixtures import (  # noqa: E402  (pytest puts tests/ on sys.path)
    RUNTIME,
    SUPER,
    build_org,
    make_client,
    phone,
)

from src.platform import auth
from src.platform.main import app

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}
MISSING = "00000000-0000-4000-8000-0000000000ee"

# --------------------------------------------------------------------- route tables

# The shareable-link call routes (calls.py) take no tenant id: they run as the system principal of the
# configured calls organisation and find a session only by an unguessable invitation code
# (tests/test_calls.py covers them, including that an organisation never sees another's call).
PUBLIC = {("GET", "/health")} | set(auth.PUBLIC_CALL_ROUTES)

# Demos/handoffs use the fixed calls organisation, ignoring X-Organization-Id.
# Their public/session/staff credential and cross-org checks are in test_demo_handoffs.py.
FIXED_ORG_ROUTES = set(
    auth.PUBLIC_DEMO_ROUTES | auth.PUBLIC_HANDOFF_ROUTES | auth.STAFF_HANDOFF_ROUTES
)

# Routes with no tenant data. Each is justified; org-management routes are additionally
# asserted superadmin-only in test_superadmin_only_routes.
GLOBAL = {
    # Invitation-code routes (calls.py): the code, not a tenant id, selects the call, and it is looked up
    # only inside the configured calls organisation. Also listed in PUBLIC.
    ("GET", "/calls/invites/{code}"),
    ("POST", "/calls/invites/{code}/join"),
    # Static catalogue shared by every org (agent templates are cloned into the caller's org
    # by POST /agent-templates/{key}/agents, which is covered by the agents module cases).
    ("GET", "/agent-templates"),
    ("GET", "/agent-templates/{key}"),
    # Provider catalogue/health: no tenant rows.
    ("GET", "/health/providers"),
    # Superadmin-only organisation management.
    ("GET", "/organizations"),
    ("POST", "/organizations"),
}

# (method, path) covered by *this* file.
PARAM_CASES = [
    # agents
    ("GET", "/agents/{agent_id}"),
    ("PATCH", "/agents/{agent_id}"),
    ("DELETE", "/agents/{agent_id}"),
    ("PUT", "/agents/{agent_id}/knowledge"),
    ("GET", "/agents/{agent_id}/knowledge"),
    ("PUT", "/agents/{agent_id}/tools"),
    ("GET", "/agents/{agent_id}/tools"),
    # voices
    ("GET", "/voices/{voice_id}"),
    ("PATCH", "/voices/{voice_id}"),
    ("DELETE", "/voices/{voice_id}"),
    ("POST", "/voices/{voice_id}/preview"),
    ("POST", "/voices/{voice_id}/provision"),
    ("POST", "/voices/{voice_id}/status"),
    # knowledge + tools
    ("GET", "/knowledge/documents/{document_id}"),
    ("PATCH", "/knowledge/documents/{document_id}"),
    ("DELETE", "/knowledge/documents/{document_id}"),
    ("GET", "/tools/{tool_id}"),
    ("PATCH", "/tools/{tool_id}"),
    ("DELETE", "/tools/{tool_id}"),
    # deployments + campaigns
    ("GET", "/deployments/{deployment_id}"),
    ("PATCH", "/deployments/{deployment_id}"),
    ("DELETE", "/deployments/{deployment_id}"),
    ("POST", "/deployments/{deployment_id}/bridge-call"),
    ("GET", "/campaigns/{campaign_id}"),
    ("PATCH", "/campaigns/{campaign_id}"),
    ("POST", "/campaigns/{campaign_id}/status"),
    ("POST", "/campaigns/{campaign_id}/dial-next"),
    ("POST", "/campaigns/{campaign_id}/targets"),
    ("GET", "/campaigns/{campaign_id}/targets"),
    # sessions
    ("GET", "/voice/sessions/{session_id}"),
    ("POST", "/voice/sessions/{session_id}/end"),
    ("POST", "/voice/sessions/{session_id}/ensure-running"),
    ("POST", "/voice/sessions/{session_id}/participants"),
    ("POST", "/voice/sessions/{session_id}/phone-participants"),
    # translations (two-person interpreter sessions)
    ("GET", "/translations/{session_id}"),
    ("POST", "/translations/{session_id}/invite"),
    ("POST", "/calls/{session_id}/invite"),
    ("POST", "/calls/{session_id}/end"),
    # conversations
    ("GET", "/conversations/{conversation_id}"),
    ("GET", "/conversations/{conversation_id}/messages"),
    ("GET", "/conversations/{conversation_id}/runtime-state"),
    ("PUT", "/conversations/{conversation_id}/outcome"),
    ("GET", "/conversations/{conversation_id}/debug"),
    ("GET", "/conversations/{conversation_id}/reports"),
    ("POST", "/conversations/{conversation_id}/reports"),
    # org / key management
    ("DELETE", "/api-keys/{key_id}"),
    ("POST", "/organizations/{organization_id}/status"),
]

COLLECTION_CASES = [
    ("GET", "/agents"),
    ("GET", "/providers"),
    ("GET", "/voices"),
    ("GET", "/knowledge/documents"),
    ("GET", "/tools"),
    ("GET", "/deployments"),
    ("GET", "/campaigns"),
    ("GET", "/conversations"),
    ("GET", "/translations"),
    ("GET", "/analytics/summary"),
    ("GET", "/audit-log"),
    ("GET", "/api-keys"),
    ("GET", "/organization"),
]

BODY_CASES = [
    ("POST", "/agents"),
    ("POST", "/providers"),
    ("POST", "/voices"),
    ("POST", "/voices/clone"),
    ("POST", "/voices/design"),
    ("POST", "/knowledge/documents"),
    ("POST", "/tools"),
    ("POST", "/deployments"),
    ("POST", "/campaigns"),
    ("POST", "/voice/sessions"),
    ("POST", "/translations"),
    ("POST", "/api-keys"),
    ("PATCH", "/organization"),
    ("POST", "/agent-templates/{key}/agents"),
]

# Runtime-token routes: the caller is Runtime, not a tenant; the org comes from the session /
# agent. Covered by test_runtime_token_routes.
RUNTIME_CASES = [
    ("POST", "/runtime/events"),
    ("POST", "/runtime/agents/{agent_id}/knowledge/search"),
]

ROUTE_CASES = set(PARAM_CASES + COLLECTION_CASES + BODY_CASES + RUNTIME_CASES)

MODULE_CASE_LISTS = {
    "agents": test_tenancy_agents.ISOLATION_CASES,
    "voices": test_tenancy_voices.ISOLATION_CASES,
    "knowledge_tools": test_tenancy_knowledge_tools.ISOLATION_CASES,
    "conversations": test_tenancy_conversations.ISOLATION_CASES,
    "sessions": test_tenancy_sessions.ISOLATION_CASES,
}
MODULE_CASES = {tuple(case) for cases in MODULE_CASE_LISTS.values() for case in cases}

# Routes that may legitimately be missing from the OpenAPI document while another ticket's
# router is not merged yet (conversation debug/report routes, ST-02). Once present they are
# held to the same standard as everything else: the completeness test demands coverage.
OPTIONAL_ROUTES: set[tuple[str, str]] = set()

# Path parameter name -> key in OrgWorld.ids.
TENANT_PARAMS = {
    "agent_id": "agent",
    "conversation_id": "conversation",
    "session_id": "session",
    "voice_id": "voice_clone",
    "document_id": "document",
    "tool_id": "tool",
    "deployment_id": "deployment",
    "campaign_id": "campaign",
    "key_id": "admin_key",
    "organization_id": "org",
}


def api_routes() -> set[tuple[str, str]]:
    return {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in HTTP_METHODS
    }


def path_params(path: str) -> list[str]:
    return [part[1:-1] for part in path.split("/") if part.startswith("{")]


# ----------------------------------------------------------------------- completeness


def test_every_api_route_is_covered():
    routes = api_routes()
    covered = ROUTE_CASES | MODULE_CASES | PUBLIC | GLOBAL | FIXED_ORG_ROUTES
    missing = sorted(routes - covered)
    assert not missing, (
        "Routes with no cross-org isolation coverage. Add them to ROUTE_CASES (and a test), "
        "to a test_tenancy_<module>.ISOLATION_CASES, or justify them in GLOBAL/PUBLIC:\n  "
        + "\n  ".join(f"{m} {p}" for m, p in missing)
    )


def test_coverage_lists_name_real_routes():
    routes = api_routes()
    stale = sorted((ROUTE_CASES | MODULE_CASES | PUBLIC | GLOBAL | FIXED_ORG_ROUTES) - routes - OPTIONAL_ROUTES)
    assert not stale, f"cases for routes that no longer exist: {stale}"


def test_no_route_is_hidden_from_the_schema():
    hidden = []

    def walk(routes, prefix=""):
        for route in routes:
            if isinstance(route, APIRoute) and not route.include_in_schema:
                hidden.append((sorted(route.methods), prefix + route.path))
            elif hasattr(route, "routes"):
                walk(route.routes, prefix + getattr(route, "path", ""))

    walk(app.routes)
    assert not hidden, f"routes excluded from OpenAPI escape the completeness check: {hidden}"


def test_tenant_id_routes_are_driven_by_the_id_matrix():
    """Every route with a tenant path id gets the B-uses-A's-id test, not just a module case."""

    param_routes = {
        route
        for route in api_routes()
        if any(name in TENANT_PARAMS for name in path_params(route[1]))
    } - set(RUNTIME_CASES)
    assert param_routes <= set(PARAM_CASES), sorted(param_routes - set(PARAM_CASES))
    unknown = sorted(
        route
        for route in api_routes()
        if any(name not in TENANT_PARAMS for name in path_params(route[1]))
        and route not in GLOBAL
        and route not in FIXED_ORG_ROUTES
        and route not in RUNTIME_CASES
        and route not in BODY_CASES
    )
    assert not unknown, f"unclassified path parameters: {unknown}"


def test_collection_routes_are_driven_by_the_collection_matrix():
    collections = {
        (method, path)
        for method, path in api_routes()
        if method == "GET" and not path_params(path)
    } - PUBLIC - GLOBAL - FIXED_ORG_ROUTES
    assert collections <= set(COLLECTION_CASES), sorted(collections - set(COLLECTION_CASES))


def test_runtime_token_routes_match_auth_table():
    assert set(RUNTIME_CASES) == set(auth.RUNTIME_TOKEN_ROUTES)


# ------------------------------------------------------------------------------ world


@pytest.fixture
def client(monkeypatch):
    with make_client(monkeypatch) as test_client:
        yield test_client


@pytest.fixture
def worlds(client):
    return build_org(client, "a"), build_org(client, "b")


def ident(world, param: str) -> str:
    kind = TENANT_PARAMS[param]
    return world.org_id if kind == "org" else world.ids[kind]


def strings(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from strings(nested)
    elif value is not None:
        yield str(value)


def leaks(response, world) -> list[str]:
    """Markers of ``world`` found anywhere (keys or values, recursively) in the response."""

    try:
        found = list(strings(response.json()))
    except ValueError:
        found = [response.text]
    return sorted(
        marker
        for marker in world.leak_markers
        if any(marker in text for text in found)
    )


def scrub(response, *identifiers: str):
    text = response.text
    for identifier in identifiers:
        text = text.replace(identifier, "<id>")
    return response.status_code, text


READ_PATHS = [
    "/agents/{agent}",
    "/voices/{voice_clone}",
    "/voices/{voice_library}",
    "/knowledge/documents/{document}",
    "/tools/{tool}",
    "/deployments/{deployment}",
    "/campaigns/{campaign}",
    "/conversations/{conversation}",
    "/voice/sessions/{session}",
    "/agents/{agent}/knowledge",
    "/agents/{agent}/tools",
    "/campaigns/{campaign}/targets",
    "/conversations/{conversation}/messages",
    "/conversations/{conversation}/debug",
    "/conversations/{conversation}/reports",
]


def snapshot(client, world) -> dict:
    """Stable view of everything in ``world`` as its owner sees it."""

    view = {}
    for template in READ_PATHS:
        response = client.get(template.format(**world.ids), headers=world.owner)
        body = response.json()
        if isinstance(body, dict):
            body = {
                key: body.get(key)
                for key in ("id", "name", "title", "status", "enabled", "deleted_at", "outcome")
            }
        view[template] = (response.status_code, json.dumps(body, sort_keys=True))
    keys = client.get("/api-keys", headers=world.owner).json()["items"]
    view["keys"] = sorted((k["id"], k["revoked_at"] is None) for k in keys)
    org = client.get("/organization", headers=world.owner).json()
    view["org"] = (org["id"], org["name"], org["status"])
    return view


# ---------------------------------------------------------------- b. id matrix

# Minimal valid bodies, so a 404 can only come from the lookup, never from validation.
PARAM_BODIES = {
    ("PATCH", "/agents/{agent_id}"): lambda: {"name": "hijacked"},
    ("PUT", "/agents/{agent_id}/knowledge"): lambda: {"document_ids": []},
    ("PUT", "/agents/{agent_id}/tools"): lambda: {"tool_ids": []},
    ("PATCH", "/voices/{voice_id}"): lambda: {"name": "hijacked"},
    ("POST", "/voices/{voice_id}/preview"): lambda: {"text": "hello"},
    ("POST", "/voices/{voice_id}/status"): lambda: {"status": "disabled"},
    ("PATCH", "/knowledge/documents/{document_id}"): lambda: {"title": "hijacked"},
    ("PATCH", "/tools/{tool_id}"): lambda: {"description": "hijacked"},
    ("PATCH", "/deployments/{deployment_id}"): lambda: {"name": "hijacked"},
    ("POST", "/deployments/{deployment_id}/bridge-call"): lambda: {
        "from_number": phone(),
        "to_number": phone(),
    },
    ("PATCH", "/campaigns/{campaign_id}"): lambda: {"name": "hijacked"},
    ("POST", "/campaigns/{campaign_id}/status"): lambda: {"status": "cancelled"},
    ("POST", "/campaigns/{campaign_id}/targets"): lambda: {
        "targets": [{"phone_number": phone()}]
    },
    ("POST", "/voice/sessions/{session_id}/participants"): lambda: {"language": "en"},
    ("POST", "/translations/{session_id}/invite"): lambda: {"language": "en"},
    ("POST", "/voice/sessions/{session_id}/phone-participants"): lambda: {
        "language": "en",
        "phone_number": phone(),
    },
    ("PUT", "/conversations/{conversation_id}/outcome"): lambda: {"outcome": "resolved"},
    ("POST", "/conversations/{conversation_id}/reports"): lambda: {"note": "planted"},
    ("POST", "/organizations/{organization_id}/status"): lambda: {"status": "suspended"},
}

# Expected status for org B's admin key; everything else must be 404 (resource not found).
# Only routes whose auth check precedes any lookup appear here.
EXPECTED_STATUS = {
    # Superadmin-only (require_superadmin runs before the lookup): an org admin gets 403 no
    # matter whose id it is, which leaks nothing. The superadmin+X-Organization-Id path is held
    # to 404 in test_superadmin_header_selects_one_org.
    ("POST", "/voices/{voice_id}/status"): 403,
    ("POST", "/organizations/{organization_id}/status"): 403,
}


@pytest.mark.parametrize("method,path", PARAM_CASES, ids=lambda v: v if isinstance(v, str) else None)
def test_foreign_id_is_404_and_indistinguishable_from_missing(client, worlds, method, path):
    a, b = worlds
    (param,) = path_params(path)
    make_body = PARAM_BODIES.get((method, path))
    expected = EXPECTED_STATUS.get((method, path), 404)
    before = snapshot(client, a)

    def call(identifier: str, headers):
        return client.request(
            method,
            path.format(**{param: identifier}),
            json=make_body() if make_body else None,
            headers=headers,
        )

    foreign = call(ident(a, param), b.admin)
    assert foreign.status_code == expected, (foreign.status_code, foreign.text)
    assert not leaks(foreign, a)
    missing = call(MISSING, b.admin)
    assert scrub(foreign, ident(a, param)) == scrub(missing, MISSING), (
        "response differs between another org's id and a non-existent id (existence oracle)"
    )
    # A lower role never gets further than the admin did.
    member = call(ident(a, param), b.member)
    assert member.status_code in (403, 404), (member.status_code, member.text)
    assert not leaks(member, a)
    # Owner of B is no different from admin of B.
    owner = call(ident(a, param), b.owner)
    assert owner.status_code == expected, (owner.status_code, owner.text)

    assert snapshot(client, a) == before, "org A's data changed after org B's attempts"


# ------------------------------------------------------------ c. collection matrix

COLLECTION_QUERIES = {
    "/agents": ["", "?limit=1000"],
    "/providers": ["", "?kind=llm", "?kind=tts"],
    "/voices": [
        "",
        "?include_deleted=true",
        "?status=ready",
        "?source=clone",
        "?source=library",
        "?source=design",
    ],
    "/knowledge/documents": ["", "?enabled=true", "?enabled=false"],
    "/tools": ["", "?enabled=true"],
    "/deployments": ["", "?agent_id={A.agent}", "?channel=sip", "?channel=web"],
    "/campaigns": ["", "?agent_id={A.agent}", "?status=draft"],
    "/conversations": [
        "",
        "?agent_id={A.agent}",
        "?channel=web",
        "?status=active",
    ],
    "/translations": ["", "?limit=200"],
    "/analytics/summary": ["", "?agent_id={A.agent}"],
    "/audit-log": [
        "",
        "?limit=200",
        "?resource_id={A.agent}",
        "?resource_id={A.voice_clone}",
        "?resource_type=voice",
        "?action=voice.created",
    ],
    "/api-keys": [""],
    "/organization": [""],
}


def expand(query: str, a) -> str:
    return query.format(A=type("A", (), a.ids)) if "{A." in query else query


@pytest.mark.parametrize("method,path", COLLECTION_CASES, ids=lambda v: v if isinstance(v, str) else None)
def test_collections_never_contain_foreign_ids(client, worlds, method, path):
    a, b = worlds
    if path == "/translations":
        # The shared world holds ordinary sessions only: give each org a translation session
        # so the leak scan has something of its own to find (and something to not leak).
        for world in (a, b):
            created = client.post(
                "/translations",
                json={"languages": ["en", "ko"], "creator": {"language": "en"}},
                headers=world.admin,
            )
            assert created.status_code == 201, created.text
            world.ids["translation"] = created.json()["session_id"]
            world.all_ids.update(
                {created.json()["session_id"], created.json()["conversation_id"]}
            )
    own_hits = 0
    for query in COLLECTION_QUERIES[path]:
        url = path + expand(query, a)
        for label, headers in (("owner", b.owner), ("admin", b.admin), ("member", b.member)):
            response = client.get(url, headers=headers)
            assert response.status_code in (200, 403), (url, label, response.text)
            found = leaks(response, a)
            assert not found, f"{label} of B saw org A data via GET {url}: {found}"
            if response.status_code == 200 and not query:
                own_hits += len(leaks(response, b))
        # Control: A sees its own data on the same query (so an empty scan is meaningful).
        control = client.get(url, headers=a.owner)
        assert control.status_code == 200, (url, control.text)
    assert own_hits > 0 or path == "/analytics/summary", (
        f"org B saw none of its own data via GET {path}; the leak scan proves nothing"
    )


def test_analytics_counts_only_own_conversations(client, worlds):
    a, b = worlds
    own = client.get("/analytics/summary", headers=b.member).json()
    assert own["conversations"]["total"] == 1
    assert own["latency"]["llm_ttfb"]["count"] == 1
    filtered = client.get(
        f"/analytics/summary?agent_id={a.ids['agent']}", headers=b.member
    ).json()
    assert filtered["conversations"]["total"] == 0
    assert filtered["latency"] == {}


def test_sub_collections_of_own_resources_hold_no_foreign_ids(client, worlds):
    a, b = worlds
    for template in READ_PATHS + [
        "/conversations/{conversation}/runtime-state",
        "/campaigns/{campaign}/targets?limit=500",
    ]:
        for headers in (b.admin, b.member):
            response = client.get(template.format(**b.ids), headers=headers)
            assert response.status_code == 200, (template, response.text)
            assert not leaks(response, a), template


def test_superadmin_only_routes(client, worlds):
    a, b = worlds
    for headers in (b.owner, b.admin, b.member):
        assert client.get("/organizations", headers=headers).status_code == 403
        created = client.post(
            "/organizations",
            json={"name": "x", "slug": f"nope-{uuid4().hex[:8]}"},
            headers=headers,
        )
        assert created.status_code == 403
    listing = client.get("/organizations", headers=SUPER).json()["items"]
    assert {a.org_id, b.org_id} <= {org["id"] for org in listing}
    # Provider health is global but must not echo tenant configuration.
    health = client.get("/health/providers", headers=b.member)
    assert health.status_code == 200 and not leaks(health, a)
    # Templates are the same for everyone and tenant-free.
    for headers in (a.member, b.member):
        for other in (a, b):
            assert not leaks(client.get("/agent-templates", headers=headers), other)


# ------------------------------------------------------------------- d. body references


def n() -> str:
    return f"n-{uuid4().hex[:10]}"


def agent_body(**refs):
    return lambda o, r: {"name": n(), **{k: f(r) for k, f in refs.items()}}


def vid(world):
    return world.extra["clone_provider_voice_id"]


# (id, method, path(own), body(own, ref), headers, expected status of the attack,
#  status of the control request made with ref=own, or None when no control is possible)
REFERENCE_CASES = [
    ("agent-llm", "POST", lambda o: "/agents", agent_body(llm_provider_id=lambda r: r["llm_provider"]), None, 422, 201),
    ("agent-stt", "POST", lambda o: "/agents", agent_body(stt_provider_id=lambda r: r["stt_provider"]), None, 422, 201),
    ("agent-tts", "POST", lambda o: "/agents", agent_body(tts_provider_id=lambda r: r["tts_provider"]), None, 422, 201),
    ("agent-voice-profile", "POST", lambda o: "/agents", agent_body(voice_profile_id=lambda r: r["voice_clone"]), None, 422, 201),
    ("agent-voice-id", "POST", lambda o: "/agents", agent_body(voice_id=vid), None, 409, 201),
    ("agent-patch-llm", "PATCH", lambda o: f"/agents/{o['agent']}", lambda o, r: {"llm_provider_id": r["llm_provider"]}, None, 422, 200),
    ("agent-patch-tts", "PATCH", lambda o: f"/agents/{o['agent']}", lambda o, r: {"tts_provider_id": r["tts_provider"]}, None, 422, 200),
    ("agent-patch-voice-profile", "PATCH", lambda o: f"/agents/{o['agent']}", lambda o, r: {"voice_profile_id": r["voice_clone"]}, None, 422, 200),
    ("agent-patch-voice-id", "PATCH", lambda o: f"/agents/{o['agent']}", lambda o, r: {"voice_id": vid(r)}, None, 409, 200),
    ("agent-link-document", "PUT", lambda o: f"/agents/{o['agent']}/knowledge", lambda o, r: {"document_ids": [r["document"]]}, None, 422, 200),
    ("agent-link-tool", "PUT", lambda o: f"/agents/{o['agent']}/tools", lambda o, r: {"tool_ids": [r["tool"]]}, None, 422, 200),
    ("voice-library-provider", "POST", lambda o: "/voices", lambda o, r: {"name": n(), "tts_provider_id": r["tts_provider"], "provider_voice_id": n()}, None, 404, 201),
    ("voice-clone-provider", "POST", lambda o: "/voices/clone", lambda o, r: {"name": n(), "tts_provider_id": r["tts_provider"], "sample_reference": f"https://example.com/{n()}.wav", "consent": {"confirmed": True, "subject_name": "Ada", "statement": "I consent.", "method": "self"}}, None, 404, 201),
    ("voice-design-provider", "POST", lambda o: "/voices/design", lambda o, r: {"name": n(), "tts_provider_id": r["tts_provider"], "design_prompt": "calm"}, None, 404, 201),
    ("deployment-agent", "POST", lambda o: "/deployments", lambda o, r: {"agent_id": r["agent"], "name": n(), "channel": "web"}, None, 422, 201),
    ("deployment-phone-taken", "POST", lambda o: "/deployments", lambda o, r: {"agent_id": o["agent"], "name": n(), "channel": "sip", "phone_number": r.extra["sip_phone_number"]}, None, 409, None),
    ("deployment-patch-phone-taken", "PATCH", lambda o: f"/deployments/{o['deployment']}", lambda o, r: {"phone_number": r.extra["sip_phone_number"]}, None, 409, None),
    ("campaign-agent", "POST", lambda o: "/campaigns", lambda o, r: {"name": n(), "agent_id": r["agent"], "deployment_id": o["deployment"]}, None, 422, 201),
    ("campaign-deployment", "POST", lambda o: "/campaigns", lambda o, r: {"name": n(), "agent_id": o["agent"], "deployment_id": r["deployment"]}, None, 422, 201),
    ("campaign-both", "POST", lambda o: "/campaigns", lambda o, r: {"name": n(), "agent_id": r["agent"], "deployment_id": r["deployment"]}, None, 422, 201),
    ("session-agent", "POST", lambda o: "/voice/sessions", lambda o, r: {"agent_id": r["agent"]}, None, 404, 201),
    ("session-deployment", "POST", lambda o: "/voice/sessions", lambda o, r: {"agent_id": o["agent"], "deployment_id": r["deployment_web"]}, None, 404, 201),
    ("session-participant-voice", "POST", lambda o: "/voice/sessions", lambda o, r: {"agent_id": o["interpreter_agent"], "participant": {"language": "en", "voice_profile_id": r["voice_clone"]}}, None, 409, 201),
    ("translation-creator-voice", "POST", lambda o: "/translations", lambda o, r: {"languages": ["en", "ko"], "creator": {"language": "en", "voice_profile_id": r["voice_clone"]}}, None, 409, 201),
    ("translation-tts-provider", "POST", lambda o: "/translations", lambda o, r: {"languages": ["en", "ko"], "creator": {"language": "en"}, "tts_provider_id": r["tts_provider"]}, None, 422, 201),
    ("runtime-event-foreign-conversation", "POST", lambda o: "/runtime/events", lambda o, r: {"event_id": str(uuid4()), "event_type": "TranscriptFinal", "session_id": o["session"], "conversation_id": r["conversation"], "agent_id": o["agent"], "payload": {"role": "user", "content": "x"}}, RUNTIME, 409, 202),
    ("runtime-event-foreign-agent", "POST", lambda o: "/runtime/events", lambda o, r: {"event_id": str(uuid4()), "event_type": "TranscriptFinal", "session_id": o["session"], "conversation_id": o["conversation"], "agent_id": r["agent"], "payload": {"role": "user", "content": "x"}}, RUNTIME, 409, 202),
    ("runtime-event-foreign-session", "POST", lambda o: "/runtime/events", lambda o, r: {"event_id": str(uuid4()), "event_type": "TranscriptFinal", "session_id": r["session"], "conversation_id": o["conversation"], "agent_id": o["agent"], "payload": {"role": "user", "content": "x"}}, RUNTIME, 409, 202),
    ("key-org-in-body", "POST", lambda o: "/api-keys", lambda o, r: {"name": n(), "role": "member", **({"organization_id": r.org_id} if r is not o else {})}, None, 422, 201),
    ("organization-patch-id", "PATCH", lambda o: "/organization", lambda o, r: {"name": "renamed", **({"organization_id": r.org_id} if r is not o else {})}, "owner", 422, 200),
]


# Body references that must not be an existence oracle: a foreign id and a random uuid
# get the same status and detail (422 invalid_*_id).
EXISTENCE_ORACLE_CASES = {
    "agent-llm", "agent-stt", "agent-tts", "agent-voice-profile", "agent-patch-llm",
    "agent-patch-tts", "agent-patch-voice-profile", "agent-link-document",
    "agent-link-tool", "deployment-agent", "campaign-agent", "campaign-deployment",
    "campaign-both", "translation-tts-provider",
}


class MissingRefs:
    """Stands in for a world's ``r``: every reference is a fresh random uuid."""

    def __getitem__(self, key):
        return str(uuid4())


@pytest.mark.parametrize(
    "case", REFERENCE_CASES, ids=[case[0] for case in REFERENCE_CASES]
)
def test_body_references_to_foreign_ids_are_rejected(client, worlds, case):
    name, method, path_of, body_of, headers, expected, control = case
    a, b = worlds
    headers = b.owner if headers == "owner" else headers or b.admin
    path = path_of(b)
    before = snapshot(client, a)

    if control is not None:
        ok = client.request(method, path, json=body_of(b, b), headers=headers)
        assert ok.status_code == control, f"control request is malformed: {ok.status_code} {ok.text}"

    attack = client.request(method, path, json=body_of(b, a), headers=headers)
    assert attack.status_code == expected, (attack.status_code, attack.text)
    if case[0] in EXISTENCE_ORACLE_CASES:
        # Foreign id must answer exactly like a random, nonexistent one.
        missing = client.request(
            method, path, json=body_of(b, MissingRefs()), headers=headers
        )
        assert (missing.status_code, missing.json()) == (
            attack.status_code,
            attack.json(),
        ), (missing.text, attack.text)
    if expected != 422:  # a 422 legitimately echoes the caller's own input
        assert not leaks(attack, a)
    assert snapshot(client, a) == before


MASS_ASSIGNMENT = [
    ("POST", "/agents", lambda w: {"name": n()}, "/agents"),
    ("POST", "/providers", lambda w: {"kind": "llm", "provider": "openai"}, "/providers"),
    ("POST", "/knowledge/documents", lambda w: {"title": n(), "content": "c"}, "/knowledge/documents"),
    ("POST", "/tools", lambda w: {"name": n(), "description": "d"}, "/tools"),
    ("POST", "/voices/design", lambda w: {"name": n(), "design_prompt": "calm"}, "/voices"),
    ("POST", "/deployments", lambda w: {"agent_id": w["agent"], "name": n(), "channel": "web"}, "/deployments"),
    ("POST", "/campaigns", lambda w: {"name": n(), "agent_id": w["agent"], "deployment_id": w["deployment"]}, "/campaigns"),
    ("POST", "/voice/sessions", lambda w: {"agent_id": w["agent"]}, "/conversations"),
    ("POST", "/translations", lambda w: {"languages": ["en", "ko"], "creator": {"language": "en"}}, "/conversations"),
]


@pytest.mark.parametrize("method,path,body_of,listing", MASS_ASSIGNMENT, ids=[c[1] for c in MASS_ASSIGNMENT])
def test_organization_id_in_a_create_body_cannot_choose_the_owner(
    client, worlds, method, path, body_of, listing
):
    a, b = worlds
    body = {**body_of(b), "organization_id": a.org_id}
    response = client.request(method, path, json=body, headers=b.admin)
    assert response.status_code < 500, response.text
    if response.status_code >= 300:
        return  # rejected outright: fine
    payload = response.json()
    created = payload.get("id") or payload.get("conversation_id")
    assert created, payload
    ids_in_a = {item["id"] for item in client.get(listing, headers=a.owner).json()}
    ids_in_b = {item["id"] for item in client.get(listing, headers=b.owner).json()}
    assert created not in ids_in_a, "row created in org A by org B's key"
    assert created in ids_in_b
    assert not leaks(response, a)


# --------------------------------------------------------------------- runtime routes


def test_runtime_token_routes(client, worlds):
    a, b = worlds
    event = {
        "event_id": str(uuid4()),
        "event_type": "TranscriptFinal",
        "session_id": a["session"],
        "conversation_id": a["conversation"],
        "agent_id": a["agent"],
        "payload": {"role": "user", "content": "x"},
    }
    search = {"query": "refund policy", "limit": 5}
    for headers in (b.owner, b.admin, b.member, SUPER, {}):
        assert client.post("/runtime/events", json=event, headers=headers).status_code in (401, 403)
        r = client.post(
            f"/runtime/agents/{a['agent']}/knowledge/search", json=search, headers=headers
        )
        assert r.status_code in (401, 403), r.text
    # The runtime token is not a tenant credential: it reads nothing else.
    for path in ("/agents", "/conversations", "/audit-log", f"/agents/{a['agent']}"):
        assert client.get(path, headers=RUNTIME).status_code in (401, 403), path
    # Knowledge search is bound to the agent's org: each agent only retrieves its own documents.
    for own, other in ((a, b), (b, a)):
        response = client.post(
            f"/runtime/agents/{own['agent']}/knowledge/search", json=search, headers=RUNTIME
        )
        assert response.status_code == 200, response.text
        assert not leaks(response, other)
    # Ingest ignores caller-claimed ownership: a foreign agent id on org A's session is rejected.
    mismatched = {**event, "agent_id": b["agent"]}
    assert client.post("/runtime/events", json=mismatched, headers=RUNTIME).status_code in range(
        400, 500
    )


# ------------------------------------------------------------------ e. superadmin scoping

LIST_KINDS = {
    "/agents": "agent",
    "/providers": "llm_provider",
    "/voices": "voice_clone",
    "/knowledge/documents": "document",
    "/tools": "tool",
    "/deployments": "deployment",
    "/campaigns": "campaign",
    "/conversations": "conversation",
}
DETAIL_PATHS = [
    "/agents/{agent}",
    "/voices/{voice_clone}",
    "/knowledge/documents/{document}",
    "/tools/{tool}",
    "/deployments/{deployment}",
    "/campaigns/{campaign}",
    "/conversations/{conversation}",
    "/conversations/{conversation}/messages",
    "/conversations/{conversation}/debug",
    "/conversations/{conversation}/reports",
    "/voice/sessions/{session}",
]


def acting_as(world) -> dict[str, str]:
    return {**SUPER, "X-Organization-Id": world.org_id}


def test_superadmin_header_selects_exactly_one_org(client, worlds):
    a, b = worlds
    for view, other in ((a, b), (b, a)):
        headers = acting_as(view)
        for path, kind in LIST_KINDS.items():
            response = client.get(path, headers=headers)
            assert response.status_code == 200, (path, response.text)
            assert view.ids[kind] in {item["id"] for item in response.json()}, path
            assert not leaks(response, other), (path, "superadmin saw the other org")
        for path in ("/audit-log?limit=200", "/api-keys", "/organization", "/analytics/summary"):
            response = client.get(path, headers=headers)
            assert response.status_code == 200, (path, response.text)
            assert not leaks(response, other), path
        assert client.get("/organization", headers=headers).json()["id"] == view.org_id
        # By id: own org's resources resolve, the other org's do not.
        for template in DETAIL_PATHS:
            assert client.get(template.format(**view.ids), headers=headers).status_code == 200
            foreign = client.get(template.format(**other.ids), headers=headers)
            assert foreign.status_code == 404, (template, foreign.status_code)
        # Mutations cannot cross either.
        attempts = [
            ("DELETE", f"/agents/{other['agent']}", None),
            ("PATCH", f"/tools/{other['tool']}", {"description": "x"}),
            ("DELETE", f"/knowledge/documents/{other['document']}", None),
            ("POST", f"/voices/{other['voice_clone']}/status", {"status": "disabled"}),
            ("POST", f"/voice/sessions/{other['session']}/end", None),
            ("POST", f"/campaigns/{other['campaign']}/status", {"status": "cancelled"}),
        ]
        for method, path, body in attempts:
            response = client.request(method, path, json=body, headers=headers)
            assert response.status_code == 404, (method, path, response.status_code)
    assert snapshot(client, a)["/agents/{agent}"][0] == 200
    assert snapshot(client, b)["/agents/{agent}"][0] == 200


def test_superadmin_creates_in_the_selected_org(client, worlds):
    a, b = worlds
    created = client.post("/agents", json={"name": n()}, headers=acting_as(a))
    assert created.status_code == 201, created.text
    new_id = created.json()["id"]
    assert new_id in {i["id"] for i in client.get("/agents", headers=a.member).json()}
    assert new_id not in {i["id"] for i in client.get("/agents", headers=b.owner).json()}
    assert client.get(f"/agents/{new_id}", headers=b.owner).status_code == 404
    # And the audit trail of the action lands in A, not B.
    log_a = client.get(f"/audit-log?resource_id={new_id}", headers=a.admin).json()["items"]
    log_b = client.get(f"/audit-log?resource_id={new_id}", headers=b.admin).json()["items"]
    assert log_b == []
    assert all(item["resource_id"] == new_id for item in log_a)


# ------------------------------------------------------------------------ f. audit log


def audit_items(client, world, query="?limit=200"):
    response = client.get("/audit-log" + query, headers=world.admin)
    assert response.status_code == 200, response.text
    return response.json()["items"]


def test_audit_log_is_per_org(client, worlds):
    a, b = worlds
    a_log = audit_items(client, a)
    b_log = audit_items(client, b)
    assert a_log and b_log, "worlds should have produced audit rows"
    assert a.ids["voice_clone"] in json.dumps(a_log), "control: A's log mentions A's voice"
    # Org B hammers org A's resources with every cross-org call from the id matrix.
    a_ids_in_log = {item["id"] for item in a_log}
    for method, path in PARAM_CASES:
        (param,) = path_params(path)
        make_body = PARAM_BODIES.get((method, path))
        client.request(
            method,
            path.format(**{param: ident(a, param)}),
            json=make_body() if make_body else None,
            headers=b.admin,
        )

    a_after = audit_items(client, a)
    b_after = audit_items(client, b)
    for label, items, other in (("B", b_after, a), ("B-before", b_log, a)):
        text = json.dumps(items)
        found = sorted(m for m in other.leak_markers | a_ids_in_log if m in text)
        assert not found, f"org {label} audit log mentions org A: {found}"
    # Failed foreign calls wrote nothing into A's log.
    assert {item["id"] for item in a_after} == a_ids_in_log, "org B's attacks altered A's audit log"

    # Filters cannot reach across: A's resource ids, filtered on, give B nothing.
    for kind in ("agent", "voice_clone", "document", "tool", "deployment", "campaign", "session"):
        assert audit_items(client, b, f"?resource_id={a[kind]}") == []
    # Pagination cursor from A's log is meaningless to B.
    cursor = a_log[-1]["created_at"]
    paged = audit_items(client, b, f"?limit=200&before={quote(cursor, safe='')}")
    assert not any(m in json.dumps(paged) for m in a.leak_markers)

    # Members cannot read the audit log at all.
    assert client.get("/audit-log", headers=b.member).status_code == 403


def test_leak_scanner_detects_foreign_ids(client, worlds):
    """Guard the guard: the scanner must flag A's data when A's own key fetches it."""

    a, b = worlds
    for path in ("/agents", "/voices", "/audit-log", "/conversations"):
        response = client.get(path, headers=a.owner)
        assert leaks(response, a), path
        assert not leaks(response, b), path
    assert snapshot(client, a) != snapshot(client, b)


def test_session_token_of_one_org_reaches_nothing_in_another(client, worlds):
    a, b = worlds
    token = {"Authorization": f"Bearer {a.extra['session_token']}"}
    own = client.get(f"/voice/sessions/{a['session']}", headers=token)
    assert own.status_code == 200, own.text
    probes = [
        ("GET", f"/voice/sessions/{b['session']}", None),
        ("POST", f"/voice/sessions/{b['session']}/end", None),
        ("POST", f"/conversations/{b['conversation']}/reports", {"note": "planted"}),
        ("GET", f"/conversations/{b['conversation']}/debug", None),
        ("GET", f"/conversations/{b['conversation']}/messages", None),
        ("GET", "/agents", None),
        ("GET", f"/agents/{b['agent']}", None),
        ("GET", "/audit-log", None),
    ]
    for method, path, body in probes:
        response = client.request(method, path, json=body, headers=token)
        assert response.status_code in (401, 403, 404), (method, path, response.status_code)
        assert not leaks(response, b), (method, path)

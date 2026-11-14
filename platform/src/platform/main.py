"""Voice Labs control-plane API.

Routes live in per-area routers; this module builds the app and wires them in.
"""

import os
from contextlib import asynccontextmanager

import httpx  # noqa: F401  (tests patch ``platform.main.httpx.AsyncClient``)
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import (
    agents,
    analytics,
    audit,
    auth,
    calls,
    campaigns,
    conversations,
    debug,
    demo,
    deployments,
    handoffs,
    knowledge,
    organizations,
    reaper,
    runtime_ingest,
    sessions,
    templates,
    tools,
    translations,
    voices,
)
from . import health as provider_health
from .agents import resolve_interpreter_engine  # noqa: F401  (re-exported)
from .db import Base, SessionLocal, engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    auth.log_auth_status()
    if os.getenv("PLATFORM_AUTO_CREATE_SCHEMA") == "1":
        Base.metadata.create_all(bind=engine)
        # Migrated databases get the default org from 0008; create_all does not.
        with SessionLocal() as db:
            organizations.ensure_default_org(db)
    reaper_task = reaper.start_reaper()
    try:
        yield
    finally:
        await reaper.stop_reaper(reaper_task)


app = FastAPI(
    title="Voice Labs Platform",
    version="0.1.0",
    lifespan=lifespan,
    # Secure by default: every route requires the admin key (when PLATFORM_API_KEYS is
    # set) unless auth.py explicitly allowlists it.
    dependencies=[Depends(auth.authenticate)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("PLATFORM_WEB_ORIGIN", "http://localhost:3000").split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=[
        "content-type",
        "x-runtime-token",
        "authorization",
        "x-organization-id",
        "x-request-id",
        "x-staff-key",
    ],
    expose_headers=["x-request-id"],
    # Opt-in for a hosted web origin calling a Platform on the viewer's localhost.
    **(
        {"allow_private_network": True}
        if os.getenv("PLATFORM_ALLOW_PRIVATE_NETWORK") == "1"
        else {}
    ),
)


@app.middleware("http")
async def echo_request_id(request: Request, call_next):
    """Return the request id (set by ``auth.authenticate``) as ``X-Request-Id``."""

    response = await call_next(request)
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        response.headers["X-Request-Id"] = request_id
    return response


for router in (
    analytics.router,
    voices.router,
    deployments.router,
    campaigns.router,
    tools.router,
    knowledge.router,
    templates.router,
    provider_health.router,
):
    app.include_router(router)

app.include_router(audit.router)
app.include_router(debug.router)
app.include_router(organizations.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


for router in (
    agents.router,
    sessions.router,
    translations.router,
    calls.router,
    demo.router,
    handoffs.router,
    conversations.router,
    runtime_ingest.router,
):
    app.include_router(router)

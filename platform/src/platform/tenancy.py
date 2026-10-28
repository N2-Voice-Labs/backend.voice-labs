"""Tenant isolation safety net (organisations plan, T4).

Layers, from implicit to explicit:

1. **Write stamping** -- a global ``before_flush`` listener fills ``organization_id`` on
   new :class:`TenantMixin` rows from ``session.info["org_id"]`` (default organisation
   when unset) and refuses rows that name a different organisation.
2. **Read filtering** -- a global ``do_orm_execute`` listener adds
   ``with_loader_criteria(TenantMixin, organization_id == org_id)`` to ORM SELECT /
   UPDATE / DELETE statements (lazy and eager relationship loads included) on sessions
   whose ``info["org_id"]`` is set, unless ``info["tenant_bypass"]`` is truthy.
3. **Explicit helpers** -- :func:`get_owned`, :func:`link_check`, :func:`scoped`.
   Routers should use these; the listener is only a backstop.

SQLAlchemy caveats (verified in tests/test_tenancy.py):

* ``Session.get`` returns an identity-map hit *without emitting SQL*, so the criteria is
  never applied to it. A row loaded earlier in the same session (e.g. under
  :func:`bypass`) leaks through. :func:`get_owned` therefore never uses ``Session.get``;
  it runs an explicit ``SELECT ... WHERE id = :id AND organization_id = :org``.
* Statements that are not ORM-enabled (``select(table.c.x)``, ``text()``, ``insert``)
  are not touched. Entity-based ``select(func.count(Model.id))`` and
  ``select(func.count()).select_from(Model)`` *are* filtered, but use :func:`scoped`
  for aggregate/analytics queries rather than relying on that.
* Bulk ``update()``/``delete()`` through ``session.execute`` are filtered; the legacy
  ``Query.update``/``Query.delete`` path is not used in this code base.
* Rows already in the identity map are not re-filtered; do not share one session
  across organisations -- use :func:`scoped_session_for`.

Principals without an organisation (``runtime``, public routes) get an unfiltered
session from :func:`get_scoped_db`; their routes must resolve the organisation from the
VoiceSession and check it explicitly.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from fastapi import Depends, HTTPException, Request
from sqlalchemy import Select, event, select
from sqlalchemy.orm import ORMExecuteState, Session, with_loader_criteria

from . import db as _db
from .db import get_db
from .models import DEFAULT_ORGANIZATION_ID, TenantMixin

if TYPE_CHECKING:  # context.py imports models; avoid a cycle at import time.
    from .context import AuthContext

T = TypeVar("T")

ORG_KEY = "org_id"
BYPASS_KEY = "tenant_bypass"


@event.listens_for(Session, "before_flush")
def stamp_organization_id(session: Session, flush_context, instances) -> None:
    org_id = session.info.get(ORG_KEY)
    bypass = bool(session.info.get(BYPASS_KEY))
    for obj in session.new:
        if not isinstance(obj, TenantMixin):
            continue
        if obj.organization_id is None:
            obj.organization_id = org_id or DEFAULT_ORGANIZATION_ID
        elif org_id and not bypass and obj.organization_id != org_id:
            raise ValueError(
                f"{type(obj).__name__} organization_id {obj.organization_id!r} "
                f"does not match session organization {org_id!r}"
            )


@event.listens_for(Session, "do_orm_execute")
def filter_by_organization(state: ORMExecuteState) -> None:
    session = state.session
    org_id = session.info.get(ORG_KEY)
    if not org_id or session.info.get(BYPASS_KEY):
        return
    if state.is_column_load or not (
        state.is_select or state.is_update or state.is_delete
    ):
        return
    state.statement = state.statement.options(
        with_loader_criteria(
            TenantMixin,
            lambda cls: cls.organization_id == org_id,
            include_aliases=True,
        )
    )


# --------------------------------------------------------------------- sessions


def get_scoped_db(request: Request, db: Session = Depends(get_db)) -> Iterator[Session]:
    """``get_db`` plus ``info["org_id"]`` from ``request.state.ctx`` when it has one.

    Built on ``get_db`` so test overrides apply and a route sharing both dependencies
    gets the same session. Runtime/public requests (no ctx or ``org_id`` None) stay
    unfiltered.
    """

    ctx = getattr(request.state, "ctx", None)
    org_id = getattr(ctx, "org_id", None)
    if org_id:
        db.info[ORG_KEY] = org_id
    try:
        yield db
    finally:
        if org_id:
            db.info.pop(ORG_KEY, None)


@contextmanager
def scoped_session_for(org_id: str) -> Iterator[Session]:
    """Standalone session pinned to one organisation (background jobs, system ctx)."""

    db = _db.SessionLocal()
    db.info[ORG_KEY] = org_id
    try:
        yield db
    finally:
        db.close()


@contextmanager
def bypass(session: Session) -> Iterator[Session]:
    """Temporarily disable read filtering (and the stamping mismatch check)."""

    previous = session.info.get(BYPASS_KEY)
    session.info[BYPASS_KEY] = True
    try:
        yield session
    finally:
        if previous is None:
            session.info.pop(BYPASS_KEY, None)
        else:
            session.info[BYPASS_KEY] = previous


# ---------------------------------------------------------------------- helpers


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def not_found_detail_for(model: type) -> str:
    """Router-style detail: ``Agent`` -> ``agent_not_found``."""

    return f"{_snake(model.__name__)}_not_found"


def scoped(db: Session, ctx: "AuthContext", model: type[T]) -> Select[tuple[T]]:
    """``select(model)`` filtered to ``ctx.org_id`` (unfiltered when it is None)."""

    stmt = select(model)
    if ctx.org_id is not None:
        stmt = stmt.where(model.organization_id == ctx.org_id)  # type: ignore[attr-defined]
    return stmt


def get_owned(
    db: Session,
    ctx: "AuthContext",
    model: type[T],
    id: Any,
    *,
    not_found_detail: str | None = None,
) -> T:
    """Row by primary key within ``ctx.org_id``; 404 when missing *or* cross-org.

    Never 403: a foreign row must be indistinguishable from a missing one. Uses an
    explicit SELECT instead of ``Session.get`` so identity-map hits cannot leak.
    """

    stmt = scoped(db, ctx, model).where(model.id == id)  # type: ignore[attr-defined]
    row = db.scalars(stmt).first()
    if row is None:
        raise HTTPException(
            status_code=404, detail=not_found_detail or not_found_detail_for(model)
        )
    return row


def link_check(ctx: "AuthContext", *rows: Any) -> None:
    """Assert every linked row belongs to ``ctx.org_id`` (e.g. agent_id on a campaign).

    Raises 404 ``<model>_not_found`` -- same status/detail as a missing row, so a foreign
    id is not distinguishable from a nonexistent one (we pick 404 over 409 for that
    reason). ``None`` rows are skipped; a ctx without org (``org_id`` None) skips the
    check entirely.
    """

    if ctx.org_id is None:
        return
    for row in rows:
        if row is None:
            continue
        if getattr(row, "organization_id", ctx.org_id) != ctx.org_id:
            raise HTTPException(
                status_code=404, detail=not_found_detail_for(type(row))
            )

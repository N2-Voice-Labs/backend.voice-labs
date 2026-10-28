"""Campaign API and outbound dialer.

Campaigns and their targets are durable records. Dialing is opt-in: only when
``LIVEKIT_SIP_TRUNK_ID`` (plus the LiveKit URL/key/secret) is configured does moving a
campaign to ``running`` start an in-process dispatcher that opens a voice session per
pending target and dials it through LiveKit SIP (see :mod:`.sip`). Without that
configuration nothing dials and targets stay ``pending``.

Target lifecycle: ``pending -> calling -> completed | failed | no_answer``. A failed
or unanswered attempt goes back to ``pending`` while ``attempts < max_attempts``.
Session-creation refusals for capacity/admission reasons (active-session cap, rate limit,
runtime at capacity, provider/runtime down) are *deferred*: the target returns to
``pending`` with its attempt refunded and the worker backs off (exponential, capped at
60 s) instead of burning attempts. ``completed`` means the callee answered and was connected to the agent; the call
itself is tracked on the conversation (``conversation_id``).

Limitations: the dispatcher is per-process (not safe to run on several replicas), does
not honour the schedule window or ``retry_delay_*`` configuration, and a crash leaves
the in-flight target in ``calling``.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import sip
from .context import AuthContext, require
from .db import SessionLocal
from .models import Agent as AgentRecord
from .models import Campaign as CampaignRecord
from .models import CampaignTarget as TargetRecord
from .models import Deployment as DeploymentRecord
from .tenancy import get_owned, get_scoped_db, scoped, scoped_session_for
from .validation import reject_secret_configuration, validate_e164

router = APIRouter(tags=["campaigns"])
_logger = logging.getLogger(__name__)

CampaignStatus = Literal[
    "draft", "scheduled", "running", "paused", "completed", "cancelled"
]
TRANSITIONS: dict[str, set[str]] = {
    "draft": {"scheduled", "cancelled"},
    "scheduled": {"draft", "running", "cancelled"},
    "running": {"paused", "completed", "cancelled"},
    "paused": {"running", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
EDITABLE = {"draft", "scheduled"}
TARGETS_EDITABLE = {"draft", "scheduled", "paused"}


def check_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise HTTPException(status_code=422, detail="invalid_timezone") from None
    return value


def as_utc(value: datetime | None) -> datetime | None:
    """Treat naive datetimes (including those read back from SQLite) as UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def check_schedule(start: datetime | None, end: datetime | None) -> None:
    if start is not None and end is not None and end <= start:
        raise HTTPException(status_code=422, detail="invalid_schedule")


class CampaignIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    agent_id: UUID
    deployment_id: UUID
    schedule_start: datetime | None = None
    schedule_end: datetime | None = None
    timezone: str = Field(default="UTC", max_length=64)
    max_attempts: int = Field(default=1, ge=1, le=10)
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schedule_start", "schedule_end")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        return as_utc(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class Campaign(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    agent_id: UUID
    deployment_id: UUID
    status: CampaignStatus
    schedule_start: datetime | None
    schedule_end: datetime | None
    timezone: str
    max_attempts: int
    configuration: dict[str, Any]
    created_at: Any
    updated_at: Any


class CampaignDetail(Campaign):
    target_counts: dict[str, int]
    dialing: Literal["disabled", "active"] = "disabled"


class CampaignPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    schedule_start: datetime | None = None
    schedule_end: datetime | None = None
    timezone: str | None = Field(default=None, max_length=64)
    max_attempts: int | None = Field(default=None, ge=1, le=10)
    configuration: dict[str, Any] | None = None

    @field_validator("name", "timezone", "max_attempts", "configuration", mode="before")
    @classmethod
    def reject_null_for_required_fields(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value

    @field_validator("schedule_start", "schedule_end")
    @classmethod
    def normalize_datetime(cls, value: datetime | None) -> datetime | None:
        return as_utc(value)

    @field_validator("configuration")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class TargetIn(BaseModel):
    phone_number: str
    name: str = Field(default="", max_length=200)
    variables: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phone_number")
    @classmethod
    def check_phone_number(cls, value: str) -> str:
        return validate_e164(value)


class TargetsIn(BaseModel):
    targets: list[TargetIn] = Field(min_length=1, max_length=1000)


class TargetsAdded(BaseModel):
    added: int
    skipped_duplicates: int


class Target(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    campaign_id: UUID
    phone_number: str
    name: str
    variables: dict[str, Any]
    status: str
    attempts: int
    conversation_id: UUID | None
    created_at: Any
    updated_at: Any


class DialNextResult(BaseModel):
    result: Literal["completed", "failed", "no_answer", "deferred"]
    target: Target


class StatusIn(BaseModel):
    status: CampaignStatus


def get_campaign_or_404(
    campaign_id: UUID, db: Session, ctx: AuthContext
) -> CampaignRecord:
    return get_owned(
        db, ctx, CampaignRecord, str(campaign_id), not_found_detail="campaign_not_found"
    )


def campaign_detail(record: CampaignRecord, db: Session) -> CampaignDetail:
    counts = dict(
        db.query(TargetRecord.status, func.count())
        .filter(
            TargetRecord.campaign_id == record.id,
            TargetRecord.organization_id == record.organization_id,
        )
        .group_by(TargetRecord.status)
        .all()
    )
    return CampaignDetail(
        **Campaign.model_validate(record).model_dump(),
        target_counts=counts,
        dialing="active" if sip.sip_configured() else "disabled",
    )


@router.post(
    "/campaigns", response_model=CampaignDetail, status_code=status.HTTP_201_CREATED
)
async def create_campaign(
    payload: CampaignIn,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> CampaignDetail:
    check_timezone(payload.timezone)
    check_schedule(payload.schedule_start, payload.schedule_end)
    # Org-scoped lookups: another organisation's agent/deployment is indistinguishable
    # from a nonexistent one (both 422 invalid_*_id).
    agent = db.scalars(
        scoped(db, ctx, AgentRecord).where(AgentRecord.id == str(payload.agent_id))
    ).first()
    deployment = db.scalars(
        scoped(db, ctx, DeploymentRecord).where(
            DeploymentRecord.id == str(payload.deployment_id)
        )
    ).first()
    if agent is None:
        raise HTTPException(status_code=422, detail="invalid_agent_id")
    if deployment is None:
        raise HTTPException(status_code=422, detail="invalid_deployment_id")
    if deployment.agent_id != str(payload.agent_id):
        raise HTTPException(status_code=409, detail="deployment_agent_mismatch")
    if deployment.channel != "sip":
        raise HTTPException(status_code=422, detail="campaign_requires_sip_deployment")
    record = CampaignRecord(
        **{
            **payload.model_dump(),
            "agent_id": str(payload.agent_id),
            "deployment_id": str(payload.deployment_id),
        }
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return campaign_detail(record, db)


@router.get("/campaigns", response_model=list[Campaign])
async def list_campaigns(
    agent_id: UUID | None = None,
    status: CampaignStatus | None = None,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> list[Campaign]:
    query = db.query(CampaignRecord).filter(CampaignRecord.organization_id == ctx.org_id)
    if agent_id is not None:
        query = query.filter(CampaignRecord.agent_id == str(agent_id))
    if status is not None:
        query = query.filter(CampaignRecord.status == status)
    return [
        Campaign.model_validate(item) for item in query.order_by(CampaignRecord.created_at)
    ]


@router.get("/campaigns/{campaign_id}", response_model=CampaignDetail)
async def get_campaign(
    campaign_id: UUID,
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> CampaignDetail:
    return campaign_detail(get_campaign_or_404(campaign_id, db, ctx), db)


@router.patch("/campaigns/{campaign_id}", response_model=CampaignDetail)
async def update_campaign(
    campaign_id: UUID,
    payload: CampaignPatch,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> CampaignDetail:
    record = get_campaign_or_404(campaign_id, db, ctx)
    if record.status not in EDITABLE:
        raise HTTPException(status_code=409, detail="campaign_not_editable")
    changes = payload.model_dump(exclude_unset=True)
    if "timezone" in changes:
        check_timezone(changes["timezone"])
    check_schedule(
        as_utc(changes.get("schedule_start", record.schedule_start)),
        as_utc(changes.get("schedule_end", record.schedule_end)),
    )
    for key, value in changes.items():
        setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return campaign_detail(record, db)


@router.post("/campaigns/{campaign_id}/targets", response_model=TargetsAdded)
async def add_targets(
    campaign_id: UUID,
    payload: TargetsIn,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> TargetsAdded:
    record = get_campaign_or_404(campaign_id, db, ctx)
    if record.status not in TARGETS_EDITABLE:
        raise HTTPException(status_code=409, detail="campaign_not_editable")
    unique: dict[str, TargetIn] = {}
    for item in payload.targets:
        unique.setdefault(item.phone_number, item)
    for _ in range(2):  # retry once if a concurrent request adds the same numbers
        existing = {
            phone
            for (phone,) in db.query(TargetRecord.phone_number).filter(
                TargetRecord.campaign_id == record.id,
                TargetRecord.phone_number.in_(unique),
            )
        }
        new = [
            TargetRecord(
                campaign_id=record.id,
                organization_id=record.organization_id,
                **item.model_dump(),
            )
            for phone, item in unique.items()
            if phone not in existing
        ]
        db.add_all(new)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            continue
        return TargetsAdded(
            added=len(new), skipped_duplicates=len(payload.targets) - len(new)
        )
    raise HTTPException(status_code=409, detail="target_conflict")


@router.get("/campaigns/{campaign_id}/targets", response_model=list[Target])
async def list_targets(
    campaign_id: UUID,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    ctx: AuthContext = Depends(require("member")),
    db: Session = Depends(get_scoped_db),
) -> list[Target]:
    record = get_campaign_or_404(campaign_id, db, ctx)
    query = db.query(TargetRecord).filter(
        TargetRecord.campaign_id == record.id,
        TargetRecord.organization_id == record.organization_id,
    )
    if status is not None:
        query = query.filter(TargetRecord.status == status)
    rows = query.order_by(TargetRecord.created_at, TargetRecord.id)
    return [Target.model_validate(item) for item in rows.offset(offset).limit(limit)]


@router.post("/campaigns/{campaign_id}/status", response_model=CampaignDetail)
async def change_campaign_status(
    campaign_id: UUID,
    payload: StatusIn,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> CampaignDetail:
    record = get_campaign_or_404(campaign_id, db, ctx)
    if payload.status not in TRANSITIONS[record.status]:
        raise HTTPException(
            status_code=409, detail="invalid_campaign_status_transition"
        )
    if payload.status in {"scheduled", "running"}:
        deployment = db.scalars(
            scoped(db, ctx, DeploymentRecord).where(
                DeploymentRecord.id == record.deployment_id
            )
        ).first()
        if deployment is None or deployment.status != "active":
            raise HTTPException(status_code=409, detail="deployment_not_active")
        has_targets = (
            db.query(TargetRecord.id)
            .filter(TargetRecord.campaign_id == record.id)
            .first()
        )
        if has_targets is None:
            raise HTTPException(status_code=409, detail="campaign_has_no_targets")
    record.status = payload.status
    db.commit()
    db.refresh(record)
    if payload.status == "running" and sip.sip_configured():
        start_dispatcher(record.id, max_concurrent_calls(record))
    return campaign_detail(record, db)


@router.post("/campaigns/{campaign_id}/dial-next", response_model=DialNextResult)
async def dial_next(
    campaign_id: UUID,
    ctx: AuthContext = Depends(require("admin")),
    db: Session = Depends(get_scoped_db),
) -> DialNextResult:
    """Dial exactly one pending target and wait for the attempt to resolve.

    For demos and tests. Returns once the callee answered or the attempt failed; it does
    not wait for the call to end.
    """

    record = get_campaign_or_404(campaign_id, db, ctx)
    if not sip.sip_configured():
        raise HTTPException(status_code=409, detail="sip_not_configured")
    if record.status != "running":
        raise HTTPException(status_code=409, detail="campaign_not_running")
    target = claim_next_target(db, record)
    if target is None:
        raise HTTPException(status_code=409, detail="no_pending_targets")
    result = await process_target(db, record, target, wait_for_end=False)
    finish_if_drained(db, record)
    return DialNextResult(result=result, target=Target.model_validate(target))


# -- dispatcher -------------------------------------------------------------------

MAX_CONCURRENT_CALLS = 10
POLL_INTERVAL_S = 3.0
CALL_WAIT_TIMEOUT_S = 3600.0
_TERMINAL_SESSION_STATUSES = {"ended", "failed"}
# Admission/capacity refusals from session creation. They say nothing about the callee,
# so the target goes back to pending without consuming an attempt and the worker backs off.
DEFERRABLE_SESSION_CODES = frozenset(
    {
        "too_many_active_sessions",
        "rate_limited",
        "runtime_at_capacity",
        "provider_unavailable",
        "runtime_unavailable",
    }
)
DEFER_BACKOFF_BASE_S = 2.0
DEFER_BACKOFF_MAX_S = 60.0
_dispatchers: dict[str, "asyncio.Task[None]"] = {}


def max_concurrent_calls(record: CampaignRecord) -> int:
    """``configuration.max_concurrent_calls`` (default 1, capped)."""
    value = (record.configuration or {}).get("max_concurrent_calls", 1)
    if isinstance(value, bool) or not isinstance(value, int):
        return 1
    return max(1, min(value, MAX_CONCURRENT_CALLS))


def claim_next_target(db: Session, record: CampaignRecord) -> TargetRecord | None:
    """Atomically move the next dialable target to ``calling`` and count the attempt.

    Never-tried targets go first, then retries; ties break by creation order.
    """

    candidates = (
        db.query(TargetRecord)
        .filter(
            TargetRecord.campaign_id == record.id,
            TargetRecord.status == "pending",
            TargetRecord.attempts < record.max_attempts,
        )
        .order_by(TargetRecord.attempts, TargetRecord.created_at, TargetRecord.id)
        .limit(5)
        .all()
    )
    for candidate in candidates:
        claimed = db.execute(
            update(TargetRecord)
            .where(TargetRecord.id == candidate.id, TargetRecord.status == "pending")
            .values(status="calling", attempts=TargetRecord.attempts + 1)
        )
        db.commit()
        if claimed.rowcount == 1:
            db.refresh(candidate)
            return candidate
    return None


def defer_delay(consecutive_deferrals: int) -> float:
    """Exponential backoff (base * 2^(n-1)) capped at ``DEFER_BACKOFF_MAX_S``."""

    exponent = max(consecutive_deferrals, 1) - 1
    return min(DEFER_BACKOFF_BASE_S * (2**exponent), DEFER_BACKOFF_MAX_S)


async def _defer_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def deferral_code(exc: sip.SessionCreateError) -> str | None:
    """The deferrable admission/capacity code behind a failed session create, else None.

    ``SessionCreateError`` only carries the status; the original ``HTTPException`` (with
    the structured ``detail``) is its ``__cause__``.
    """

    if exc.status_code not in (429, 503):
        return None
    detail = getattr(exc.__cause__, "detail", None)
    code = detail.get("code") if isinstance(detail, dict) else detail
    return code if isinstance(code, str) and code in DEFERRABLE_SESSION_CODES else None


async def _wait_for_session_end(session_id: str, org_id: str) -> None:
    waited = 0.0
    unreadable = 0
    while waited < CALL_WAIT_TIMEOUT_S:
        # A fresh short-lived session per poll: no transaction is held while sleeping.
        status_value = await sip.get_session_status(session_id, org_id=org_id)
        unreadable = unreadable + 1 if status_value is None else 0
        if status_value in _TERMINAL_SESSION_STATUSES or unreadable >= 3:
            return
        await asyncio.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S


async def process_target(
    db: Session,
    record: CampaignRecord,
    target: TargetRecord,
    *,
    wait_for_end: bool,
) -> Literal["completed", "failed", "no_answer", "deferred"]:
    """Run one claimed attempt: open a session, dial into its room, record the result.

    ``deferred`` means session creation was refused for capacity/admission reasons: the
    claim's attempt is refunded and the target returns to ``pending``.
    """

    outcome: Literal["completed", "failed", "no_answer", "deferred"] = "failed"
    session_id: str | None = None
    try:
        session = await sip.create_voice_session(
            record.agent_id,
            record.deployment_id,
            org_id=record.organization_id,
            db=db,
        )
        session_id = str(session["session_id"])
        target.conversation_id = str(session["conversation_id"])
        db.commit()
        await sip.dial(
            target.phone_number,
            str(session["room_name"]),
            f"{sip.SIP_IDENTITY_PREFIX}{target.id}",
            display_name=target.name or None,
        )
        outcome = "completed"
    except sip.SipDialError as exc:
        outcome = "no_answer" if exc.no_answer else "failed"
        _logger.warning("Dial failed for target %s: %s", target.id, exc)
    except (sip.SessionCreateError, sip.SipError) as exc:
        db.rollback()
        code = deferral_code(exc) if isinstance(exc, sip.SessionCreateError) else None
        if code is not None:
            outcome = "deferred"
            _logger.warning("Deferring target %s: %s (%s)", target.id, exc, code)
        else:
            _logger.warning("Attempt failed for target %s: %s", target.id, exc)
    except Exception:
        _logger.exception("Unexpected dialer error for target %s", target.id)
    if outcome != "completed" and session_id is not None:
        await sip.end_voice_session(session_id, org_id=record.organization_id)
    db.refresh(target)
    if outcome == "deferred":
        target.attempts = max(target.attempts - 1, 0)  # undo the claim's increment
        target.status = "pending"
    elif outcome == "completed" or target.attempts >= record.max_attempts:
        target.status = outcome
    else:
        target.status = "pending"  # retry later
    db.commit()
    if outcome == "completed" and wait_for_end and session_id is not None:
        await _wait_for_session_end(session_id, record.organization_id)
    return outcome


def finish_if_drained(db: Session, record: CampaignRecord) -> bool:
    """Mark a running campaign ``completed`` once no target is pending or calling."""

    db.refresh(record)
    if record.status != "running":
        return False
    open_targets = (
        db.query(TargetRecord.id)
        .filter(
            TargetRecord.campaign_id == record.id,
            TargetRecord.status.in_(("pending", "calling")),
        )
        .first()
    )
    if open_targets is not None:
        return False
    record.status = "completed"
    db.commit()
    return True


async def run_campaign_batch(campaign_id: str, max_concurrent: int = 1) -> None:
    """Dial a running campaign's pending targets, ``max_concurrent`` calls at a time.

    Runs under the campaign's own organisation: every worker uses a session pinned to
    ``campaign.organization_id`` (:func:`scoped_session_for`) and the sessions it opens
    are created with a ``system`` context for that organisation. Each worker re-reads the
    campaign before taking a target, so pausing or cancelling stops new calls (calls
    already connected keep going). When nothing is left the campaign is marked
    ``completed``.
    """

    with SessionLocal() as lookup:  # unscoped: the org is only known from the campaign
        org_id = lookup.scalar(
            select(CampaignRecord.organization_id).where(CampaignRecord.id == campaign_id)
        )
    if org_id is None:
        return

    async def worker() -> None:
        deferrals = 0
        while True:
            with scoped_session_for(org_id) as db:
                record = db.get(CampaignRecord, campaign_id)
                if record is None or record.status != "running":
                    return
                target = claim_next_target(db, record)
                if target is None:
                    return
                outcome = await process_target(db, record, target, wait_for_end=True)
            if outcome == "deferred":
                # Capacity/admission refusal: back off, then re-read the campaign (it may
                # have been paused) and try again. The target is still pending, so the
                # campaign cannot complete meanwhile.
                deferrals += 1
                await _defer_sleep(defer_delay(deferrals))
            else:
                deferrals = 0

    await asyncio.gather(*(worker() for _ in range(max(1, max_concurrent))))
    with scoped_session_for(org_id) as db:
        record = db.get(CampaignRecord, campaign_id)
        if record is not None:
            finish_if_drained(db, record)


def start_dispatcher(campaign_id: str, max_concurrent: int = 1) -> bool:
    """Start :func:`run_campaign_batch` in the background unless one is running."""

    existing = _dispatchers.get(campaign_id)
    if existing is not None and not existing.done():
        return False
    task = asyncio.create_task(
        run_campaign_batch(campaign_id, max_concurrent), name=f"campaign-{campaign_id}"
    )
    _dispatchers[campaign_id] = task

    def done(finished: "asyncio.Task[None]") -> None:
        if _dispatchers.get(campaign_id) is finished:
            _dispatchers.pop(campaign_id, None)
        if not finished.cancelled() and finished.exception() is not None:
            _logger.error(
                "Campaign dispatcher %s crashed", campaign_id, exc_info=finished.exception()
            )

    task.add_done_callback(done)
    return True

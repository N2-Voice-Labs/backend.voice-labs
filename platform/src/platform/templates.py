"""Agent template API."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .context import AuthContext, require
from .models import Agent as AgentRecord
from .tenancy import get_scoped_db

router = APIRouter(tags=["templates"])


AGENT_TEMPLATES = {
    "customer_support": {
        "key": "customer_support",
        "name": "Customer Support Agent",
        "description": "A professional customer support representative who handles inquiries and resolves customer issues efficiently.",
        "system_prompt": "You are a professional customer support agent. Answer customer inquiries clearly and concisely. Respond in the caller's language, including mixed English and Russian. For voice interactions, keep responses brief and natural. If asked, disclose that you are an AI assistant. Always maintain a helpful and courteous tone.",
        "enabled_languages": ["en", "ru"],
        "suggested_tools": ["check_ticket_status", "escalate_to_human", "process_refund"],
    },
    "sales": {
        "key": "sales",
        "name": "Sales Agent",
        "description": "An engaging sales representative who qualifies leads and helps close deals with professionalism.",
        "system_prompt": "You are a professional sales agent. Present product benefits persuasively and answer prospects' questions. Respond in the caller's language, including mixed English and Russian. Keep your voice responses concise and natural. If asked, disclose that you are an AI assistant. Focus on understanding customer needs and providing tailored solutions.",
        "enabled_languages": ["en", "ru"],
        "suggested_tools": ["check_inventory", "apply_discount", "schedule_demo"],
    },
    "receptionist": {
        "key": "receptionist",
        "name": "Receptionist Agent",
        "description": "A friendly receptionist who manages calls, schedules appointments, and provides information.",
        "system_prompt": "You are a professional receptionist. Handle incoming calls professionally, answer questions about business hours and services, and help schedule appointments. Respond in the caller's language, including mixed English and Russian. Keep responses brief for voice calls. If asked, disclose that you are an AI assistant. Be warm and welcoming.",
        "enabled_languages": ["en", "ru"],
        "suggested_tools": ["check_availability", "book_appointment", "transfer_call"],
    },
    "booking": {
        "key": "booking",
        "name": "Booking Assistant",
        "description": "An efficient booking assistant that helps customers reserve services or accommodations.",
        "system_prompt": "You are a professional booking assistant. Help customers find and reserve available services or accommodations. Respond in the caller's language, including mixed English and Russian. Keep voice responses concise and clear. If asked, disclose that you are an AI assistant. Provide accurate information about availability and pricing.",
        "enabled_languages": ["en", "ru"],
        "suggested_tools": ["check_availability", "confirm_booking", "apply_promo_code"],
    },
    "restaurant": {
        "key": "restaurant",
        "name": "Restaurant Host",
        "description": "A welcoming restaurant host who takes reservations and provides menu information.",
        "system_prompt": "You are a professional restaurant host. Welcome callers warmly, take dinner reservations, answer questions about the menu and specials. Respond in the caller's language, including mixed English and Russian. Keep responses brief for voice calls. If asked, disclose that you are an AI assistant. Be hospitable and informative.",
        "enabled_languages": ["en", "ru"],
        "suggested_tools": ["check_table_availability", "make_reservation", "provide_menu_info"],
    },
    "travel": {
        "key": "travel",
        "name": "Travel Assistant",
        "description": "A knowledgeable travel assistant who helps with bookings, recommendations, and travel planning.",
        "system_prompt": "You are a professional travel assistant. Help customers plan trips, provide destination recommendations, and assist with bookings. Respond in the caller's language, including mixed English and Russian. Keep voice responses concise and helpful. If asked, disclose that you are an AI assistant. Provide accurate travel information and personalized suggestions.",
        "enabled_languages": ["en", "ru"],
        "suggested_tools": ["search_flights", "book_accommodation", "get_travel_info"],
    },
}


class AgentTemplateResponse(BaseModel):
    key: str
    name: str
    description: str
    system_prompt: str
    enabled_languages: list[str]
    suggested_tools: list[str]


class AgentFromTemplateIn(BaseModel):
    name: str | None = Field(None, min_length=1)
    description: str | None = None
    system_prompt: str | None = None
    enabled_languages: list[str] | None = None


class AgentFromTemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    enabled_languages: list[str]
    status: str
    template_key: str


@router.get("/agent-templates")
async def list_agent_templates(
    ctx: AuthContext = Depends(require("member")),
) -> list[AgentTemplateResponse]:
    """List all available agent templates."""
    return [AgentTemplateResponse(**template) for template in AGENT_TEMPLATES.values()]


@router.get("/agent-templates/{key}")
async def get_agent_template(
    key: str, ctx: AuthContext = Depends(require("member"))
) -> AgentTemplateResponse:
    """Get a specific agent template by key."""
    if key not in AGENT_TEMPLATES:
        raise HTTPException(
            status_code=404,
            detail="agent_template_not_found",
        )
    template = AGENT_TEMPLATES[key]
    return AgentTemplateResponse(**template)


@router.post("/agent-templates/{key}/agents", status_code=201)
async def create_agent_from_template(
    key: str,
    overrides: AgentFromTemplateIn,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> AgentFromTemplateResponse:
    """Create an agent from a template with optional overrides."""
    if key not in AGENT_TEMPLATES:
        raise HTTPException(
            status_code=404,
            detail="agent_template_not_found",
        )

    template = AGENT_TEMPLATES[key]

    # Apply overrides
    name = overrides.name if overrides.name is not None else template["name"]
    description = (
        overrides.description
        if overrides.description is not None
        else template["description"]
    )
    system_prompt = (
        overrides.system_prompt
        if overrides.system_prompt is not None
        else template["system_prompt"]
    )
    enabled_languages = (
        overrides.enabled_languages
        if overrides.enabled_languages is not None
        else template["enabled_languages"]
    )

    # Create agent record
    agent = AgentRecord(
        name=name,
        description=description,
        system_prompt=system_prompt,
        enabled_languages=enabled_languages,
        status="enabled",
        organization_id=ctx.org_id,
    )

    db.add(agent)
    db.commit()
    db.refresh(agent)

    return AgentFromTemplateResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        system_prompt=agent.system_prompt,
        enabled_languages=agent.enabled_languages,
        status=agent.status,
        template_key=key,
    )

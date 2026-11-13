"""Opt-in public agent demos in the calls organisation.

PLATFORM_DEMO_AGENTS maps public keys to agent IDs ("key=id,key2=id2").
PLATFORM_DEMO_MAX_SECONDS defaults to 300; the session reaper enforces the limit
on its next pass, even if Runtime events keep arriving.
"""
import os
from datetime import timedelta
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from . import limits
from .calls import client_address, limit_create, limit_open
from .context import AuthContext, get_ctx
from .knowledge import agent_knowledge_text, search_agent_knowledge
from .models import Agent, Conversation, Message, VoiceSession, utcnow
from .sessions import open_voice_session
from .tenancy import get_scoped_db, scoped

router = APIRouter(tags=["demo"])
_chat_bucket = limits.TokenBucket()
CHAT_INSTRUCTION = (
    "This is a text chat, not a voice call. Reply in plain text, up to 4 short sentences. "
    "If the user needs a person, say they can ask to be connected to a staff member. "
    "Append [HANDOFF] when the user asks for a person or the answer is not in the knowledge."
)


def configured_agents() -> dict[str, str]:
    return {key.strip(): value.strip() for item in os.getenv("PLATFORM_DEMO_AGENTS", "").split(",")
            for key, sep, value in [item.partition("=")] if sep and key.strip() and value.strip()}


def public_ctx(request: Request) -> AuthContext:
    if not configured_agents():
        raise HTTPException(404, "demo_disabled")
    return get_ctx(request)


def demo_agent(key: str, db: Session, ctx: AuthContext) -> Agent:
    agent = db.scalars(scoped(db, ctx, Agent).where(
        Agent.id == configured_agents().get(key, ""), Agent.system.is_(False), Agent.status == "enabled"
    )).first()
    if agent is None:
        raise HTTPException(404, "demo_agent_not_found")
    return agent


class DemoIn(BaseModel):
    name: str | None = Field(default=None, max_length=100)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


def limit_chat(request: Request) -> None:
    per_minute = max(0, int(os.getenv("PLATFORM_CHAT_PER_IP_MINUTE", "20")))
    if per_minute <= 0:
        return
    wait = _chat_bucket.take(client_address(request), per_minute)
    if wait > 0:
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": "Too many chat requests."},
            headers={"Retry-After": str(max(1, int(wait) + 1))},
        )


async def chat_completion(messages: list[dict[str, str]]) -> str:
    """Send a chat completion request to the configured OpenAI-compatible endpoint."""
    key = os.getenv("PLATFORM_CHAT_LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="chat_unavailable")
    base_url = os.getenv("PLATFORM_CHAT_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {"model": os.getenv("PLATFORM_CHAT_LLM_MODEL", "gemini-3.5-flash-lite"), "messages": messages,
               "temperature": 0.3, "max_tokens": 300}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        raise HTTPException(status_code=502, detail="chat_failed") from None


@router.post("/demo/agents/{key}/chat")
async def chat(key: str, payload: ChatIn, ctx: AuthContext = Depends(public_ctx),
               _limit: None = Depends(limit_chat), db: Session = Depends(get_scoped_db)) -> dict[str, Any]:
    agent = demo_agent(key, db, ctx)
    if payload.conversation_id:
        conversation = db.scalars(scoped(db, ctx, Conversation).where(
            Conversation.id == payload.conversation_id,
            Conversation.agent_id == agent.id,
            Conversation.channel == "chat",
            Conversation.started_at >= utcnow() - timedelta(minutes=60),
        )).first()
        if conversation is None:
            raise HTTPException(404, "conversation_not_found")
    else:
        conversation = Conversation(agent_id=agent.id, organization_id=ctx.org_id,
                                    metadata_json={"channel": "chat"}, channel="chat")
        db.add(conversation)
        db.flush()

    user_count = db.query(func.count(Message.id)).filter(
        Message.conversation_id == conversation.id, Message.role == "user"
    ).scalar() or 0
    if user_count >= 30:
        raise HTTPException(429, "chat_limit")

    db.add(Message(conversation_id=conversation.id, role="user", content=payload.message,
                   organization_id=ctx.org_id))
    db.flush()
    max_chars = max(0, int(os.getenv("PLATFORM_INLINE_KNOWLEDGE_MAX_CHARS", "60000")))
    knowledge = agent_knowledge_text(db, ctx.org_id, agent.id)
    if knowledge and len(knowledge) > max_chars:
        snippets = search_agent_knowledge(db, ctx.org_id, agent.id, query=payload.message, limit=5)
        knowledge = "\n\n".join(f"### {item['title']}\n{item['snippet']}" for item in snippets)
    system_parts = [agent.system_prompt or "", CHAT_INSTRUCTION]
    if knowledge:
        system_parts.insert(1, f"Knowledge:\n{knowledge}")
    history = db.query(Message).filter(Message.conversation_id == conversation.id).order_by(
        Message.created_at.desc(), Message.id.desc()
    ).limit(10).all()
    history.reverse()
    messages = [{"role": "system", "content": "\n\n".join(p for p in system_parts if p)}]
    messages.extend({"role": row.role, "content": row.content} for row in history)
    try:
        answer = await chat_completion(messages)
    except HTTPException:
        db.rollback()
        raise
    handoff_suggested = "[HANDOFF]" in answer
    reply = answer.replace("[HANDOFF]", "").strip()
    db.add(Message(conversation_id=conversation.id, role="assistant", content=reply, organization_id=ctx.org_id))
    db.commit()
    return {"conversation_id": conversation.id, "reply": reply, "handoff_suggested": handoff_suggested}


@router.get("/demo/agents/{key}")
async def info(key: str, ctx: AuthContext = Depends(public_ctx),
               _limit: None = Depends(limit_open), db: Session = Depends(get_scoped_db)) -> dict[str, Any]:
    agent = demo_agent(key, db, ctx)
    return {"key": key, "name": agent.name, "description": agent.description,
            "enabled_languages": agent.enabled_languages}


@router.post("/demo/agents/{key}/sessions", status_code=201)
async def create(key: str, payload: DemoIn, ctx: AuthContext = Depends(public_ctx),
                 _limit: None = Depends(limit_create), db: Session = Depends(get_scoped_db)) -> dict[str, Any]:
    agent = demo_agent(key, db, ctx)
    result = await open_voice_session(db, ctx, agent_id=agent.id)
    session = db.get(VoiceSession, str(result["session_id"]))
    session.runtime_contract = {**(session.runtime_contract or {}), "demo": True}
    conversation = db.get(Conversation, session.conversation_id)
    conversation.metadata_json = {**(conversation.metadata_json or {}), "requester_name": payload.name}
    db.commit()
    return result

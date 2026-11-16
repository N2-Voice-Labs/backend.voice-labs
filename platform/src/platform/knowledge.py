"""Knowledge API."""

import re
import os
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from .context import AuthContext, require
from .db import get_db
from .models import Agent as AgentRecord
from .models import AgentKnowledgeDocument, KnowledgeDocument
from .runtime_auth import require_runtime_token
from .tenancy import bypass, get_owned, get_public_agent, get_scoped_db, scoped
from .validation import reject_secret_configuration

router = APIRouter(tags=["knowledge"])
MAX_CONTENT_CHARS = 1_000_000


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    source: str = Field(default="", max_length=1024)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class DocumentPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    source: str | None = Field(default=None, max_length=1024)
    content: str | None = Field(default=None, min_length=1, max_length=MAX_CONTENT_CHARS)
    metadata: dict[str, Any] | None = None
    enabled: bool | None = None

    @field_validator("*", mode="before")
    @classmethod
    def reject_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("field cannot be null")
        return value

    @field_validator("metadata")
    @classmethod
    def reject_secret_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        return reject_secret_configuration(value)


class DocumentSummary(BaseModel):
    id: UUID
    title: str
    source: str
    content_length: int
    metadata: dict[str, Any]
    version: int
    enabled: bool
    created_at: datetime
    updated_at: datetime


class Document(DocumentSummary):
    content: str


class AgentKnowledgeIn(BaseModel):
    document_ids: list[UUID]


def summary_dict(record: KnowledgeDocument) -> dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "source": record.source,
        "content_length": len(record.content),
        "metadata": record.metadata_json,
        "version": record.version,
        "enabled": record.enabled,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def document_dict(record: KnowledgeDocument) -> dict[str, Any]:
    return {**summary_dict(record), "content": record.content}


def get_document_or_404(
    document_id: UUID, db: Session, ctx: AuthContext
) -> KnowledgeDocument:
    return get_owned(
        db,
        ctx,
        KnowledgeDocument,
        str(document_id),
        not_found_detail="knowledge_document_not_found",
    )


@router.post(
    "/knowledge/documents", response_model=Document, status_code=status.HTTP_201_CREATED
)
async def create_document(
    payload: DocumentIn,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> dict[str, Any]:
    data = payload.model_dump()
    record = KnowledgeDocument(metadata_json=data.pop("metadata"), version=1, **data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return document_dict(record)


@router.get("/knowledge/documents", response_model=list[DocumentSummary])
async def list_documents(
    enabled: bool | None = None,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> list[dict[str, Any]]:
    query = db.query(KnowledgeDocument)
    if enabled is not None:
        query = query.filter(KnowledgeDocument.enabled == enabled)
    return [summary_dict(r) for r in query.order_by(KnowledgeDocument.created_at).all()]


@router.get("/knowledge/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> dict[str, Any]:
    return document_dict(get_document_or_404(document_id, db, ctx))


@router.patch("/knowledge/documents/{document_id}", response_model=Document)
async def patch_document(
    document_id: UUID,
    payload: DocumentPatch,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> dict[str, Any]:
    record = get_document_or_404(document_id, db, ctx)
    for key, value in payload.model_dump(exclude_unset=True).items():
        if key == "content" and value != record.content:
            record.version += 1
        setattr(record, "metadata_json" if key == "metadata" else key, value)
    db.commit()
    db.refresh(record)
    return document_dict(record)


@router.delete("/knowledge/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> Response:
    get_document_or_404(document_id, db, ctx).enabled = False
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_agent_or_404(agent_id: UUID, db: Session, ctx: AuthContext) -> AgentRecord:
    return get_public_agent(db, ctx, agent_id)


@router.put("/agents/{agent_id}/knowledge", response_model=list[DocumentSummary])
async def set_agent_knowledge(
    agent_id: UUID,
    payload: AgentKnowledgeIn,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> list[dict[str, Any]]:
    get_agent_or_404(agent_id, db, ctx)
    ids = list(dict.fromkeys(str(i) for i in payload.document_ids))
    docs: list[KnowledgeDocument] = []
    if ids:
        # Org-scoped: another organisation's document looks exactly like a missing one.
        docs = list(
            db.scalars(
                scoped(db, ctx, KnowledgeDocument).where(KnowledgeDocument.id.in_(ids))
            )
        )
    if len(docs) != len(ids) or not all(doc.enabled for doc in docs):
        raise HTTPException(status_code=422, detail="invalid_knowledge_document_id")
    db.query(AgentKnowledgeDocument).filter(
        AgentKnowledgeDocument.agent_id == str(agent_id)
    ).delete()
    db.add_all(
        AgentKnowledgeDocument(agent_id=str(agent_id), document_id=doc.id)
        for doc in docs
    )
    db.commit()
    return [summary_dict(doc) for doc in sorted(docs, key=lambda d: d.title)]


@router.get("/agents/{agent_id}/knowledge", response_model=list[DocumentSummary])
async def list_agent_knowledge(
    agent_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("member")),
) -> list[dict[str, Any]]:
    get_agent_or_404(agent_id, db, ctx)
    docs = (
        db.query(KnowledgeDocument)
        .join(
            AgentKnowledgeDocument,
            AgentKnowledgeDocument.document_id == KnowledgeDocument.id,
        )
        .filter(AgentKnowledgeDocument.agent_id == str(agent_id))
        .order_by(KnowledgeDocument.title)
        .all()
    )
    return [summary_dict(doc) for doc in docs]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    document_id: UUID
    title: str
    source: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


def _tokenize(text: str) -> list[str]:
    """Tokenize text using unicode-aware regex, filtering short tokens."""
    tokens = re.findall(r"\w+", text.lower())
    result = [t for t in tokens if len(t) >= 2]
    for run in re.findall(r"[\uac00-\ud7a3]+", text.lower()):
        result.extend(run[i : i + 2] for i in range(len(run) - 1))
    return result


def agent_knowledge_text(db: Session, org_id: str, agent_id: str) -> str | None:
    """Return enabled linked source material in stable title order."""
    docs = (
        db.query(KnowledgeDocument)
        .join(AgentKnowledgeDocument, AgentKnowledgeDocument.document_id == KnowledgeDocument.id)
        .filter(
            AgentKnowledgeDocument.agent_id == str(agent_id),
            KnowledgeDocument.organization_id == str(org_id),
            KnowledgeDocument.enabled.is_(True),
        )
        .order_by(KnowledgeDocument.title, KnowledgeDocument.id)
        .all()
    )
    if not docs:
        return None
    return "\n\n".join(f"### {doc.title}\n{doc.content}" for doc in docs)


def _split_into_chunks(content: str, chunk_size: int = 800) -> list[str]:
    """Split content into chunks of approximately chunk_size characters.

    Attempts to break at paragraph boundaries (double newline), then sentence
    boundaries (period, question mark, exclamation mark), and finally word
    boundaries if necessary.
    """
    if len(content) <= chunk_size:
        return [content]

    chunks = []
    # First, split by paragraphs (double newline)
    paragraphs = content.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if not para.strip():
            continue

        # If adding this paragraph would exceed chunk_size, finalize current chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        # If the paragraph itself is larger than chunk_size, split by sentences
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [content]


def _score_chunk(
    chunk: str, query_tokens: list[str], title: str, title_tokens: list[str]
) -> float:
    """Score a chunk based on query token occurrences and title matches."""
    chunk_tokens = _tokenize(chunk)
    score = 0.0

    # Count occurrences of each query token in the chunk
    for query_token in query_tokens:
        score += chunk_tokens.count(query_token)

    # Add bonus for distinct tokens that appear in the title
    distinct_title_matches = set(query_tokens) & set(title_tokens)
    score += 2.0 * len(distinct_title_matches)

    return score


def search_agent_knowledge(
    db: Session, org_id: str, agent_id: str, query: str, limit: int
) -> list[dict[str, Any]]:
    """Search enabled documents linked to an agent within the supplied organization."""
    docs = (
        db.query(KnowledgeDocument)
        .join(AgentKnowledgeDocument, AgentKnowledgeDocument.document_id == KnowledgeDocument.id)
        .filter(
            AgentKnowledgeDocument.agent_id == str(agent_id),
            KnowledgeDocument.enabled.is_(True),
            KnowledgeDocument.organization_id == str(org_id),
        )
        .order_by(KnowledgeDocument.title, KnowledgeDocument.id)
        .all()
    )
    query_tokens = _tokenize(query)
    scored_chunks: list[tuple[float, str, UUID, str, str]] = []
    for doc in docs:
        title_tokens = _tokenize(doc.title)
        for chunk in _split_into_chunks(doc.content):
            score = _score_chunk(chunk, query_tokens, doc.title, title_tokens)
            if score > 0:
                scored_chunks.append((score, chunk, doc.id, doc.title, doc.source))
    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [
        {"document_id": doc_id, "title": title, "source": source, "snippet": chunk[:800], "score": score}
        for score, chunk, doc_id, title, source in scored_chunks[:limit]
    ]


@router.post(
    "/runtime/agents/{agent_id}/knowledge/search",
    response_model=SearchResponse,
    dependencies=[Depends(require_runtime_token)],
)
async def search_agent_knowledge_route(
    agent_id: UUID,
    payload: SearchRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Search through an agent's assigned knowledge documents.

    Returns snippets from enabled documents assigned to the agent,
    ranked by keyword relevance.
    """
    # Runtime callbacks carry no organisation: resolve it from the agent, then require
    # documents to belong to that same organisation (defence against stale cross-org
    # links in agent_knowledge_documents).
    with bypass(db):
        agent = db.scalars(
            select(AgentRecord).where(AgentRecord.id == str(agent_id))
        ).first()
        if agent is None:
            raise HTTPException(status_code=404, detail="agent_not_found")

    return {"results": search_agent_knowledge(db, agent.organization_id, str(agent_id), payload.query, payload.limit)}


@router.post("/agents/{agent_id}/knowledge/reindex")
async def reindex_agent_knowledge(
    agent_id: UUID,
    db: Session = Depends(get_scoped_db),
    ctx: AuthContext = Depends(require("admin")),
) -> dict[str, int]:
    from .rag import ensure_index
    get_agent_or_404(agent_id, db, ctx)
    result = ensure_index(db, ctx.org_id, str(agent_id))
    db.commit()
    return result

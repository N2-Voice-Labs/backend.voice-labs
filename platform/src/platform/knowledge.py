"""Knowledge API."""

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from .db import get_db
from .models import Agent as AgentRecord
from .models import AgentKnowledgeDocument, KnowledgeDocument
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


def get_document_or_404(document_id: UUID, db: Session) -> KnowledgeDocument:
    record = db.get(KnowledgeDocument, str(document_id))
    if record is None:
        raise HTTPException(status_code=404, detail="knowledge_document_not_found")
    return record


@router.post(
    "/knowledge/documents", response_model=Document, status_code=status.HTTP_201_CREATED
)
async def create_document(
    payload: DocumentIn, db: Session = Depends(get_db)
) -> dict[str, Any]:
    data = payload.model_dump()
    record = KnowledgeDocument(metadata_json=data.pop("metadata"), version=1, **data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return document_dict(record)


@router.get("/knowledge/documents", response_model=list[DocumentSummary])
async def list_documents(
    enabled: bool | None = None, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    query = db.query(KnowledgeDocument)
    if enabled is not None:
        query = query.filter(KnowledgeDocument.enabled == enabled)
    return [summary_dict(r) for r in query.order_by(KnowledgeDocument.created_at).all()]


@router.get("/knowledge/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: UUID, db: Session = Depends(get_db)
) -> dict[str, Any]:
    return document_dict(get_document_or_404(document_id, db))


@router.patch("/knowledge/documents/{document_id}", response_model=Document)
async def patch_document(
    document_id: UUID, payload: DocumentPatch, db: Session = Depends(get_db)
) -> dict[str, Any]:
    record = get_document_or_404(document_id, db)
    for key, value in payload.model_dump(exclude_unset=True).items():
        if key == "content" and value != record.content:
            record.version += 1
        setattr(record, "metadata_json" if key == "metadata" else key, value)
    db.commit()
    db.refresh(record)
    return document_dict(record)


@router.delete("/knowledge/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: UUID, db: Session = Depends(get_db)) -> Response:
    get_document_or_404(document_id, db).enabled = False
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def get_agent_or_404(agent_id: UUID, db: Session) -> AgentRecord:
    agent = db.get(AgentRecord, str(agent_id))
    if agent is None:
        raise HTTPException(status_code=404, detail="agent_not_found")
    return agent


@router.put("/agents/{agent_id}/knowledge", response_model=list[DocumentSummary])
async def set_agent_knowledge(
    agent_id: UUID, payload: AgentKnowledgeIn, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    get_agent_or_404(agent_id, db)
    ids = list(dict.fromkeys(str(i) for i in payload.document_ids))
    docs = (
        db.query(KnowledgeDocument).filter(KnowledgeDocument.id.in_(ids)).all()
        if ids
        else []
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
    agent_id: UUID, db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    get_agent_or_404(agent_id, db)
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

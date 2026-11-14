"""Tenant-scoped hybrid retrieval without database-specific vector extensions."""
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import math
import os
import re

import httpx
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from .knowledge import _tokenize
from .models import AgentKnowledgeDocument, KnowledgeDocument, KnowledgeChunk

log = logging.getLogger(__name__)
CHUNKER_VERSION = "1"
_CACHE: OrderedDict = OrderedDict()


class EmbeddingUnavailable(Exception):
    """The configured embedding provider cannot serve this request."""


@dataclass(frozen=True)
class Chunk:
    heading: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document_id: str
    title: str
    url: str
    heading: str
    text: str
    score: float = 0.0


def chunk_document(title: str, content: str) -> list[Chunk]:
    lines = content.splitlines()
    while lines and (not lines[0].strip() or re.match(r"^(Source|Retrieved):", lines[0], re.I)):
        lines.pop(0)
    sections = []
    path = []
    body = []
    heading = ""
    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            if "\n".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            level = len(match[1])
            path = [(n, h) for n, h in path if n < level] + [(level, match[2])]
            heading = " > ".join(h for _, h in path)
            body = []
        else:
            body.append(line)
    if "\n".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    chunks = []
    for heading, text in sections:
        current = ""
        # Paragraphs preserve contiguous Markdown tables. Oversized prose gets a hard split.
        for paragraph in re.split(r"\n\s*\n", text):
            units = [paragraph] if "|" in paragraph or len(paragraph) <= 1200 else [paragraph[i:i+1200] for i in range(0, len(paragraph), 1050)]
            for unit in units:
                if current and len(current) + len(unit) + 2 > 1200:
                    chunks.append(Chunk(heading[:300], current))
                    overlap = current[-150:] if "|" not in current.split("\n\n")[-1] else ""
                    current = overlap
                current = (current + "\n\n" + unit).strip()
        if current:
            chunks.append(Chunk(heading[:300], current))
    return chunks


def embedding_model() -> str:
    return os.getenv("PLATFORM_EMBEDDING_MODEL", "gemini-embedding-001")


def _normalise(vector):
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v*v for v in values))
    if not norm or not math.isfinite(norm):
        raise ValueError("invalid embedding")
    return [v/norm for v in values]


def embed_texts(texts: list[str]) -> list[list[float]]:
    key = os.getenv("PLATFORM_CHAT_LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EmbeddingUnavailable("No embedding API key configured")
    base = os.getenv("PLATFORM_CHAT_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    result = []
    try:
        with httpx.Client(timeout=15.0) as client:
            for start in range(0, len(texts), 100):
                batch = texts[start:start+100]
                response = client.post(f"{base}/embeddings", headers={"Authorization": f"Bearer {key}"},
                                       json={"model": embedding_model(), "input": batch, "dimensions": 768})
                response.raise_for_status()
                rows = response.json()["data"]
                # Gemini omits index; in that case preserve provider input order.
                rows = [dict(row, index=row.get("index", i)) for i, row in enumerate(rows)]
                rows = sorted(rows, key=lambda row: row["index"])
                if [row["index"] for row in rows] != list(range(len(batch))):
                    raise ValueError("incomplete embedding response")
                vectors = [_normalise(row["embedding"]) for row in rows]
                if any(len(v) != 768 for v in vectors):
                    raise ValueError("invalid embedding dimensions")
                result.extend(vectors)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
        raise EmbeddingUnavailable(f"Embedding provider unavailable ({status})") from None
    return result


def _documents(db, org_id, agent_id):
    return list(db.scalars(select(KnowledgeDocument).join(AgentKnowledgeDocument,
        AgentKnowledgeDocument.document_id == KnowledgeDocument.id).where(
        AgentKnowledgeDocument.agent_id == str(agent_id), KnowledgeDocument.organization_id == str(org_id),
        KnowledgeDocument.enabled.is_(True)).order_by(KnowledgeDocument.id)))


def _hash(doc):
    return hashlib.sha256(json.dumps([doc.content, doc.title, CHUNKER_VERSION, embedding_model()], ensure_ascii=False).encode()).hexdigest()


def ensure_index(db: Session, org_id: str, agent_id: str) -> dict:
    docs = _documents(db, org_id, agent_id)
    counts = {"documents": len(docs), "chunks": 0, "embedded": 0}
    unavailable = False
    for doc in docs:
        digest = _hash(doc)
        state = list(db.execute(select(KnowledgeChunk.content_hash, KnowledgeChunk.embedding.is_not(None)).where(
            KnowledgeChunk.document_id == doc.id, KnowledgeChunk.organization_id == str(org_id))))
        stale = not state or any(digest != row[0] for row in state)
        missing = any(not row[1] for row in state)
        if not stale and not missing:
            counts["chunks"] += len(state)
            counts["embedded"] += len(state)
            continue
        rows = list(db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.document_id == doc.id,
            KnowledgeChunk.organization_id == str(org_id)).order_by(KnowledgeChunk.position)))
        parts = chunk_document(doc.title, doc.content) if stale else [Chunk(row.heading, row.text) for row in rows]
        if stale or any(row.embedding is None for row in rows):
            vectors = None
            if parts and not unavailable:
                try:
                    vectors = embed_texts([f"{doc.title} > {p.heading}\n{p.text}" for p in parts])
                except EmbeddingUnavailable as exc:
                    unavailable = True
                    log.warning("%s; using keyword retrieval", exc)
            if stale:
                db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc.id,
                    KnowledgeChunk.organization_id == str(org_id)))
                rows = [KnowledgeChunk(organization_id=str(org_id), document_id=doc.id, position=i,
                    heading=p.heading, text=p.text, content_hash=digest, embedding_model=embedding_model()) for i,p in enumerate(parts)]
                db.add_all(rows)
            if vectors is not None:
                for row, vector in zip(rows, vectors):
                    row.embedding = vector
        counts["chunks"] += len(rows)
        counts["embedded"] += sum(row.embedding is not None for row in rows)
    db.flush()
    return counts


def retrieve(db: Session, org_id: str, agent_id: str, query: str, k: int = 6) -> list[Hit]:
    ensure_index(db, org_id, agent_id)
    docs = _documents(db, org_id, agent_id)
    if not docs or k <= 0:
        return []
    # Include chunk identities/state to avoid stale cache entries after rollback/reindex.
    state = list(db.execute(select(KnowledgeChunk.id, KnowledgeChunk.content_hash,
        KnowledgeChunk.embedding.is_not(None)).where(KnowledgeChunk.organization_id == str(org_id),
        KnowledgeChunk.document_id.in_([d.id for d in docs])).order_by(KnowledgeChunk.id)))
    fingerprint = tuple((d.id, _hash(d), d.source) for d in docs), tuple(tuple(r) for r in state)
    key = (db.get_bind(), str(org_id), str(agent_id), fingerprint)
    cached = _CACHE.get(key)
    if cached is None:
        by_id = {d.id: d for d in docs}
        rows = list(db.scalars(select(KnowledgeChunk).where(KnowledgeChunk.organization_id == str(org_id),
            KnowledgeChunk.document_id.in_(by_id)).order_by(KnowledgeChunk.document_id, KnowledgeChunk.position)))
        cached = [(Hit(r.id, r.document_id, by_id[r.document_id].title, by_id[r.document_id].source,
            r.heading, r.text), r.embedding, set(_tokenize(r.text + " " + by_id[r.document_id].title + " " + r.heading))) for r in rows]
        _CACHE[key] = cached
        while len(_CACHE) > 32:
            _CACHE.popitem(last=False)
    tokens = set(_tokenize(query))
    keyword = sorted(((len(tokens & ts), i) for i, (_, _, ts) in enumerate(cached)), key=lambda x: (-x[0], x[1]))
    ranks = [[i for score, i in keyword if score > 0][:20]]
    if any(vector for _, vector, _ in cached):
        try:
            q = _normalise(embed_texts([query])[0])
            semantic = [(sum(a*b for a,b in zip(q, v)), i) for i, (_,v,_) in enumerate(cached) if v and len(v) == len(q)]
            ranks.append([i for _,i in sorted(semantic, key=lambda x: (-x[0], x[1]))[:20]])
        except EmbeddingUnavailable:
            log.warning("Query embeddings unavailable; using keyword retrieval")
    scores = {}
    for rank in ranks:
        for position, i in enumerate(rank, 1):
            scores[i] = scores.get(i, 0) + 1/(60+position)
    ordered = sorted(scores, key=lambda i: (-scores[i], i))
    return [replace(cached[i][0], score=scores[i]) for i in ordered[:k]]

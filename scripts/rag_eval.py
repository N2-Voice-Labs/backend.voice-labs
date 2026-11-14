#!/usr/bin/env python3
"""Evaluate real embedding retrieval in an isolated SQLite database (no LLM).

Run: uv run --directory evals --with sqlalchemy --with httpx python ../scripts/rag_eval.py
Credentials must already be exported in the environment; this script never reads .env.
"""
import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "platform"))
sys.path.insert(0, str(ROOT / "evals/src"))
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.platform.db import Base
from src.platform.models import Agent, Organization, KnowledgeDocument, AgentKnowledgeDocument
from src.platform.rag import ensure_index, retrieve
from evals.faq_eval import normalize_text


def main():
    items = yaml.safe_load((ROOT / "evals/datasets/inha_admissions_faq.yaml").read_text())
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        org = Organization(name="RAG eval", slug="rag-eval")
        db.add(org)
        db.flush()
        agent = Agent(name="Inha eval", organization_id=org.id)
        db.add(agent)
        db.flush()
        for path in sorted((ROOT / "agents/inha-admissions/knowledge").glob("*.md")):
            if path.name.startswith("_") or path.name.upper() == "README.MD":
                continue
            content = path.read_text()
            source = re.search(r"(?m)^Source:\s*(\S+)", content)
            doc = KnowledgeDocument(title=path.stem, content=content, source=source[1] if source else "", organization_id=org.id)
            db.add(doc)
            db.flush()
            db.add(AgentKnowledgeDocument(agent_id=agent.id, document_id=doc.id))
        db.flush()
        counts = ensure_index(db, org.id, agent.id)
        if counts["embedded"] != counts["chunks"]:
            raise SystemExit("Real embedding index unavailable; export provider credentials before running.")
        db.commit()
        # A hit at k: the top-k passages together contain every must_include fact (what the model needs to answer).
        # "file@6" also checks that the dataset's source_file is among the top 6. Unanswerable items are skipped.
        print("| Question | facts@1 | facts@3 | facts@6 | file@6 |")
        print("|---|---:|---:|---:|---:|")
        scored = [item for item in items if item.get("answerable", True) is not False and item.get("must_include")]
        totals = [0, 0, 0, 0]
        for item in scored:
            hits = retrieve(db, org.id, agent.id, item["question"])
            facts = [normalize_text(str(term)) for term in item["must_include"]]
            row = []
            for k in (1, 3, 6):
                context = normalize_text("\n".join(h.text for h in hits[:k]))
                row.append(all(f in context for f in facts))
            titles = {h.title for h in hits[:6]}
            row.append(Path(str(item.get("source_file", ""))).stem in titles)
            totals = [a + int(b) for a, b in zip(totals, row)]
            print(f"| {item['id']} | " + " | ".join(str(int(s)) for s in row) + " |")
        n = len(scored)
        print("| Total | " + " | ".join(f"{s}/{n} ({s/n:.0%})" for s in totals) + " |")

if __name__ == "__main__":
    main()

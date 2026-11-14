import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from src.platform import rag
from src.platform.db import Base
from src.platform.models import Agent, Organization, KnowledgeDocument, KnowledgeChunk, AgentKnowledgeDocument


@pytest.fixture
def corpus(monkeypatch):
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        org = Organization(name='test', slug='rag-test')
        db.add(org)
        db.flush()
        agent = Agent(name='test', organization_id=org.id)
        db.add(agent)
        db.flush()
        doc = KnowledgeDocument(title='Tuition', content='# Fees\n\n등록금 tuition costs 100 won.', organization_id=org.id)
        db.add(doc)
        db.flush()
        db.add(AgentKnowledgeDocument(agent_id=agent.id, document_id=doc.id))
        db.commit()
        monkeypatch.setattr(rag, 'embed_texts', lambda texts: [[1.0, 0.0] for _ in texts])
        yield db, org, agent, doc
    rag._CACHE.clear()


def test_chunker():
    chunks = rag.chunk_document('Guide', 'Source: https://example.org\nRetrieved: today\n\n# Parent\n## Child\n\n' + ('paragraph ' * 55 + '\n\n') * 5)
    assert len(chunks) > 1
    assert all(c.heading == 'Parent > Child' for c in chunks)
    assert all('Source:' not in c.text for c in chunks)
    table = '| a | b |\n' * 160
    assert any(table.strip() in c.text for c in rag.chunk_document('Guide', '# Table\n' + table))


def test_retrieve_edit_and_disabled(corpus, monkeypatch):
    db, org, agent, doc = corpus
    calls = []
    def embed(texts):
        calls.extend(texts)
        return [[1., 0.] for _ in texts]
    monkeypatch.setattr(rag, 'embed_texts', embed)
    hits = rag.retrieve(db, org.id, agent.id, '등록금')
    assert hits[0].document_id == doc.id
    assert calls[0].startswith('Tuition > Fees\n')
    old_id = hits[0].chunk_id
    doc.content = '# Fees\n\nTuition now costs 200 won.'
    db.flush()
    hits = rag.retrieve(db, org.id, agent.id, 'tuition')
    assert hits[0].chunk_id != old_id and '200' in hits[0].text
    assert any('200' in text for text in calls)
    doc.enabled = False
    assert rag.retrieve(db, org.id, agent.id, 'tuition') == []


def test_keyword_fallback_and_recovery(corpus, monkeypatch, caplog):
    db, org, agent, doc = corpus
    def unavailable(texts):
        raise rag.EmbeddingUnavailable()
    monkeypatch.setattr(rag, 'embed_texts', unavailable)
    assert rag.retrieve(db, org.id, agent.id, '등록금')[0].document_id == doc.id
    assert 'keyword' in caplog.text
    assert db.scalars(select(KnowledgeChunk)).one().embedding is None
    monkeypatch.setattr(rag, 'embed_texts', lambda texts: [[1., 0.] for _ in texts])
    assert rag.ensure_index(db, org.id, agent.id)['embedded'] == 1


def test_citations_and_followup(monkeypatch):
    from fastapi.testclient import TestClient
    from src.platform.main import app
    from src.platform import demo as module
    from test_demo_handoffs import demo
    monkeypatch.setenv('PLATFORM_CHAT_KNOWLEDGE_MODE', 'rag')
    monkeypatch.setenv('PLATFORM_CHAT_PER_IP_MINUTE', '0')
    queries = []
    def retrieve(db, org, agent, query):
        queries.append(query)
        return [rag.Hit('1', 'doc', 'Title', 'https://example.org', 'Fees', 'x'*300),
                rag.Hit('2', 'doc2', 'Uncited', '', '', 'unused')]
    monkeypatch.setattr(rag, 'retrieve', retrieve)
    async def completion(messages):
        assert '[1] Title' in messages[0]['content']
        return 'Fee is 100 won. [1] Bogus [9] [HANDOFF]'
    monkeypatch.setattr(module, 'chat_completion', completion)
    with TestClient(app) as client:
        demo(client, monkeypatch)
        result = client.post('/demo/agents/sample/chat', json={'message': 'tuition?'}).json()
        assert '[9]' not in result['reply'] and '[1]' in result['reply']
        assert result['sources'] == [{'n': 1, 'title': 'Title', 'heading': 'Fees', 'url': 'https://example.org', 'snippet': 'x'*240}]
        client.post('/demo/agents/sample/chat', json={'message': 'and graduate?', 'conversation_id': result['conversation_id']})
        assert queries[-1] == 'tuition?\nand graduate?'


def test_reindex_requires_auth(monkeypatch):
    from fastapi.testclient import TestClient
    from src.platform.main import app
    monkeypatch.setenv('PLATFORM_API_KEYS', 'admin:admin-key')
    with TestClient(app) as client:
        response = client.post('/agents/00000000-0000-0000-0000-000000000001/knowledge/reindex')
        assert response.status_code in (401, 403)


def test_embed_batches_gemini_order_and_normalisation(monkeypatch):
    import httpx
    requests = []
    def respond(request):
        import json
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={'data': [dict({'embedding': [2.] + [0.]*767}, **({'index': i} if i else {})) for i, _ in enumerate(body['input'])]})
    real_client = httpx.Client
    monkeypatch.setenv('GEMINI_API_KEY', 'test')
    monkeypatch.delenv('PLATFORM_CHAT_LLM_API_KEY', raising=False)
    monkeypatch.setattr(rag.httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    vectors = rag.embed_texts(['text']*101)
    assert len(vectors) == 101 and vectors[0] == [1.] + [0.]*767
    assert [len(r['input']) for r in requests] == [100, 1]
    assert all(r['dimensions'] == 768 for r in requests)


def test_query_embedding_failure_and_unlink(corpus, monkeypatch, caplog):
    from sqlalchemy import delete
    db, org, agent, doc = corpus
    rag.ensure_index(db, org.id, agent.id)
    def unavailable(texts):
        raise rag.EmbeddingUnavailable()
    monkeypatch.setattr(rag, 'embed_texts', unavailable)
    assert rag.retrieve(db, org.id, agent.id, 'tuition')[0].document_id == doc.id
    assert 'Query embeddings unavailable' in caplog.text
    assert rag.retrieve(db, 'another-org', agent.id, 'tuition') == []
    db.execute(delete(AgentKnowledgeDocument).where(AgentKnowledgeDocument.agent_id == agent.id))
    assert rag.retrieve(db, org.id, agent.id, 'tuition') == []

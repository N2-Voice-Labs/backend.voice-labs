import os
from uuid import uuid4

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from fastapi.testclient import TestClient

from src.platform.main import app


def make_doc(client: TestClient, **overrides) -> dict:
    body = {
        "title": f"Doc {uuid4().hex[:8]}",
        "content": "Opening hours are 9 to 5.",
        "source": "https://example.test/faq",
        "metadata": {"lang": "en"},
        **overrides,
    }
    response = client.post("/knowledge/documents", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_search_requires_runtime_token(monkeypatch) -> None:
    """Test that search endpoint requires valid runtime token."""
    monkeypatch.delenv("PLATFORM_RUNTIME_TOKEN", raising=False)
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        # Missing env should return 503
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "test"},
            headers={"x-runtime-token": "any-token"},
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "runtime_auth_not_configured"


def test_search_agent_knowledge_helper_and_hangul_bigrams(monkeypatch) -> None:
    from src.platform.knowledge import _tokenize, search_agent_knowledge
    from src.platform.models import DEFAULT_ORGANIZATION_ID

    assert "입학" in _tokenize("입학서류는")
    assert "학서" in _tokenize("입학서류는")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Korean helper"}).json()
        doc = make_doc(client, title="Admissions", content="입학 서류는 여권 사본입니다.")
        client.put(f"/agents/{agent['id']}/knowledge", json={"document_ids": [doc["id"]]})
        with __import__("src.platform.db", fromlist=["SessionLocal"]).SessionLocal() as db:
            result = search_agent_knowledge(db, DEFAULT_ORGANIZATION_ID, agent["id"], "입학서류", 3)
        assert result and result[0]["document_id"] == doc["id"]
        monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "입학서류", "limit": 3},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 200
        assert response.json()["results"]


def test_search_rejects_invalid_token(monkeypatch) -> None:
    """Test that search endpoint rejects invalid tokens."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "correct-secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        # Missing token
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "test"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid_runtime_token"

        # Wrong token
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "test"},
            headers={"x-runtime-token": "wrong-secret"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid_runtime_token"


def test_search_requires_valid_agent(monkeypatch) -> None:
    """Test that search returns 404 for non-existent agent."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        response = client.post(
            f"/runtime/agents/{uuid4()}/knowledge/search",
            json={"query": "test"},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "agent_not_found"


def test_search_empty_results(monkeypatch) -> None:
    """Test that search returns empty results when no documents match."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()
        doc = make_doc(client, content="Python programming language")

        # Assign doc to agent
        client.put(
            f"/agents/{agent['id']}/knowledge",
            json={"document_ids": [doc["id"]]},
        )

        # Search for non-existent word
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "javascript"},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 200
        assert response.json() == {"results": []}


def test_search_english_ranking(monkeypatch) -> None:
    """Test that search correctly ranks English documents by relevance."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        # Create documents with varying relevance for "language"
        doc1 = make_doc(
            client,
            title="Python Language",
            content="Python is a programming language. The language is powerful.",
        )
        doc2 = make_doc(
            client,
            title="JavaScript Guide",
            content="JavaScript is a language used for web programming.",
        )
        doc3 = make_doc(
            client,
            title="Programming Concepts",
            content="Understanding language design patterns.",
        )

        # Assign all docs to agent
        client.put(
            f"/agents/{agent['id']}/knowledge",
            json={"document_ids": [doc1["id"], doc2["id"], doc3["id"]]},
        )

        # Search for "language"
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "language", "limit": 10},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 200
        results = response.json()["results"]

        # Results should be sorted by score descending
        assert len(results) >= 2
        assert results[0]["score"] >= results[1]["score"] if len(results) > 1 else True

        # Top result should be from doc1 (has "language" in title and multiple in content)
        assert results[0]["document_id"] == doc1["id"]


def test_search_russian_support(monkeypatch) -> None:
    """Test that search works with Russian Cyrillic text."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        # Create Russian documents
        doc_ru = make_doc(
            client,
            title="Руководство Python",
            content="Python - это мощный язык программирования. Изучите основы Python.",
        )
        doc_en = make_doc(
            client,
            title="Python Guide",
            content="Python is a powerful language.",
        )

        # Assign docs to agent
        client.put(
            f"/agents/{agent['id']}/knowledge",
            json={"document_ids": [doc_ru["id"], doc_en["id"]]},
        )

        # Search with Russian query
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "Python язык", "limit": 10},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 200
        results = response.json()["results"]

        # Should match the Russian document
        assert len(results) > 0
        doc_ids = {r["document_id"] for r in results}
        assert doc_ru["id"] in doc_ids


def test_search_excludes_unassigned_documents(monkeypatch) -> None:
    """Test that search only returns documents assigned to the agent."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent1 = client.post("/agents", json={"name": "Agent 1"}).json()
        agent2 = client.post("/agents", json={"name": "Agent 2"}).json()

        doc1 = make_doc(client, content="Python programming tutorial")
        doc2 = make_doc(client, content="Python advanced concepts")

        # Assign doc1 to agent1, doc2 to agent2
        client.put(
            f"/agents/{agent1['id']}/knowledge",
            json={"document_ids": [doc1["id"]]},
        )
        client.put(
            f"/agents/{agent2['id']}/knowledge",
            json={"document_ids": [doc2["id"]]},
        )

        # Search in agent1 for "Python"
        response = client.post(
            f"/runtime/agents/{agent1['id']}/knowledge/search",
            json={"query": "Python"},
            headers={"x-runtime-token": "secret"},
        )
        results = response.json()["results"]

        # Should only return doc1
        assert len(results) == 1
        assert results[0]["document_id"] == doc1["id"]


def test_search_excludes_disabled_documents(monkeypatch) -> None:
    """Test that search excludes disabled documents."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        doc1 = make_doc(client, content="Active Python document")
        doc2 = make_doc(client, content="Another Python document")

        # Assign both docs to agent
        client.put(
            f"/agents/{agent['id']}/knowledge",
            json={"document_ids": [doc1["id"], doc2["id"]]},
        )

        # Disable doc2
        client.patch(
            f"/knowledge/documents/{doc2['id']}",
            json={"enabled": False},
        )

        # Search should only return doc1
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "Python"},
            headers={"x-runtime-token": "secret"},
        )
        results = response.json()["results"]

        assert len(results) == 1
        assert results[0]["document_id"] == doc1["id"]


def test_search_limit_respected(monkeypatch) -> None:
    """Test that search respects the limit parameter."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        # Create 5 documents with varying relevance
        docs = [
            make_doc(client, content=f"Python programming concept {i} explained in detail")
            for i in range(5)
        ]

        # Assign all docs to agent
        client.put(
            f"/agents/{agent['id']}/knowledge",
            json={"document_ids": [d["id"] for d in docs]},
        )

        # Search with limit=2
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "Python", "limit": 2},
            headers={"x-runtime-token": "secret"},
        )
        results = response.json()["results"]

        # Should return at most 2 results
        assert len(results) <= 2

        # Search with default limit (3)
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "Python"},
            headers={"x-runtime-token": "secret"},
        )
        results = response.json()["results"]

        # Should return at most 3 results
        assert len(results) <= 3


def test_search_title_bonus(monkeypatch) -> None:
    """Test that matches in title get a 2x bonus in scoring."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        # doc_with_title: "Python" in title and content
        doc_with_title = make_doc(
            client,
            title="Python Basics",
            content="Introduction to Python programming.",
        )
        # doc_without_title: "Python" only in content
        doc_without_title = make_doc(
            client,
            title="Programming Guide",
            content="Python is a great programming language. Learn Python today.",
        )

        # Assign docs to agent
        client.put(
            f"/agents/{agent['id']}/knowledge",
            json={"document_ids": [doc_with_title["id"], doc_without_title["id"]]},
        )

        # Search for "Python"
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "Python", "limit": 10},
            headers={"x-runtime-token": "secret"},
        )
        results = response.json()["results"]

        # Both should match, but doc_with_title should score higher
        assert len(results) >= 1
        # First result should be from doc_with_title (has Python in title)
        if len(results) > 1:
            assert results[0]["document_id"] == doc_with_title["id"]


def test_search_query_validation(monkeypatch) -> None:
    """Test that search validates query length."""
    monkeypatch.setenv("PLATFORM_RUNTIME_TOKEN", "secret")
    with TestClient(app) as client:
        agent = client.post("/agents", json={"name": "Search agent"}).json()

        # Empty query should fail
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": ""},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 422

        # Query longer than 500 chars should fail
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "x" * 501},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 422

        # Invalid limit (0) should fail
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "test", "limit": 0},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 422

        # Invalid limit (>10) should fail
        response = client.post(
            f"/runtime/agents/{agent['id']}/knowledge/search",
            json={"query": "test", "limit": 11},
            headers={"x-runtime-token": "secret"},
        )
        assert response.status_code == 422

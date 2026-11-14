"""Public text chat demo endpoint tests."""
import pytest
from fastapi.testclient import TestClient

from src.platform import calls, demo as demo_module, limits as limits_module
from src.platform.db import SessionLocal
from src.platform.main import app
from src.platform.models import AgentKnowledgeDocument, Conversation, KnowledgeDocument, Message
from test_demo_handoffs import demo


@pytest.fixture(autouse=True)
def chat_config(monkeypatch):
    monkeypatch.delenv("PLATFORM_DEMO_AGENTS", raising=False)
    monkeypatch.delenv("PLATFORM_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PLATFORM_CHAT_LLM_API_KEY", raising=False)
    calls.reset_limits()
    demo_module._chat_bucket.reset()
    yield
    calls.reset_limits()
    demo_module._chat_bucket.reset()


def test_chat_disabled_unknown_and_no_key(monkeypatch):
    with TestClient(app) as client:
        assert client.post("/demo/agents/sample/chat", json={"message": "Hi"}).status_code == 404
        agent = demo(client, monkeypatch)
        assert client.post("/demo/agents/missing/chat", json={"message": "Hi"}).status_code == 404
        response = client.post("/demo/agents/sample/chat", json={"message": "Hi"})
        assert response.status_code == 503


def test_first_chat_persists_and_followup_includes_history_and_knowledge(monkeypatch):
    monkeypatch.setenv("PLATFORM_CHAT_KNOWLEDGE_MODE", "inline")
    captured = []

    async def completion(messages):
        captured.append(messages)
        return "A reply"

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(demo_module, "chat_completion", completion)
    with TestClient(app) as client:
        agent = demo(client, monkeypatch)
        with SessionLocal() as db:
            from src.platform.models import Agent
            record = db.query(Agent).filter_by(id=agent["id"]).one()
            record.system_prompt = "System base"
            document = KnowledgeDocument(title="Support guide", content="Support hours are weekdays.", organization_id=record.organization_id)
            db.add(document)
            db.flush()
            db.add(AgentKnowledgeDocument(agent_id=record.id, document_id=document.id))
            db.commit()
        first = client.post("/demo/agents/sample/chat", json={"message": "First turn"})
        assert first.status_code == 200, first.text
        result = first.json()
        assert result["reply"] == "A reply" and not result["handoff_suggested"]
        with SessionLocal() as db:
            conversation = db.get(Conversation, result["conversation_id"])
            assert conversation.channel == "chat" and conversation.metadata_json["channel"] == "chat"
            rows = db.query(Message).filter_by(conversation_id=conversation.id).order_by(Message.created_at, Message.id).all()
            assert [(row.role, row.content) for row in rows] == [("user", "First turn"), ("assistant", "A reply")]
        second = client.post("/demo/agents/sample/chat", json={"message": "Second turn", "conversation_id": result["conversation_id"]})
        assert second.status_code == 200, second.text
        sent = captured[-1]
        assert sent[0]["content"].startswith("System base")
        assert any("First turn" in item["content"] for item in sent)
        assert any(item["role"] == "assistant" and item["content"] == "A reply" for item in sent)
        assert any("Support hours are weekdays" in item["content"] for item in sent)


def test_wrong_conversation_marker_failure_and_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    async def marked(messages):
        return "Please contact staff. [HANDOFF]"
    monkeypatch.setattr(demo_module, "chat_completion", marked)
    with TestClient(app) as client:
        demo(client, monkeypatch)
        first = client.post("/demo/agents/sample/chat", json={"message": "Connect me"}).json()
        assert first["handoff_suggested"] is True
        assert "[HANDOFF]" not in first["reply"]
        assert client.post("/demo/agents/sample/chat", json={"message": "Hi", "conversation_id": "not-a-conversation"}).status_code == 404
        async def failed(messages):
            raise demo_module.HTTPException(status_code=502, detail="chat_failed")
        monkeypatch.setattr(demo_module, "chat_completion", failed)
        error = client.post("/demo/agents/sample/chat", json={"message": "Fail"})
        assert error.status_code == 502 and error.json()["detail"] == "chat_failed"
        monkeypatch.setattr(demo_module, "chat_completion", marked)
        with SessionLocal() as db:
            conversation_id = first["conversation_id"]
            for _ in range(30):
                db.add(Message(conversation_id=conversation_id, role="user", content="x"))
            db.commit()
        limited = client.post("/demo/agents/sample/chat", json={"message": "one more", "conversation_id": first["conversation_id"]})
        assert limited.status_code == 429 and limited.json()["detail"] == "chat_limit"


def test_chat_ip_rate_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PLATFORM_CHAT_PER_IP_MINUTE", "1")
    async def completion(messages):
        return "Reply"
    monkeypatch.setattr(demo_module, "chat_completion", completion)
    with TestClient(app) as client:
        demo(client, monkeypatch)
        assert client.post("/demo/agents/sample/chat", json={"message": "First"}).status_code == 200
        limited = client.post("/demo/agents/sample/chat", json={"message": "Second"})
        assert limited.status_code == 429


def test_chat_ip_rate_limit_refills_per_minute(monkeypatch):
    """20/minute means one new message every 3 seconds, not 20 per hour."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PLATFORM_CHAT_PER_IP_MINUTE", "20")
    async def completion(messages):
        return "Reply"
    monkeypatch.setattr(demo_module, "chat_completion", completion)
    clock = [1000.0]
    demo_module._chat_bucket.reset()
    monkeypatch.setattr(limits_module.time, "monotonic", lambda: clock[0])
    with TestClient(app) as client:
        demo(client, monkeypatch)
        for _ in range(20):
            assert client.post("/demo/agents/sample/chat", json={"message": "Hi"}).status_code == 200
        limited = client.post("/demo/agents/sample/chat", json={"message": "Hi"})
        assert limited.status_code == 429 and limited.headers["Retry-After"] in {"3", "4"}
        clock[0] += 3.1
        assert client.post("/demo/agents/sample/chat", json={"message": "Hi"}).status_code == 200

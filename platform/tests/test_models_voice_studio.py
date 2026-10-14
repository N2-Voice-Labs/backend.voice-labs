import os

import pytest

os.environ.setdefault("PLATFORM_DATABASE_URL", "sqlite://")
os.environ.setdefault("PLATFORM_AUTO_CREATE_SCHEMA", "1")

from sqlalchemy.exc import IntegrityError

from src.platform.db import Base, SessionLocal, engine
from src.platform.models import (
    Agent,
    AgentKnowledgeDocument,
    AgentTool,
    Campaign,
    CampaignTarget,
    Conversation,
    Deployment,
    KnowledgeDocument,
    ProviderConfig,
    Tool,
    Voice,
)
from src.platform.validation import (
    E164_PATTERN,
    contains_secret_key,
    reject_secret_configuration,
    validate_e164,
)


def test_voice_studio_models_round_trip() -> None:
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        tts = ProviderConfig(kind="tts", provider="fake")
        db.add(tts)
        db.flush()
        voice = Voice(
            name="Warm", source="clone", tts_provider_id=tts.id,
            languages=["en"], style={"warmth": 0.8},
            consent={"subject_name": "A", "confirmed": True},
        )
        db.add(voice)
        db.flush()
        agent = Agent(name="Agent", voice_profile_id=voice.id)
        tool = Tool(name="lookup", description="Look up")
        doc = KnowledgeDocument(title="FAQ", content="text", metadata_json={"k": 1})
        db.add_all([agent, tool, doc])
        db.flush()
        deployment = Deployment(
            agent_id=agent.id, name="Line", channel="sip", phone_number="+14155550123"
        )
        db.add(deployment)
        db.flush()
        campaign = Campaign(name="C", agent_id=agent.id, deployment_id=deployment.id)
        conversation = Conversation(
            agent_id=agent.id, deployment_id=deployment.id, channel="sip"
        )
        db.add_all([campaign, conversation])
        db.flush()
        db.add_all(
            [
                CampaignTarget(
                    campaign_id=campaign.id,
                    phone_number="+14155550124",
                    conversation_id=conversation.id,
                ),
                AgentTool(agent_id=agent.id, tool_id=tool.id),
                AgentKnowledgeDocument(agent_id=agent.id, document_id=doc.id),
            ]
        )
        db.commit()

        assert voice.status == "pending" and agent.voice_profile_id == voice.id
        assert deployment.status == "active" and campaign.status == "draft"
        assert (campaign.timezone, campaign.max_attempts) == ("UTC", 1)
        assert (conversation.channel, conversation.actions) == ("sip", [])
        assert (conversation.outcome, doc.version, tool.tool_type) == (None, 1, "http")

        db.add(CampaignTarget(campaign_id=campaign.id, phone_number="+14155550124"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()


def test_validation_helpers() -> None:
    assert contains_secret_key({"a": [{"Api_Key": "x"}]})
    assert not contains_secret_key({"voice": "x", "list": [1, {"b": 2}]})
    assert reject_secret_configuration({"model": "m"}) == {"model": "m"}
    with pytest.raises(ValueError, match="secret_reference"):
        reject_secret_configuration({"nested": {"password": "x"}})
    assert E164_PATTERN == r"^\+[1-9]\d{6,14}$"
    assert validate_e164("+14155550123") == "+14155550123"
    for bad in ("14155550123", "+0123456", "+12345", "+1415 555 0123"):
        with pytest.raises(ValueError):
            validate_e164(bad)

from uuid import uuid4

from src.runtime.events import RuntimeEvent
from src.runtime.session import RuntimeSession, SessionState


def test_event_has_version_and_identity() -> None:
    event = RuntimeEvent(
        event_type="TranscriptFinal", session_id=uuid4(), conversation_id=uuid4(), agent_id=uuid4()
    )
    assert event.schema_version == 1
    assert event.event_id


def test_failure_event_types_keep_schema_version_one() -> None:
    for event_type in ("SessionFailed", "RuntimeFailed"):
        event = RuntimeEvent(
            event_type=event_type,
            session_id=uuid4(),
            conversation_id=uuid4(),
            agent_id=uuid4(),
            payload={"reason": "RuntimeError"},
        )
        assert event.schema_version == 1
        assert event.payload["reason"] == "RuntimeError"


def test_interrupt_cancels_queued_output() -> None:
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    session.state = SessionState.SPEAKING
    session.output_queue.append(object())
    session.interrupt()
    assert session.cancel_requested is True
    assert session.output_queue == []
    assert session.state == SessionState.LISTENING

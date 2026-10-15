import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from src.runtime import tool_runtime
from src.runtime.session import RuntimeSession
from src.runtime.tool_runtime import build_tool_runtime


def make_session() -> tuple[RuntimeSession, list]:
    session = RuntimeSession(uuid4(), uuid4(), uuid4(), event_client=object())
    events: list = []
    session.enqueue_event = events.append  # type: ignore[method-assign]
    return session, events


def http_tool(**overrides) -> dict:
    tool = {
        "name": "lookup_order",
        "description": "Look up an order",
        "tool_type": "http",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        "configuration": {"url": "https://orders.test/lookup"},
    }
    tool.update(overrides)
    return tool


def run(runtime, name, arguments):
    return asyncio.run(runtime.invoke(name, arguments))


def types(events):
    return [(event.event_type, event.payload) for event in events]


def test_builds_tools_schema_with_builtins_and_http_tool() -> None:
    session, _ = make_session()
    runtime = build_tool_runtime(
        [http_tool(), {"name": "transfer_to_human", "description": "Escalate", "tool_type": "builtin"}],
        True,
        session,
    )

    assert runtime.names == ["record_outcome", "search_knowledge", "lookup_order", "transfer_to_human"]
    pytest.importorskip("pipecat")
    schema = runtime.tools_schema()
    by_name = {tool.name: tool for tool in schema.standard_tools}
    assert by_name["lookup_order"].required == ["order_id"]
    assert by_name["lookup_order"].properties == {"order_id": {"type": "string"}}
    assert by_name["record_outcome"].properties["outcome"]["enum"][0] == "resolved"


def test_without_knowledge_or_transfer_only_record_outcome_and_http_tools() -> None:
    session, _ = make_session()
    runtime = build_tool_runtime([http_tool()], False, session)
    assert runtime.names == ["record_outcome", "lookup_order"]


def test_registers_handlers_on_llm_service() -> None:
    session, _ = make_session()
    runtime = build_tool_runtime([http_tool()], False, session)
    registered: dict = {}

    class FakeLLM:
        def register_function(self, name, handler, **options):
            registered[name] = (handler, options)

    runtime.register(FakeLLM())

    assert set(registered) == {"record_outcome", "lookup_order"}
    assert registered["lookup_order"][1] == {"timeout_secs": 7.0}


def test_pipecat_handler_returns_result_through_callback() -> None:
    session, _ = make_session()
    runtime = build_tool_runtime([], False, session)
    results: list = []

    class Params:
        arguments = {"outcome": "resolved"}

        async def result_callback(self, result):
            results.append(result)

    asyncio.run(runtime.make_handler("record_outcome")(Params()))
    assert results == [{"status": "ok"}]


def test_http_post_success_sends_json_body_and_emits_tool_called() -> None:
    session, events = make_session()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "shipped"})

    runtime = build_tool_runtime([http_tool()], False, session, http_transport=httpx.MockTransport(handler))
    result = run(runtime, "lookup_order", {"order_id": "A1"})

    assert result == {"status": "ok", "data": {"status": "shipped"}}
    assert seen[0].method == "POST"
    assert json.loads(seen[0].content) == {"order_id": "A1"}
    assert types(events) == [
        ("ToolCalled", {"tool_name": "lookup_order", "arguments": {"order_id": "A1"}, "status": "ok"})
    ]


def test_http_get_sends_query_params_and_text_is_truncated() -> None:
    session, _ = make_session()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="x" * 10000)

    tool = http_tool(configuration={"url": "https://orders.test/lookup", "method": "get"})
    runtime = build_tool_runtime([tool], False, session, http_transport=httpx.MockTransport(handler))
    result = run(runtime, "lookup_order", {"order_id": "A 1"})

    assert seen[0].method == "GET"
    assert seen[0].url.params["order_id"] == "A 1"
    assert result["status"] == "ok"
    assert len(result["data"]) == 4000


def test_large_json_is_truncated_to_text() -> None:
    session, _ = make_session()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"rows": ["y" * 100] * 100}))
    runtime = build_tool_runtime([http_tool()], False, session, http_transport=transport)
    result = run(runtime, "lookup_order", {"order_id": "A1"})
    assert isinstance(result["data"], str) and len(result["data"]) == 4000


def test_http_error_status_returns_error_and_emits_error_event() -> None:
    session, events = make_session()
    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="down"))
    runtime = build_tool_runtime([http_tool()], False, session, http_transport=transport)
    result = run(runtime, "lookup_order", {"order_id": "A1"})

    assert result == {"status": "error", "error": "http_503"}
    assert events[0].payload["status"] == "error"
    assert events[0].payload["error"] == "http_503"


def test_http_timeout_returns_error_without_raising() -> None:
    session, events = make_session()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    runtime = build_tool_runtime([http_tool()], False, session, http_transport=httpx.MockTransport(handler))
    assert run(runtime, "lookup_order", {"order_id": "A1"}) == {"status": "error", "error": "timeout"}
    assert events[0].payload["error"] == "timeout"


def test_http_connection_failure_returns_error() -> None:
    session, _ = make_session()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    runtime = build_tool_runtime([http_tool()], False, session, http_transport=httpx.MockTransport(handler))
    assert run(runtime, "lookup_order", {"order_id": "A1"})["error"] == "request_failed"


def test_missing_required_argument_returns_error_without_calling() -> None:
    session, events = make_session()
    called: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200, json={})

    runtime = build_tool_runtime([http_tool()], False, session, http_transport=httpx.MockTransport(handler))
    result = run(runtime, "lookup_order", {})

    assert result["status"] == "error" and "order_id" in result["error"]
    assert called == []
    assert events[0].payload["status"] == "error"


def test_non_object_arguments_are_rejected_with_dict_event_payload() -> None:
    session, events = make_session()
    runtime = build_tool_runtime([http_tool()], False, session)
    result = run(runtime, "lookup_order", ["not", "an", "object"])
    assert result["status"] == "error"
    assert events[0].payload["arguments"] == {}


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"url": "ftp://orders.test"},
        {"url": "not a url"},
        {"url": "https://orders.test", "method": "DELETE"},
        {"url": "https://orders.test", "timeout_s": 30},
        {"url": "https://orders.test", "timeout_s": "5"},
        {"url": "https://orders.test", "timeout_s": 0},
    ],
)
def test_invalid_http_configuration_is_skipped(configuration, caplog) -> None:
    session, _ = make_session()
    runtime = build_tool_runtime([http_tool(configuration=configuration)], False, session)
    assert runtime.names == ["record_outcome"]
    assert "Skipping tool 'lookup_order'" in caplog.text


def test_invalid_tool_entries_never_crash() -> None:
    session, _ = make_session()
    runtime = build_tool_runtime(
        [None, {}, {"name": 5}, http_tool(input_schema="bad"), http_tool(description=""),
         http_tool(tool_type="soap"), {"name": "nope", "description": "d", "tool_type": "builtin"}],
        False,
        session,
    )
    assert runtime.names == ["record_outcome"]


def test_builtin_names_win_over_user_tools() -> None:
    session, _ = make_session()
    runtime = build_tool_runtime(
        [http_tool(name="record_outcome"), http_tool(name="search_knowledge"),
         http_tool(name="transfer_to_human"), http_tool(name="lookup_order"), http_tool(name="lookup_order")],
        True,
        session,
    )
    definitions = {definition.name: definition for definition in runtime.definitions}
    assert runtime.names == ["record_outcome", "search_knowledge", "lookup_order"]
    assert definitions["record_outcome"].builtin and definitions["search_knowledge"].builtin


def test_record_outcome_emits_outcome_and_tool_called() -> None:
    session, events = make_session()
    runtime = build_tool_runtime([], False, session)
    result = run(runtime, "record_outcome", {"outcome": "booked", "summary": "Booked Tuesday", "intent": "booking"})

    assert result == {"status": "ok"}
    assert types(events) == [
        ("OutcomeRecorded", {"outcome": "booked", "summary": "Booked Tuesday", "intent": "booking"}),
        ("ToolCalled", {"tool_name": "record_outcome",
                        "arguments": {"outcome": "booked", "summary": "Booked Tuesday", "intent": "booking"},
                        "status": "ok"}),
    ]


def test_record_outcome_omits_empty_optional_fields_and_rejects_unknown_outcome() -> None:
    session, events = make_session()
    runtime = build_tool_runtime([], False, session)
    run(runtime, "record_outcome", {"outcome": "resolved"})
    assert events[0].payload == {"outcome": "resolved"}

    events.clear()
    result = run(runtime, "record_outcome", {"outcome": "great"})
    assert result == {"status": "error", "error": "invalid_outcome"}
    assert [event.event_type for event in events] == ["ToolCalled"]
    assert events[0].payload["status"] == "error"


TRANSFER = {"name": "transfer_to_human", "description": "Escalate", "tool_type": "builtin"}


def test_transfer_emits_handoff_and_calls_hook(monkeypatch) -> None:
    session, events = make_session()
    calls: list = []

    async def hook(received_session, reason):
        calls.append((received_session, reason))

    monkeypatch.setattr(tool_runtime, "transfer_handler", hook)
    runtime = build_tool_runtime([TRANSFER], False, session)
    result = run(runtime, "transfer_to_human", {"reason": "wants a person"})

    assert result["status"] == "ok"
    assert calls == [(session, "wants a person")]
    assert types(events)[0] == ("HandoffRequested", {"reason": "wants a person", "target": None})
    assert events[1].event_type == "ToolCalled" and events[1].payload["status"] == "ok"


def test_transfer_hook_failure_is_an_error_result(monkeypatch) -> None:
    session, events = make_session()

    async def hook(received_session, reason):
        raise RuntimeError("sip down")

    monkeypatch.setattr(tool_runtime, "transfer_handler", hook)
    runtime = build_tool_runtime([TRANSFER], False, session)
    result = run(runtime, "transfer_to_human", {})

    assert result == {"status": "error", "error": "transfer_failed"}
    assert events[0].event_type == "HandoffRequested"
    assert events[1].payload["status"] == "error"


def test_transfer_without_hook_only_records_handoff() -> None:
    session, events = make_session()
    runtime = build_tool_runtime([TRANSFER], False, session)
    assert run(runtime, "transfer_to_human", {})["status"] == "ok"
    assert events[0].payload == {"reason": None, "target": None}


def test_transfer_not_registered_unless_assigned() -> None:
    session, _ = make_session()
    assert "transfer_to_human" not in build_tool_runtime([], False, session).names


def mock_platform(monkeypatch, handler) -> None:
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(tool_runtime.httpx, "AsyncClient", client)


def test_search_knowledge_calls_platform_with_runtime_token(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_URL", "http://platform.test/")
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "secret-token")
    session, events = make_session()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [{"title": "FAQ", "snippet": "Open 9-5"}]})

    mock_platform(monkeypatch, handler)
    runtime = build_tool_runtime([], True, session)
    result = run(runtime, "search_knowledge", {"query": "hours"})

    assert result == {"status": "ok", "data": {"results": [{"title": "FAQ", "snippet": "Open 9-5"}]}}
    assert str(seen[0].url) == f"http://platform.test/runtime/agents/{session.agent_id}/knowledge/search"
    assert seen[0].headers["x-runtime-token"] == "secret-token"
    assert json.loads(seen[0].content) == {"query": "hours", "limit": 3}
    assert events[0].event_type == "ToolCalled"
    assert events[0].payload == {"tool_name": "search_knowledge", "arguments": {"query": "hours"}, "status": "ok"}


def test_search_knowledge_platform_failure_is_error_result(monkeypatch) -> None:
    monkeypatch.setenv("RUNTIME_PLATFORM_TOKEN", "secret-token")
    session, events = make_session()
    mock_platform(monkeypatch, lambda request: httpx.Response(500))
    runtime = build_tool_runtime([], True, session)
    result = run(runtime, "search_knowledge", {"query": "hours"})
    assert result == {"status": "error", "error": "knowledge_unavailable"}
    assert events[0].payload["status"] == "error"


def test_search_knowledge_requires_query_and_token(monkeypatch) -> None:
    session, _ = make_session()
    runtime = build_tool_runtime([], True, session)
    monkeypatch.delenv("RUNTIME_PLATFORM_TOKEN", raising=False)
    assert run(runtime, "search_knowledge", {"query": "x"})["error"] == "knowledge_unavailable"
    assert run(runtime, "search_knowledge", {"query": "  "})["error"] == "query_required"


def test_event_queue_failure_does_not_break_tool_result() -> None:
    session, _ = make_session()

    def fail(event):
        raise RuntimeError("queue closed")

    session.enqueue_event = fail  # type: ignore[method-assign]
    runtime = build_tool_runtime([], False, session)
    assert run(runtime, "record_outcome", {"outcome": "other"}) == {"status": "ok"}


def test_sessions_without_event_client_emit_nothing() -> None:
    session = RuntimeSession(uuid4(), uuid4(), uuid4())
    runtime = build_tool_runtime([], False, session)
    assert run(runtime, "record_outcome", {"outcome": "other"}) == {"status": "ok"}

"""Run the agent's tools as native Pipecat LLM function calls.

The Platform session contract carries ``tools`` (assigned and enabled) and
``knowledge_enabled``. :func:`build_tool_runtime` turns them into function
definitions plus handlers. ``ToolRuntime.tools_schema()`` builds the Pipecat
``ToolsSchema`` and ``ToolRuntime.register(llm)`` registers the handlers with
``LLMService.register_function``. Pipecat is imported lazily; the invocation logic
(``ToolRuntime.invoke``) is plain asyncio and never raises into the pipeline.

Every invocation emits ``ToolCalled``. Built-ins also emit ``OutcomeRecorded`` or
``HandoffRequested`` through the session's existing event queue.
"""

import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from .events import RuntimeEventType

if TYPE_CHECKING:
    from .session import RuntimeSession

_logger = logging.getLogger(__name__)

OUTCOMES = (
    "resolved",
    "booked",
    "callback_requested",
    "no_answer",
    "declined",
    "escalated",
    "failed",
    "other",
)
MAX_RESULT_CHARS = 4000
DEFAULT_HTTP_TIMEOUT_S = 5.0
MAX_HTTP_TIMEOUT_S = 10.0
KNOWLEDGE_SEARCH_LIMIT = 3
KNOWLEDGE_TIMEOUT_S = 5.0
_NAME_MAX_LENGTH = 64
_TOOL_TIMEOUT_MARGIN_S = 2.0
_INTENT_MAX_LENGTH = 200
_SUMMARY_MAX_LENGTH = 2000

RECORD_OUTCOME = "record_outcome"
TRANSFER_TO_HUMAN = "transfer_to_human"
SEARCH_KNOWLEDGE = "search_knowledge"
BUILTIN_NAMES = frozenset({RECORD_OUTCOME, TRANSFER_TO_HUMAN, SEARCH_KNOWLEDGE})

INSTRUCTION_SUFFIX = (
    "When the conversation's goal is reached or the caller ends, call record_outcome."
)

# Wave C plugs SIP transfer in here: ``async def handler(session, reason) -> None``.
# ``session`` is the RuntimeSession; ``reason`` is the LLM-supplied string or None.
# A session attribute of the same name takes precedence over this module default.
TransferHandler = Callable[["RuntimeSession", "str | None"], Awaitable[object]]
transfer_handler: TransferHandler | None = None


def set_transfer_handler(handler: TransferHandler | None) -> None:
    global transfer_handler
    transfer_handler = handler


ToolFunction = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class ToolDefinition:
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    run: ToolFunction
    builtin: bool = False
    timeout_secs: float | None = None


def _error(reason: str) -> dict[str, Any]:
    return {"status": "error", "error": reason}


def _truncate_data(value: Any) -> Any:
    if isinstance(value, str):
        return value[:MAX_RESULT_CHARS]
    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)[:MAX_RESULT_CHARS]
    if len(encoded) <= MAX_RESULT_CHARS:
        return value
    return encoded[:MAX_RESULT_CHARS]


def _missing_required(
    arguments: Mapping[str, Any], required: list[str]
) -> list[str]:
    return [key for key in required if key not in arguments or arguments[key] is None]


class ToolRuntime:
    def __init__(
        self,
        session: "RuntimeSession | None",
        definitions: list[ToolDefinition],
    ) -> None:
        self._session = session
        self._definitions = {definition.name: definition for definition in definitions}

    @property
    def definitions(self) -> list[ToolDefinition]:
        return list(self._definitions.values())

    @property
    def names(self) -> list[str]:
        return list(self._definitions)

    def __bool__(self) -> bool:
        return bool(self._definitions)

    # -- events -----------------------------------------------------------------

    def emit(
        self, event_type: RuntimeEventType, payload: dict[str, object]
    ) -> None:
        """Queue a Platform event; telemetry failures never break a tool call."""

        session = self._session
        if session is None or not session.event_client:
            return
        try:
            session.enqueue_event(session.event(event_type, payload))
        except RuntimeError:
            _logger.warning("Dropped %s event for session %s", event_type, session.session_id)

    # -- invocation -------------------------------------------------------------

    async def invoke(self, name: str, arguments: Any) -> dict[str, Any]:
        """Run one tool and emit ``ToolCalled``. Never raises (except cancellation)."""

        definition = self._definitions.get(name)
        if not isinstance(arguments, Mapping):
            result = _error("arguments_must_be_object")
            arguments = {}
        else:
            arguments = dict(arguments)
            if definition is None:
                result = _error("unknown_tool")
            else:
                missing = _missing_required(arguments, definition.required)
                if missing:
                    result = _error(f"missing_required_arguments: {', '.join(missing)}")
                else:
                    try:
                        result = await definition.run(arguments)
                    except Exception:
                        _logger.exception("Tool %s failed unexpectedly", name)
                        result = _error("tool_failed")
        payload: dict[str, object] = {
            "tool_name": name,
            "arguments": arguments,
            "status": "ok" if result.get("status") == "ok" else "error",
        }
        if payload["status"] == "error":
            payload["error"] = str(result.get("error", "error"))[:200]
        self.emit("ToolCalled", payload)
        return result

    def make_handler(self, name: str) -> Callable[[Any], Awaitable[None]]:
        """Build a Pipecat function-call handler taking ``FunctionCallParams``."""

        async def handler(params: Any) -> None:
            result = await self.invoke(name, params.arguments)
            await params.result_callback(result)

        return handler

    # -- Pipecat ------------------------------------------------------------------

    def tools_schema(self) -> Any:
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        from pipecat.adapters.schemas.tools_schema import ToolsSchema

        return ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name=definition.name,
                    description=definition.description,
                    properties=definition.properties,
                    required=definition.required,
                )
                for definition in self._definitions.values()
            ]
        )

    def register(self, llm: Any) -> None:
        for definition in self._definitions.values():
            options: dict[str, Any] = {}
            if definition.timeout_secs is not None:
                options["timeout_secs"] = definition.timeout_secs
            llm.register_function(
                definition.name, self.make_handler(definition.name), **options
            )


# -- http tools ------------------------------------------------------------------


@dataclass
class _HttpConfig:
    url: str
    method: str
    timeout_s: float


def _parse_http_config(configuration: Any) -> _HttpConfig:
    if not isinstance(configuration, dict):
        raise ValueError("configuration must be an object")
    url = configuration.get("url")
    if not isinstance(url, str):
        raise ValueError("url is required")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an http(s) URL")
    method = configuration.get("method", "POST")
    if not isinstance(method, str) or method.upper() not in {"GET", "POST"}:
        raise ValueError("method must be GET or POST")
    timeout = configuration.get("timeout_s", DEFAULT_HTTP_TIMEOUT_S)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= MAX_HTTP_TIMEOUT_S
    ):
        raise ValueError(f"timeout_s must be a number in (0, {MAX_HTTP_TIMEOUT_S:g}]")
    return _HttpConfig(url=url, method=method.upper(), timeout_s=float(timeout))


def _schema_parts(input_schema: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(input_schema, dict):
        raise ValueError("input_schema must be an object")
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    if not isinstance(properties, dict):
        raise ValueError("input_schema properties must be an object")
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise ValueError("input_schema required must be a list of strings")
    return properties, required


def _http_tool(
    config: _HttpConfig,
    transport: httpx.AsyncBaseTransport | None,
) -> ToolFunction:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=config.timeout_s, transport=transport, follow_redirects=False
            ) as client:
                if config.method == "GET":
                    query = {
                        key: value
                        if isinstance(value, (str, int, float, bool))
                        else json.dumps(value)
                        for key, value in arguments.items()
                    }
                    response = await client.get(config.url, params=query)
                else:
                    response = await client.post(config.url, json=arguments)
        except httpx.TimeoutException:
            return _error("timeout")
        except httpx.HTTPError:
            return _error("request_failed")
        if response.status_code >= 400:
            return _error(f"http_{response.status_code}")
        try:
            data: Any = response.json()
        except ValueError:
            data = response.text
        return {"status": "ok", "data": _truncate_data(data)}

    return run


# -- built-ins -------------------------------------------------------------------


def _optional_string(value: Any, limit: int) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()[:limit]
    return None


def _builtin_record_outcome(runtime: "ToolRuntime") -> ToolDefinition:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        outcome = arguments.get("outcome")
        if outcome not in OUTCOMES:
            return _error("invalid_outcome")
        payload: dict[str, object] = {"outcome": outcome}
        summary = _optional_string(arguments.get("summary"), _SUMMARY_MAX_LENGTH)
        intent = _optional_string(arguments.get("intent"), _INTENT_MAX_LENGTH)
        if summary:
            payload["summary"] = summary
        if intent:
            payload["intent"] = intent
        runtime.emit("OutcomeRecorded", payload)
        return {"status": "ok"}

    return ToolDefinition(
        name=RECORD_OUTCOME,
        description=(
            "Record how the conversation ended. Call once when the caller's goal is "
            "reached or the caller is done."
        ),
        properties={
            "outcome": {
                "type": "string",
                "enum": list(OUTCOMES),
                "description": "Final result of the conversation.",
            },
            "summary": {"type": "string", "description": "One or two sentence summary."},
            "intent": {"type": "string", "description": "What the caller wanted."},
        },
        required=["outcome"],
        run=run,
        builtin=True,
    )


def _builtin_transfer_to_human(
    runtime: "ToolRuntime", session: "RuntimeSession | None", description: str | None
) -> ToolDefinition:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        reason = _optional_string(arguments.get("reason"), 500)
        runtime.emit("HandoffRequested", {"reason": reason, "target": None})
        handler = getattr(session, "transfer_handler", None) or transfer_handler
        if handler is not None:
            try:
                await handler(session, reason)
            except Exception:
                _logger.exception("transfer_handler failed")
                return _error("transfer_failed")
        return {"status": "ok", "data": "Handoff to a human has been requested."}

    return ToolDefinition(
        name=TRANSFER_TO_HUMAN,
        description=description
        or "Hand the caller over to a human agent when they ask for one or you cannot help.",
        properties={
            "reason": {"type": "string", "description": "Why the caller needs a human."}
        },
        required=[],
        run=run,
        builtin=True,
    )


def _builtin_search_knowledge(agent_id: Any) -> ToolDefinition:
    async def run(arguments: dict[str, Any]) -> dict[str, Any]:
        query = _optional_string(arguments.get("query"), 500)
        if not query:
            return _error("query_required")
        token = os.getenv("RUNTIME_PLATFORM_TOKEN")
        if not token:
            return _error("knowledge_unavailable")
        base_url = os.getenv("RUNTIME_PLATFORM_URL", "http://localhost:8000").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=KNOWLEDGE_TIMEOUT_S) as client:
                response = await client.post(
                    f"{base_url}/runtime/agents/{agent_id}/knowledge/search",
                    json={"query": query, "limit": KNOWLEDGE_SEARCH_LIMIT},
                    headers={"X-Runtime-Token": token},
                )
                response.raise_for_status()
                results = response.json().get("results", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            return _error("knowledge_unavailable")
        if not isinstance(results, list):
            return _error("knowledge_unavailable")
        return {"status": "ok", "data": {"results": _truncate_data(results)}}

    return ToolDefinition(
        name=SEARCH_KNOWLEDGE,
        description=(
            "Search the agent's knowledge base for facts to answer the caller. "
            "Use before answering factual questions about the business."
        ),
        properties={"query": {"type": "string", "description": "What to look up."}},
        required=["query"],
        run=run,
        builtin=True,
        timeout_secs=KNOWLEDGE_TIMEOUT_S + _TOOL_TIMEOUT_MARGIN_S,
    )


# -- assembly --------------------------------------------------------------------


def build_tool_runtime(
    tools: list[dict[str, Any]] | None,
    knowledge_enabled: bool,
    session: "RuntimeSession | None",
    *,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> ToolRuntime:
    """Build the tool set for one session. Invalid tools are skipped, never fatal."""

    runtime = ToolRuntime(session, [])
    definitions: dict[str, ToolDefinition] = {}

    def add(definition: ToolDefinition) -> None:
        definitions[definition.name] = definition

    add(_builtin_record_outcome(runtime))
    if knowledge_enabled and session is not None:
        add(_builtin_search_knowledge(session.agent_id))

    for tool in tools or []:
        name = tool.get("name") if isinstance(tool, dict) else None
        try:
            if not isinstance(name, str) or not name or len(name) > _NAME_MAX_LENGTH:
                raise ValueError("invalid tool name")
            tool_type = tool.get("tool_type", "http")
            if tool_type == "builtin":
                if name == TRANSFER_TO_HUMAN and name not in definitions:
                    description = tool.get("description")
                    add(
                        _builtin_transfer_to_human(
                            runtime,
                            session,
                            description if isinstance(description, str) else None,
                        )
                    )
                elif name in BUILTIN_NAMES:
                    continue  # always-on built-in, nothing to add
                else:
                    raise ValueError("unknown builtin tool")
                continue
            if tool_type != "http":
                raise ValueError(f"unsupported tool_type {tool_type!r}")
            if name in BUILTIN_NAMES or name in definitions:
                raise ValueError("tool name collides with an existing tool")
            description = tool.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("description is required")
            config = _parse_http_config(tool.get("configuration"))
            properties, required = _schema_parts(tool.get("input_schema", {}))
            add(
                ToolDefinition(
                    name=name,
                    description=description,
                    properties=properties,
                    required=required,
                    run=_http_tool(config, http_transport),
                    timeout_secs=config.timeout_s + _TOOL_TIMEOUT_MARGIN_S,
                )
            )
        except ValueError as exc:
            _logger.warning("Skipping tool %r: %s", name, exc)
    runtime._definitions = definitions
    return runtime

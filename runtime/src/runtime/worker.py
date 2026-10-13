"""Runtime process entrypoint for one configured LiveKit room."""

import asyncio
import os
from uuid import UUID, uuid4

from .pipeline import PipelineConfig
from .session import RuntimeSession
from .events_client import PlatformEventClient


def main() -> None:
    asyncio.run(run())


async def run() -> None:
    config = PipelineConfig.from_environment()
    session_id = UUID(os.getenv("RUNTIME_SESSION_ID", str(uuid4())))
    conversation_id = UUID(os.getenv("RUNTIME_CONVERSATION_ID", str(uuid4())))
    agent_id = UUID(os.getenv("RUNTIME_AGENT_ID", str(uuid4())))
    event_client = None
    if os.getenv("RUNTIME_PLATFORM_TOKEN"):
        event_client = PlatformEventClient(os.getenv("RUNTIME_PLATFORM_URL", "http://localhost:8000"), os.environ["RUNTIME_PLATFORM_TOKEN"])
    try:
        await RuntimeSession(session_id, conversation_id, agent_id, event_client).run_pipeline(config)
    finally:
        if event_client:
            await event_client.aclose()


if __name__ == "__main__":
    main()

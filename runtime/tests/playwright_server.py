"""Run the real Runtime API with inference services replaced for browser acceptance."""

import asyncio
import importlib
import math
import os
from array import array

import uvicorn

from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


assistant_audio_started = asyncio.Event()


class BrowserAcceptanceSTT(FrameProcessor):
    """Simulate initial recognition and optionally a deterministic follow-up turn."""

    class Settings:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    def __init__(self, *args, **kwargs) -> None:
        self.settings = kwargs.get("settings")
        self.transcription_emitted = False
        self.interruption_emitted = False
        super().__init__(*args, **kwargs)

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            return
        if not self.transcription_emitted:
            self.transcription_emitted = True
            await self.push_frame(
                TranscriptionFrame(
                    "synthetic ingress transcript",
                    "participant-1",
                    "integration-test",
                    finalized=True,
                ),
                direction,
            )
            return
        if (
            not os.getenv("VOICE_TEST_AUDIO_FILE")
            and assistant_audio_started.is_set()
            and not self.interruption_emitted
        ):
            self.interruption_emitted = True
            await asyncio.sleep(0.35)
            await self.push_frame(
                TranscriptionFrame(
                    "synthetic barge-in transcript",
                    "participant-1",
                    "integration-test",
                    finalized=True,
                ),
                direction,
            )


class BrowserAcceptanceLLM(FrameProcessor):
    """Produce a deterministic response while keeping Pipecat frame flow real."""

    class Settings:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            for response_frame in (
                LLMFullResponseStartFrame(),
                TextFrame("This deterministic response stays active for interruption testing."),
                LLMFullResponseEndFrame(),
            ):
                await self.push_frame(response_frame, direction)
            return
        await self.push_frame(frame, direction)


class BrowserAcceptanceTTS(TTSService):
    """Publish a short deterministic audio stream that can be interrupted."""

    class Settings(TTSSettings):
        def __init__(self, **kwargs) -> None:
            super().__init__(
                model=kwargs.get("model"),
                voice=kwargs.get("voice"),
                language=kwargs.get("language"),
            )

    def __init__(self, **kwargs) -> None:
        super().__init__(settings=kwargs.get("settings"))

    async def run_tts(self, text: str, context_id: str):
        assistant_audio_started.set()
        sample_rate = 24_000
        samples_per_frame = 480
        for frame_index in range(100):
            samples = array(
                "h",
                (
                    int(
                        4_000
                        * math.sin(
                            2
                            * math.pi
                            * 440
                            * (frame_index * samples_per_frame + sample)
                            / sample_rate
                        )
                    )
                    for sample in range(samples_per_frame)
                ),
            )
            yield TTSAudioRawFrame(
                samples.tobytes(), sample_rate, 1, context_id=context_id
            )
            await asyncio.sleep(0.02)


def main() -> None:
    """Start Runtime after swapping only external inference implementations."""
    for module_name, class_name, test_service in (
        ("pipecat.services.deepgram.stt", "DeepgramSTTService", BrowserAcceptanceSTT),
        ("pipecat.services.openai.llm", "OpenAILLMService", BrowserAcceptanceLLM),
        ("pipecat.services.cartesia.tts", "CartesiaTTSService", BrowserAcceptanceTTS),
    ):
        module = importlib.import_module(module_name)
        setattr(module, class_name, test_service)

    runtime_api = importlib.import_module("src.runtime.api")
    uvicorn.run(runtime_api.app, host="127.0.0.1", port=8001, log_level="warning")


if __name__ == "__main__":
    main()

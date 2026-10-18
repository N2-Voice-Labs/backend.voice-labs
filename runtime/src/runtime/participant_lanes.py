"""Per-participant audio lanes for two-person interpreter sessions (TR-020).

``LiveKitTransport`` emits one ``UserAudioRawFrame`` stream per remote participant
(``user_id`` is the LiveKit identity), interleaved on a single Pipecat stream. A
Deepgram connection transcribes one voice, so the interpreter pipeline runs one *lane*
per participant inside a native ``ParallelPipeline``::

    transport.input -> ParallelPipeline(
        [ParticipantLaneGate(0), VADProcessor, DeepgramSTTService, UserTurnProcessor, LaneTurnTagger(0)],
        [ParticipantLaneGate(1), VADProcessor, DeepgramSTTService, UserTurnProcessor, LaneTurnTagger(1)],
    ) -> TranslationProcessor -> ...

* :class:`ParticipantLaneGate` is a native ``FunctionFilter`` that lets through only the
  audio of its lane's participant (other participants' audio, and audio of unknown
  identities, is dropped for this lane), and keeps the lane's Deepgram language in sync
  with the participant's declared language, including for a participant that joins after
  the pipeline started.
* :class:`PerParticipantLiveKitTransport` gives every participant its own audio
  resampler. Pipecat's stock ``LiveKitInputTransport`` shares ONE stateful streaming
  resampler (48 kHz -> 16 kHz) between all participants; interleaved chunks of two
  people then bleed into each other (each lane hears the other person's speech,
  delayed and mixed in), so the lanes could never be clean. See the class docstring.
* :class:`LaneTurnTagger` stamps the lane's participant identity on the turn boundary
  frames (``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame``) as
  ``frame.metadata["user_id"]``. Transcripts already carry ``user_id`` from the STT
  service; the boundary frames do not, and the translator needs them per speaker.
"""

from __future__ import annotations

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    STTUpdateSettingsFrame,
    UserAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings
from pipecat.transports.livekit.transport import LiveKitInputTransport, LiveKitTransport

from .participants import Participant, ParticipantDirectory


class ParticipantLaneGate(FunctionFilter):
    """Pass only this lane's participant audio; keep the lane's STT language current."""

    def __init__(
        self,
        directory: ParticipantDirectory,
        lane: int,
        *,
        stt: FrameProcessor,
        language: str,
    ) -> None:
        super().__init__(self._allow, filter_system_frames=True, name=f"ParticipantLane{lane}")
        self._directory = directory
        self._lane = lane
        self._stt = stt
        self._language = language
        self._started = False
        directory.add_listener(self._on_participant)

    async def _allow(self, frame: Frame) -> bool:
        if isinstance(frame, InputAudioRawFrame):
            return self._directory.lane_of(getattr(frame, "user_id", "")) == self._lane
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._started = True
            # A participant may have joined between building the lane and starting it.
            await self._sync_language()

    async def _on_participant(self, participant: Participant) -> None:
        if self._started and self._directory.lane_of(participant.identity) == self._lane:
            await self._sync_language()

    async def _sync_language(self) -> None:
        language = self._directory.language_for_lane(self._lane)
        if language == self._language:
            return
        self._language = language
        # Queued on this processor so it travels the lane in order; only this lane's
        # Deepgram service applies it (``service=``) and reconnects with the new language.
        await self.queue_frame(
            STTUpdateSettingsFrame(delta=STTSettings(language=language), service=self._stt)
        )


class LaneTurnTagger(FrameProcessor):
    """Mark the lane participant's identity on user turn boundary frames."""

    def __init__(self, directory: ParticipantDirectory, lane: int) -> None:
        super().__init__(name=f"LaneTurnTagger{lane}")
        self._directory = directory
        self._lane = lane

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)
        ):
            identity = self._directory.identity_for_lane(self._lane)
            if identity:
                frame.metadata["user_id"] = identity
        await self.push_frame(frame, direction)


class PerParticipantLiveKitInput(LiveKitInputTransport):
    """LiveKit input that resamples each participant's audio with its own resampler.

    ``LiveKitInputTransport`` (pipecat 1.12) converts every participant's 48 kHz chunks
    with a single ``SOXRStreamAudioResampler``. That resampler is stateful (it keeps the
    filter history of the previous chunk), so when two microphones are interleaved on
    the one stream, the output labelled with one participant's ``user_id`` contains the
    other participant's audio. Live symptom: each lane transcribed the first words of
    the *other* person's speech. This subclass is otherwise a copy of
    ``_audio_in_task_handler`` with one resampler per participant identity.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._participant_resamplers: dict[str, object] = {}

    async def _audio_in_task_handler(self) -> None:
        async for audio_data in self._client.get_next_audio_frame():
            if not audio_data:
                continue
            event, participant_id = audio_data
            resampler = self._participant_resamplers.get(participant_id)
            if resampler is None:
                resampler = self._participant_resamplers[participant_id] = (
                    create_stream_resampler()
                )
            frame = event.frame
            audio = await resampler.resample(
                frame.data.tobytes(), frame.sample_rate, self.sample_rate
            )
            if not audio:  # the filter is still buffering
                continue
            await self.push_audio_frame(
                UserAudioRawFrame(
                    user_id=participant_id,
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=frame.num_channels,
                )
            )


class PerParticipantLiveKitTransport(LiveKitTransport):
    """``LiveKitTransport`` whose input resamples per participant (see above)."""

    def input(self) -> PerParticipantLiveKitInput:
        if not self._input:
            self._input = PerParticipantLiveKitInput(
                self, self._client, self._params, name=self._input_name
            )
        return self._input

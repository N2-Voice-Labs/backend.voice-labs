"""Per-participant audio lanes for two-person interpreter sessions (TR-020).

``LiveKitTransport`` emits one ``UserAudioRawFrame`` stream per remote participant
(``user_id`` is the LiveKit identity), interleaved on a single Pipecat stream. A
single STT connection transcribes one voice, so the interpreter pipeline runs one *lane*
per participant inside a native ``ParallelPipeline``::

    transport.input -> ParallelPipeline(
        [ParticipantLaneGate(0), VADProcessor, STTService, UserTurnProcessor, LaneTurnTagger(0)],
        [ParticipantLaneGate(1), VADProcessor, STTService, UserTurnProcessor, LaneTurnTagger(1)],
    ) -> TranslationProcessor -> ...

* :class:`ParticipantLaneGate` is a native ``FunctionFilter`` that lets through only the
  audio of its lane's participant (other participants' audio, and audio of unknown
  identities, is dropped for this lane), and keeps the lane's STT language in sync
  with the participant's declared language, including for a participant that joins after
  the pipeline started. Auto-detection lanes disable this synchronization.
* :class:`PerParticipantLiveKitTransport` gives every participant its own audio
  resampler. Pipecat's stock ``LiveKitInputTransport`` shares ONE stateful streaming
  resampler (48 kHz -> 16 kHz) between all participants; interleaved chunks of two
  people then bleed into each other (each lane hears the other person's speech,
  delayed and mixed in), so the lanes could never be clean. See the class docstring.
* :class:`TTSLaneGate` and :class:`ListenerRoutedLiveKitOutput` route the *output* (RM-001):
  one TTS per listener, each stamping its audio with its own ``transport_destination``
  (``translation-lane-N``); the output publishes that destination as the listener's own
  LiveKit track (``translation-for-<identity>``). See ``audio_routing``.
* :class:`LaneTurnTagger` stamps the lane's participant identity on the turn boundary
  frames (``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` and the VAD frames) as
  ``frame.metadata["user_id"]``. Transcripts already carry ``user_id`` from the STT
  service; the boundary frames do not, and the translator needs them per speaker.
"""

from __future__ import annotations

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    STTUpdateSettingsFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings
from pipecat.transports.livekit.transport import (
    LiveKitInputTransport,
    LiveKitOutputTransport,
    LiveKitTransport,
)

from .audio_routing import ListenerAudioRouter
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
        sync_language: bool = True,
    ) -> None:
        super().__init__(self._allow, filter_system_frames=True, name=f"ParticipantLane{lane}")
        self._directory = directory
        self._lane = lane
        self._stt = stt
        self._sync_declared_language = sync_language
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
        if not self._sync_declared_language:
            return
        language = self._directory.language_for_lane(self._lane)
        if language == self._language:
            return
        self._language = language
        # Queued on this processor so it travels the lane in order; only this lane's
        # STT service applies it (``service=``) and reconnects with the new language.
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
            frame,
            (
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
                # The VAD start is T0/T1 of the utterance timeline (RM-003).
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
            ),
        ):
            identity = self._directory.identity_for_lane(self._lane)
            if identity:
                frame.metadata["user_id"] = identity
        await self.push_frame(frame, direction)


class TTSLaneGate(FunctionFilter):
    """Let a listener lane's TTS see only the speech meant for that listener.

    ``TranslationProcessor`` tags every ``TTSSpeakFrame`` / ``TTSUpdateSettingsFrame`` with
    ``metadata["listener_lane"]``; each of the parallel TTS branches keeps its own and
    drops the rest, so a translation is synthesised exactly once, on the listener's lane.
    """

    def __init__(self, lane: int) -> None:
        super().__init__(self._allow, filter_system_frames=True, name=f"TTSLane{lane}")
        self._lane = lane

    async def _allow(self, frame: Frame) -> bool:
        if isinstance(frame, (TTSSpeakFrame, TTSUpdateSettingsFrame)):
            return frame.metadata.get("listener_lane") == self._lane
        return True


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

    def __init__(self, *args, router: ListenerAudioRouter | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._participant_resamplers: dict[str, object] = {}
        self._router = router  # only to hand a copy of each microphone to cinematic dubbing

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
            router = self._router
            if router is not None and router.dubbing is not None:
                # A copy for the other person's mix, taken after the lane's own audio is
                # resampled and before anything else: the lane's frame below is unchanged.
                await router.feed_original(participant_id, audio, self.sample_rate)
            await self.push_audio_frame(
                UserAudioRawFrame(
                    user_id=participant_id,
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=frame.num_channels,
                )
            )


class PerParticipantLiveKitTransport(LiveKitTransport):
    """``LiveKitTransport`` whose input resamples per participant (see above).

    With a ``router`` the output publishes one translation track per listener.
    """

    def __init__(self, *args, router: ListenerAudioRouter | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._router = router

    def output(self) -> LiveKitOutputTransport:
        if self._router is None:
            return super().output()
        if not self._output:
            self._output = ListenerRoutedLiveKitOutput(
                self, self._client, self._params, router=self._router, name=self._output_name
            )
        return self._output

    def input(self) -> PerParticipantLiveKitInput:
        if not self._input:
            self._input = PerParticipantLiveKitInput(
                self, self._client, self._params, name=self._input_name, router=self._router
            )
        return self._input


class ListenerRoutedLiveKitOutput(LiveKitOutputTransport):
    """LiveKit output that publishes one translation track per listener.

    Stock ``LiveKitOutputTransport`` writes every frame to its single ``pipecat-audio``
    track and ignores ``frame.transport_destination`` (``register_audio_destination`` is a
    no-op). Frames for a ``translation-lane-N`` destination are handed to the
    :class:`~runtime.audio_routing.ListenerAudioRouter`, which owns the per-listener tracks
    and their subscription permissions; the default track carries nothing and no
    participant is permitted to subscribe to it.
    """

    def __init__(self, *args, router: ListenerAudioRouter, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._router = router
        # Set by the pipeline builder; marks when the first voiced chunk is handed to LiveKit.
        self.timeline = None

    async def setup(self, setup) -> None:
        await super().setup(setup)  # connects to the room and publishes the default track
        client = self._client
        await self._router.attach(
            client.room,
            sample_rate=client._out_sample_rate,
            num_channels=self._params.audio_out_channels,
            queue_size_ms=self._params.audio_out_queue_size_ms,
            audio_sink=self._push_phone_audio,
        )

    async def _push_phone_audio(self, identity: str, event) -> None:
        """Feed a phone-room participant's microphone to the input like any other's."""

        await self._client._audio_queue.put((event, identity))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._router.clear()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        destination = frame.transport_destination
        if destination is None:
            return await super().write_audio_frame(frame)
        tracker = self.timeline
        submitting = tracker.output_submitting(frame.audio, destination) if tracker is not None else None
        try:
            return await self._router.write(
                destination, self._convert_pipecat_audio_to_livekit(frame.audio)
            )
        finally:
            if tracker is not None:
                tracker.output_submitted(submitting)

    async def stop(self, frame) -> None:
        await super().stop(frame)  # drains queued audio, then leaves the room
        await self._router.close()

    async def cancel(self, frame) -> None:
        await super().cancel(frame)
        await self._router.close()

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._router.close()

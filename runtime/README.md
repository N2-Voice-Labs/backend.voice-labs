# runtime.voice-labs

The live data plane. `RuntimeSession` owns in-memory pipeline state and emits versioned
events to the platform. Pipecat, LiveKit, VAD, turn detection, and provider adapters are
integration points; no fake audio path is presented as production-ready.

Run the health endpoint with `uvicorn src.runtime.api:app --reload`.

The real pipeline entrypoint is `python -m src.runtime.worker`. The standalone worker
uses `LIVEKIT_ROOM_NAME` and either `RUNTIME_LIVEKIT_TOKEN` or `LIVEKIT_API_KEY` plus
`LIVEKIT_API_SECRET` to mint a room-scoped agent token. API-triggered sessions receive
their room name from Platform and mint that token automatically; `RUNTIME_LIVEKIT_TOKEN`
is not required for that flow. Both paths also require configured Deepgram,
OpenAI-compatible, and TTS credentials: `CARTESIA_API_KEY`/`CARTESIA_VOICE_ID` for the
default `cartesia` provider, or `ELEVENLABS_API_KEY`/`ELEVENLABS_VOICE_ID` when the
session selects `elevenlabs` (only the selected TTS provider's credentials are required;
ElevenLabs voice cloning/deletion is not implemented).
The factory loads Silero VAD and Pipecat's bundled Smart Turn v3 model, then connects the
native LiveKit transport to STT, LLM, and TTS. It fails at startup when provider settings
are missing; it does not substitute fake audio.

API-triggered session ownership and status are currently process-local. Run one Runtime
process/replica; multiple workers or replicas are unsupported until session admission has
shared ownership and crash-safe fencing. Runtime retains at most 1,024 recent terminal
session statuses for handoff idempotency; Platform remains the durable session authority.

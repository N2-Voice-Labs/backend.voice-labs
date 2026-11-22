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
session selects `elevenlabs` (only the selected TTS provider's credentials are required).
If a session's ElevenLabs voice id is missing or malformed, the pipeline falls back to
`ELEVENLABS_VOICE_ID` and logs a warning; it fails only when that default is unusable too.

The factory loads Silero VAD and Pipecat's bundled Smart Turn v3 model, then connects the
native LiveKit transport to STT, LLM, and TTS. It fails at startup when provider settings
are missing; it does not substitute fake audio.

API-triggered session ownership and status are currently process-local. Run one Runtime
process/replica; multiple workers or replicas are unsupported until session admission has
shared ownership and crash-safe fencing. Runtime retains at most 1,024 recent terminal
session statuses for handoff idempotency; Platform remains the durable session authority.

## Tenant HTTP tools (SSRF guard)

Every HTTP tool call resolves the configured host, requires all addresses to be globally
routable (no loopback, private, link-local/metadata, CGNAT, multicast) and connects to the
validated IP (DNS-rebinding safe, `runtime/net_guard.py`, shared with voice-sample
download). http and https are allowed for tools; samples stay https only. Redirects are
never followed (a 3xx returns the tool error `redirect_not_followed`). Blocked targets
return the tool error `url_not_allowed`. For local development and tests only, set
`RUNTIME_TOOL_ALLOW_PRIVATE=1` (default off) to skip the check.

## Platform event delivery

Events are classified (`runtime/events.py`):

- Critical: `RuntimeStarted`, `TranscriptFinal`, `ToolCalled`, `OutcomeRecorded`,
  `HandoffRequested` and the terminal `SessionEnded`/`SessionFailed`/`RuntimeFailed`.
  Delivery is retried with exponential backoff (0.1 s doubling, capped at 8 s) for about
  60 s. If it still fails, the call is ended (the Platform would otherwise hold an
  incomplete record), as it is when the queue (128) is full of critical events.
- Best-effort: everything else (`LatencyMeasured`, `TurnDebug`, speech/turn timeline
  events, `ProviderError`, `EngineFallback`, ...; new types default to best-effort).
  Three delivery attempts, then dropped. Under queue pressure they are dropped first
  (a critical event evicts the oldest queued best-effort one), counted in
  `RuntimeSession.dropped_events`, with a rate-limited warning (one per 10 s). They never
  end the call.
- A non-retryable 4xx from the Platform on any non-terminal event is logged and the event
  dropped; it never ends the call.

## LiveKit token on engine fallback

A token the Runtime minted itself lives 10 minutes. When the engine fallback restarts the
pipeline, the Runtime mints a fresh token (`PipelineConfig.livekit_token_minted`). A token
supplied by the Platform or `RUNTIME_LIVEKIT_TOKEN` is reused unchanged.

## Bounded waits

`POST /sessions/{id}/end` waits up to 40 s for the session to stop; if it is still
stopping it answers `{"status": "ending"}` and the session keeps stopping in the
background. Shutdown waits up to 30 s for sessions, then cancels the rest.

## Voice cloning

Platform-only endpoints (`X-Runtime-Token`), for providers `cartesia` (`CARTESIA_API_KEY`)
and `elevenlabs` (`ELEVENLABS_API_KEY`, Instant Voice Cloning):

- `POST /voices/clone`: JSON `provider`, `name`, `language`, `sample_url`, `description?`.
  Downloads the consented sample (https, public hosts, <=10 MiB, audio) and clones it.
- `POST /voices/clone-upload`: `multipart/form-data` `provider`, `name` (1-200 chars),
  `language`, `description?`, `file` (a recorded sample; held in memory only, never stored).
  Over 10 MiB, empty or non-audio files are `422 invalid_sample`; the size limit is enforced
  while the body streams in.
- `DELETE /voices/{provider}/{provider_voice_id}`: deletes the cloned voice (404 upstream
  counts as deleted).

Both clone endpoints answer `200 {"provider_voice_id"}`, `422 unsupported_tts_provider`,
`503 tts_not_configured`, `502 clone_provider_error`, or `502 clone_plan_upgrade_required`
when the provider's plan does not include cloning. The ElevenLabs request shape follows its
API reference but has not been run against the real API. `python-multipart` is a required
dependency (form parsing). See `docs/contracts.md` for the full contract.

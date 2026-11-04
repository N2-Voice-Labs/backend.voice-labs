# Backend Milestone 10: Live translation (interpreter mode)

Status: updated 2026-11-05 (stage 3c added; TR-050 hardened and mock-tested, TR-051 real-provider check still open). Stages 1–3b were verified with real providers on 2026-10-15 to 2026-10-18; **stage 3's per-listener routing was rebuilt after that real-provider check and has not been re-verified** (mock-tested only); stage 4 (phone) implemented behind configuration, not verified without a SIP trunk (backend only; frontend untouched)

An agent can run as an **interpreter** instead of an assistant: it repeats what each
speaker says in the other language, using a chosen voice. This builds on the existing
Runtime (LiveKit + Pipecat) and providers. The first stage needs no new provider keys.

## Out of scope

- **Instant cloning of a caller's voice from live call audio.** Without that person's
  consent it is impersonation, so voices always come from consented voice profiles.
- **Emotion and prosody transfer.** Limited to what the TTS provider exposes (speed and
  emotion controls); not promised.

## Design decisions

- **Cascaded first, streaming later.** Stage 1 uses Deepgram STT → OpenAI translation →
  Cartesia TTS per finished utterance. Expected delay is about 1.5–3 s after the speaker
  stops. Stage 2 adds a streaming speech-to-speech engine (OpenAI `gpt-realtime-translate`),
  which translates while the speaker is still talking, as a second engine. We compare the
  two on latency and quality rather than assuming either is better.
- **Direction.** `auto` translates each utterance into the *other* language of the pair,
  using Deepgram's multilingual detection. It is only allowed when both languages are in
  Deepgram's code-switching set (`en, es, fr, de, hi, ru, pt, ja, it, nl`), so English ↔
  Russian works. `one_way` fixes the source and target (e.g. `ko → en`) and supports any
  language both providers handle.
- **Voice.** Translations are spoken by the agent's voice profile (a Cartesia multilingual
  voice), so a consented clone of the user's own voice can speak the translation.
- **No instant cloning of other callers.** Copying a caller's voice in real time without
  their consent is impersonation, so we only use voice profiles that passed the consent
  check. Interpreter sessions should announce that an AI interpreter is on the call.
- **No barge-in in interpreter mode.** A translation is never cut off by the other person
  starting to speak; their speech is translated next.
- **Not marketed as zero latency.** We measure the delay from the end of speech to the first
  translated audio and report p50/p95 through the existing analytics.

## Contract

Agents gain `mode` (`agent` | `interpreter`, default `agent`) and `interpreter`
(`null` or an object):

```json
{
  "mode": "interpreter",
  "interpreter": {
    "languages": ["en", "ru"],
    "direction": "auto"
  }
}
```

- `languages`: exactly two distinct codes from the supported set
  `en, ru, ko, ja, zh, es, fr, de, it, pt, nl, hi, tr, pl`.
- `direction`: `auto` (both languages must be in Deepgram's code-switching set) or
  `one_way` (`languages[0]` → `languages[1]`).
- `mode: "interpreter"` requires `interpreter`; `mode: "agent"` requires it to be null.

The Platform→Runtime session contract always carries `mode` and `interpreter` (null for
assistant agents).

## Tasks

### Stage 1: cascaded interpreter

- [x] **TR-001 Platform.** `agents.mode` and `agents.interpreter` with migration `0005`,
  validation and error codes, API read/write, session contract fields, tests.
- [x] **TR-002 Runtime.** Interpreter pipeline: STT with `multi` (auto) or the fixed source
  language (one_way) → translation processor (strict translate-only prompt, no memory
  beyond a short window for consistency) → TTS in the target language with the agent's
  voice. Emits `TranscriptFinal` for the original (user) and the translation (assistant)
  with language metadata, plus latency events. Interruptions disabled.
- [x] **TR-003 Verification.** Real-provider check with synthesized English and Russian
  speech in both directions; record measured latency.

  Result (2026-10-15, English ↔ Russian, `auto`, real Deepgram/OpenAI `gpt-4.1-mini`/Cartesia,
  synthesized speech played into a live LiveKit session, n=2):

  | Utterance | Translation |
  | --- | --- |
  | ru: "Здравствуйте. Хотел бы записаться на прием на завтра утром." | en: "Hello. I would like to make an appointment for tomorrow morning." |
  | en: "Hello. Would like to book an appointment for tomorrow morning." | ru: "Здравствуйте. Хотел бы записаться на прием на завтра утром." |

  | Stage | p50 |
  | --- | --- |
  | STT first result | 415 ms |
  | Translation (OpenAI) | 1,623 ms |
  | TTS first audio | 191 ms |
  | End of speech → first translated audio | 1,819 ms |

  The translation call dominates. Next optimizations: stream the translation into TTS
  sentence by sentence, try a faster model, and compare with the streaming engine (TR-010).

### Stage 1b: speed and voice

- [x] **TR-004 Stream the translation.** Speak the translation sentence by sentence while
  the model is still generating, and end the turn sooner (0.3 s pause instead of 0.5 s).
  Result (2026-10-18, same EN ↔ RU test, n=4): end of speech → first translated audio p50
  1.47 s (was 1.82 s), p95 2.67 s. OpenAI time to the first translated words now dominates
  and varies; the streaming engine (TR-010) targets that.
- [x] **TR-005 Speak in the user's own voice.** `POST /voices/{id}/provision` clones a
  consented `https://` sample at Cartesia and marks the voice `ready`; assign it to the
  interpreter agent. Not yet exercised against Cartesia (needs a consented recording). Depends on consent-gated cloning (LV-022 in
  the [Backend Live Plan](../../stack.voice-labs/docs/BACKEND_LIVE_PLAN.md)): a cloned voice becomes `ready` and the
  interpreter speaks with it.

### Stage 2: streaming engine

- [x] **TR-010** `engine: "cascaded" | "realtime"` on the interpreter config; a Runtime
  adapter for OpenAI `gpt-realtime-translate` (provider voice, not the user's; `one_way`
  only, because a session has one fixed output language and doesn't report the source
  language). Verified 2026-10-18 with real OpenAI, Russian → English, n=3: correct
  translations; first audible translated audio p50 0.96 s **after the speaker starts**
  (it translates while they talk), p95 1.6 s. Silent output chunks are not counted.
  Known issue: transcript text occasionally misses a space between streamed pieces.
- [ ] **TR-011** Latency and quality comparison between engines on the evaluation set.
  First data point (same Russian sentence): cascaded speaks 1.47 s after the speaker
  stops; realtime starts 0.96 s after the speaker starts. Cascaded keeps the user's own
  voice and supports `auto` two-way; realtime is faster but one-way with provider voices.

### Stage 3: two speakers, two voices

- [x] **TR-020** Per-participant language and voice: each speaker hears the other in that
  speaker's own consented voice profile. `POST /voice/sessions` takes a `participant`
  (language, voice profile) and `POST /voice/sessions/{id}/participants` adds the second
  person. Each person gets their own VAD and STT lane in their declared language, so no
  detection is needed and Korean two-way works. Pipecat's LiveKit input resampled every
  participant through one shared resampler, which leaked one person's audio into the
  other's lane; Runtime now resamples per participant. Verified 2026-10-18 with real
  providers, Korean ↔ English: correct translations both ways, no crosstalk,
  end of speech → first translated audio 0.8–1.4 s. Per-listener routing (each person hears
  only the other's translation) is done in RM-001 of the [Roadmap](../../stack.voice-labs/docs/BACKEND_ROADMAP.md).

### Stage 3b: agent + translator

- [x] **TR-040** An assistant agent keeps working in its own language while the caller
  speaks another: the caller's speech is translated in, and the agent's reply is translated
  out and spoken in the caller's language. Agent field `translation`
  `{working_language, caller_languages}`; the LLM context, tools and knowledge stay in the
  working language. Verified 2026-10-18 with real providers (English receptionist, Russian
  caller, n=2): correct Russian replies using the agent's English instructions; end of
  speech → first audio p50 1.9 s, p95 2.3 s (translation legs default to `gpt-4.1-nano`).

### Stage 3c: translation sessions as the product surface

- [x] **TR-050** `POST /translations` and `POST /translations/{id}/invite`: a two-person
  interpreter session with no agent to configure. The Platform keeps a hidden system agent
  (`agents.system`, migration `0014`) per organization and configuration, always cascaded and
  two-way with private listening. Done 2026-11-05 with Platform tests and a mocked Runtime;
  not yet run with real providers. *Hardening added 2026-11-05 (same day, mock-tested; second pass after an independent review):* refused
  requests leave no hidden agent or session rows, hidden configurations are bounded per organization
  (`PLATFORM_MAX_TRANSLATION_CONFIGS`, `429 too_many_translation_configs`), `tts_provider_id` is
  accepted (`422 invalid_tts_provider_id` when not an enabled tts record of the organization); the
  hidden agents are invisible and immutable through the public agent, tool, knowledge, session,
  deployment and campaign routes (`404 agent_not_found` / `422 invalid_agent_id`), unique per
  organization and key, race-free on first use; `GET /translations/{id}` and the invite route
  answer `404 translation_not_found` for sessions of ordinary agents; Runtime warns when a
  session request omits `mode`. Details in [Contracts](contracts.md#translation-sessions-translations).
  Why: the engine already interpreted, but the only way to
  reach it was to create an "agent" with a prompt, LLM and tools, so the API read as an AI
  voice agent rather than a translator. Contract: [Contracts](contracts.md#translation-sessions-translations).
- [ ] **TR-051** Run the first real-provider check through `/translations` (two browsers,
  Korean ↔ English) and record latency next to TR-020.
  *Note 2026-11-05:* still open for `/translations`; nothing has run `/translations` with real providers (only the ElevenLabs layer was exercised for real, by `runtime/scripts/elevenlabs_smoke.py`, 15 of 15 checks). The procedure is
  [Smoke test](smoke-test.md) (billable, human-run, results section empty). `/translations` takes an
  optional `tts_provider_id`, so Part A (Cartesia) and Part B (ElevenLabs) of the smoke test both use
  it. Platform rejects a voice from a different provider (`409 voice_provider_mismatch`) when both
  providers are known. Done when the smoke test's pass/fail table is filled in and the
  latency is recorded here.

### Stage 4: phone calls

- [ ] **TR-030** Bridge two SIP legs through LiveKit with the interpreter in the middle
  (depends on SIP work in [Backend Live Plan](../../stack.voice-labs/docs/BACKEND_LIVE_PLAN.md)). PSTN audio is often
  8 kHz, which lowers recognition quality, so prefer HD voice trunks.

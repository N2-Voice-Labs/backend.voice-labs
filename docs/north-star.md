# North Star plan

Status: agreed 2026-10-20; updated 2026-11-05 (translation sessions API added, ElevenLabs preview and live TTS added in code, neither verified with real providers). NS-06, NS-08 and NS-01 (runner) done; full NS-01 run next. Backend only; the web app is
planned separately, except the one demo-critical change listed under "Needs you".

## The goal

> Voice Labs is the orchestration layer for realtime multilingual voice communication. It
> connects existing AI and realtime infrastructure so that humans and AI can talk naturally
> across languages, browsers and phones.

We do not build our own WebRTC, SIP, STT, LLM, translation model, TTS or VAD. We connect the
best existing ones, make them interchangeable, route them correctly, measure them and turn
them into one reliable experience. Research tools (REA, Playwright, Wireshark, DevTools)
help us decide what to integrate. They are not the product.

Every task in this plan is judged by one question: **does it make the first proof better?**

## Product surface

Translation between two people is the primary product, and its entry point is
`POST /translations` (see [Contracts](contracts.md#translation-sessions-translations)), which needs no
agent, prompt or LLM. AI voice agents (`mode=agent`) and agents that translate for their caller
(`translation`) are secondary and use the older agent-first endpoints.

## The first proof

A Korean speaker in a browser and an English speaker on a phone talk to each other in real
time. Each hears only the other person, translated into their own language.

```text
Korean person (browser) ──► Voice Labs ──► English audio ──► English person (phone)
Korean person (browser) ◄── Korean audio ◄── Voice Labs ◄── English person (phone)
```

It is done when all of these are true in a live call:

| Requirement | Target | Status |
| --- | --- | --- |
| Realtime speech in both directions | two people, one room | ✅ browser ↔ browser (2026-10-18, Korean ↔ English, 0.8–1.4 s; before the routing rebuild) |
| Correct per-person audio routing | nobody hears their own voice or the other's original | 🧪 RM-001/002 implemented; historical result, routing later rebuilt, not re-measured with real providers (no date or measurement for the current code) |
| Transcripts for both people | original and translation, per speaker | ✅ (2026-10-18 runs; not re-checked since) |
| Measurable latency | p50 and p95 time to first translated audio, per engine | 🟡 p50 only |
| Low latency | realtime ≤ 1.2 s p95 from speech start; cascaded ≤ 1.5 s p95 after speech ends | 🟡 not yet measured at p95 |
| Interruption handling | a defined policy, measured | ❌ |
| Phone support | one side on a real phone number | ❌ needs SIP trunk |
| Interchangeable providers | swap OpenAI → Qwen, Cartesia → ElevenLabs, Deepgram → another STT by configuration only | 🟡 translation engines are swappable; Cartesia → ElevenLabs works by provider record for agent sessions and for `/translations` via `tts_provider_id` (2026-11-05, mock-tested only) |
| Survives failures | a provider error or a dropped participant does not kill the call | ❌ |

## What Voice Labs owns

| Area | Where it lives today | Gap |
| --- | --- | --- |
| Participant context (who, which language in and out, role) | `participants.py`, session contract | none for two people |
| Audio routing (who hears which track) | `audio_routing.py` | phone participants not tested yet |
| Language routing (A → B's language, B → A's language) | `interpreter.py` | none |
| Orchestration (STT → translate → TTS, or realtime audio → audio) | `translation_engine.py` | none |
| Provider abstraction | `translation_engine.py` (engines); `pipeline.py` hard-codes Deepgram, OpenAI, Cartesia | **STT and TTS registries** *(note 2026-11-05: the registries exist since NS-06, `speech_providers.py`; this row is the original gap statement)* |
| Reliability | events client retries only | **fallbacks, reconnection, failure states** |
| Evaluation | `evals/` dataset and runner, `utterance_timeline` events | **p95, interruptions, audio quality, provider benchmark** |

## What we integrate instead of building

| Problem | Decision |
| --- | --- |
| WebRTC | LiveKit |
| SIP | LiveKit SIP with Twilio or Telnyx |
| STT | Deepgram today; add one more provider behind the same interface |
| LLM and text translation | OpenAI today; Qwen through its OpenAI-compatible API |
| Realtime speech translation | OpenAI `gpt-realtime-translate`; Qwen LiveTranslate as a second engine |
| TTS and voice cloning | Cartesia today; add ElevenLabs |
| VAD and turn detection | Silero and Pipecat smart-turn |
| Audio analysis | FFmpeg and Python |

## Phases

IDs continue from the [roadmap](../../stack.voice-labs/docs/BACKEND_ROADMAP.md). ✅ means done in the code and marked done there; it is not a claim of real-provider verification (see [Status](status.md) for evidence levels).

### P0: correct interpreter (done)

- ✅ **RM-001** Participant context
- ✅ **RM-002** Per-listener audio routing
- ✅ **RM-010** Translation engine interface
- ✅ **RM-011** Realtime and cascaded engines with latency modes

### P0: make it measurable

- ✅ **RM-003** T0 → first translated audio, per utterance and engine.
- [ ] **NS-01 p50/p95 report.** `GET /analytics/summary` already returns p50 and p95. Add a
  repeatable script that plays the evaluation audio into a live room (two simulated
  participants, both engines, every latency mode, at least 30 utterances each) and writes one
  table. Done when the table in [Evaluation](../../stack.voice-labs/docs/evaluation.md) can be regenerated with one
  command.
- [ ] **NS-02 Interruption policy and measurement.** Agreed interpreter policy:
  - When A speaks again while A's previous translation is still playing to B, the old
    translation keeps playing and the new one is queued (an interpreter should not drop
    meaning).
  - When B starts speaking while hearing A's translation, B's own speech is translated as
    normal, and A's translation playing to B is ducked by about 10 dB instead of being cut.
  - In agent modes, barge-in cuts the agent off (today's behavior).

  Then measure: time from barge-in to agent audio stopping (target ≤ 300 ms), and whether any
  translated meaning is lost in interpreter mode. Emit `interruption_timeline` events next to
  `utterance_timeline`.
- [ ] **NS-03 Audio quality checks.** Record each `translation-for-<identity>` track in test
  runs and check it with FFmpeg and Python for gaps longer than 200 ms inside an utterance,
  clipping, and loudness (target about −16 LUFS, the same for both directions). Done when the
  NS-01 script reports these columns.

### P1: make it real

- [ ] **RM-013 SIP trunk.** Needs a Twilio or Telnyx account and a phone number. The code is
  ready behind `LIVEKIT_SIP_TRUNK_ID`.
- [ ] **RM-020 Browser ↔ phone interpreter** (the first proof). A phone participant joins the
  same room through LiveKit SIP and gets the same participant context and routing. Check:
  8 kHz phone audio works with STT, the Korean voice sounds right through a phone codec, and
  the phone side never hears the original.
- ✅ **RM-004** Authentication
- [ ] **NS-04 Provider failure handling.**
  - Realtime engine error or timeout → switch that direction to the cascaded engine for the
    rest of the call and emit `EngineFallback`.
  - STT or TTS websocket drop → reconnect with backoff and emit `ProviderReconnected`;
    after three failures, emit `ProviderFailed` and end that direction cleanly instead of
    hanging.
  - Test each case by injecting failures in unit tests and by killing the network to one
    provider in a live run.
- [ ] **NS-05 Participant reconnection.** A participant who drops and rejoins within 60 s gets
  the same context, tracks and routing back without restarting the session. A participant who
  does not come back ends their side cleanly and the other side is told (`ParticipantLeft`).

### P1: make providers interchangeable and comparable

- [x] **NS-06 STT and TTS registries.** Done 2026-10-21 (`runtime/speech_providers.py`). Same pattern as `translation_engine.py`: pipelines ask
  a registry for "stt for Korean" or "tts with this voice" instead of importing Deepgram and
  Cartesia directly. Provider choice comes from the provider records the Platform already
  stores. No behavior change; all tests stay green.
- [ ] **NS-07 Second provider for each slot.** (2026-11-05 note, replaces the earlier "live calls are not wired":
  ElevenLabs live TTS, preview and the apology-clip renderer are implemented and mock-tested; the ElevenLabs layer (REST preview, live WebSocket TTS, voice switch,
  interruption, shutdown, invalid-key failure) was then verified against the real API on 2026-11-05, 15 of
  15 automated checks on a free-tier key, objective measurements only (see
  [Status](status.md#elevenlabs-real-provider-run-2026-11-05)); see also
  [Contracts](contracts.md#elevenlabs-live-tts). Still open for ElevenLabs: the full two-way call
  ([Smoke test](smoke-test.md)), which also needs Deepgram, OpenAI and LiveKit. `/translations` takes `tts_provider_id` and
  Platform rejects a voice whose provider differs (`409 voice_provider_mismatch`, only when both are known).
  Qwen and the second STT are untouched, so NS-07
  stays open.)
  - TTS: ElevenLabs (Pipecat has a service for it), including its voice IDs in voice records. *Done in code
    2026-11-05; service layer real-verified 2026-11-05 (15 of 15 checks), in-call use unverified:* the Pipecat `elevenlabs` extra is in `runtime/pyproject.toml` with regenerated
    locks, and the TTS credentials follow the selected provider (`ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`;
    the `cartesia_*` fields keep their names). *Original text kept below as history:* Live wiring
    needs the Pipecat `elevenlabs` extra in `runtime/pyproject.toml` (and a lock update) and the
    Cartesia-named config fields (`cartesia_api_key`, `cartesia_voice_id`) generalized.
  - Text translation: Qwen through the existing `openai-compatible` LLM path.
  - Realtime engine: Qwen LiveTranslate as a second `TranslationEngine`.
  - STT: one provider with strong Korean (candidates: ElevenLabs Scribe, Speechmatics,
    AssemblyAI; pick by a quick test on our Korean items).
  - Done when the demo below works by changing only provider records.
- ✅ **RM-012** Evaluation dataset (references still need native-speaker review)
- [ ] **RM-022 Provider benchmark.** Run every provider combination on the evaluation set and
  report time to first audio (p50/p95), WER/CER, chrF, terminology recall, audio quality
  (NS-03) and cost per minute. Publish the table in [Evaluation](../../stack.voice-labs/docs/evaluation.md) and choose
  the default stack from it.
- [x] **NS-08 Glossary** (done 2026-10-21; quality work, done early because it is cheap and fixes a failure
  we have already seen). Per-deployment term list passed to translation prompts and STT
  keyword boosting. The evaluation already found a wrong term (residence permit translated as
  "разрешение на проживание" instead of "вид на жительство").

### P2: add AI

- [ ] **RM-021** Phone ↔ phone interpreter.
- [ ] **NS-09 Human ↔ AI multilingual.** Agent + translator works today in the browser; check
  it on the phone and fix the slow split turn (one run took 7 s).
- [ ] **NS-10 AI ↔ human phone calls** using the campaign dispatcher and SIP.
- [ ] **NS-11 Human ↔ AI ↔ human.** An AI agent joins an interpreter room as a third
  participant with its own context, so it can answer or use tools while two people talk.

### P3: advanced

- [ ] **RM-030** Stability layer for mid-sentence translation.
- [ ] **RM-031** Voice-preserving translation with consented voice profiles.
- [ ] **RM-032** Live self-cloning, with consent.

## The demo

Run the same Korean ↔ English call three times. Between runs, change only provider records
in the Platform; no code changes and no redeploy:

1. OpenAI + Deepgram + Cartesia
2. Qwen + Deepgram + ElevenLabs
3. Qwen LiveTranslate (realtime engine)

Show the NS-01 table for each run side by side.

## Order of work

The order follows what each step proves: the call works correctly and quickly, then
reliably, then with any provider, then on the phone. Tasks that need nothing from outside run
first; the SIP account is requested in parallel so it does not hold up the first proof.
Agent size follows difficulty.

```text
NS-06 ──┬──► NS-04 reliability
        └──► NS-07 second providers ──► RM-022 benchmark ──► choose default stack
NS-01 ──┬──► NS-02 interruption
        └──► NS-03 audio quality
SIP account ──► RM-013 ──► RM-020 browser ↔ phone
```

| Order | Task | Agent | Blocked by |
| --- | --- | --- | --- |
| 1 | NS-06 STT/TTS registries | Sonnet | nothing |
| 1 | NS-01 p50/p95 script | Sonnet | nothing |
| 1 | NS-08 glossary | Sonnet | nothing |
| 2 | NS-02 interruption policy and metrics | Sonnet | NS-01 |
| 2 | NS-03 audio quality checks | Haiku | NS-01 |
| 2 | NS-04 provider failure handling | Sonnet | NS-06 |
| 3 | NS-07 second providers | Sonnet | NS-06, API keys |
| 3 | NS-05 participant reconnection | Sonnet | nothing |
| 4 | RM-013 → RM-020 browser ↔ phone | Sonnet + a live test with you | SIP account |
| 5 | RM-022 benchmark | Haiku runs, Sonnet reports | NS-01, NS-03, NS-07 |
| 6 | P2 items | Sonnet | RM-020 |

After each task: all test suites green, no tracked `.pyc` changes, docs updated, and a live
check with real providers where the task touches audio.

## Needs you

1. **SIP account** (Twilio or Telnyx) and one phone number. This is the only blocker for
   the first proof; start it now, in parallel with NS-06, NS-01 and NS-08.
2. **Web app (demo-critical, do now):** play any subscribed audio track
   (`translation-for-<identity>`), not only `pipecat-audio`. Without it, two-person calls are
   silent in the browser even though the backend is routing correctly.
3. **API keys** for ElevenLabs and Qwen (Alibaba Cloud Model Studio), and one more STT
   provider once we pick it.
4. **A native Korean and Russian speaker** to review the evaluation references.
5. **Rotate the API keys** shared in chat, and commit the current work.

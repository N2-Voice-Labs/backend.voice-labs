# Backend status

Updated 2026-11-11 (rewritten; the long dated run logs that used to be here are in git history,
see [Run history](#run-history)). Backend only. The goal and what comes next are in
[Goal](goal.md); API details are in [Contracts](contracts.md).

**Legend.** A row gets the highest level that is actually evidenced, never the one we expect.

| Mark | Level | Meaning |
| --- | --- | --- |
| ✅ | Real-provider-verified | Run against the real service (Deepgram, OpenAI, Cartesia, ElevenLabs, Google, LiveKit, PostgreSQL) and the result observed. The date is in the notes. Code changed after that date has only mock evidence unless the row says otherwise. |
| 🧪 | Mock-tested | Covered by automated tests with fakes. Says nothing about the real provider. |
| 🛠 | Implemented | Code exists, no test covers the whole path. |
| ⏳ / 🔒 | Planned / blocked | Not built / waiting on something outside the code. |

## In one paragraph

Two services. **Platform** (FastAPI, PostgreSQL) stores agents, voices, conversations,
organizations and keys, and hands each call to Runtime as a session contract. **Runtime**
(FastAPI, Pipecat, LiveKit) runs the live audio and reports signed events back. Platform never
touches audio. The goal is an **AI voice agent** ([Goal](goal.md)). The same Runtime also runs
two-person translated calls and an agent that translates for its caller.
`feature/voice-renovation` was merged into `main` on 2026-11-11 (PR #1).

## Voice agents (the goal)

| Feature | Status | Notes |
| --- | --- | --- |
| Talk to an agent in a browser | ✅ 2026-10 (on older code) | Deepgram → OpenAI → Cartesia through LiveKit, about 1.65 s from end of speech to first reply audio. Not re-run since the 2026-11 provider changes. |
| Agent + translator (English agent, Russian caller) | ✅ 2026-10-18, n=2 | About 1.9 s. |
| Agent settings: prompt, language, model, voice, temperature; templates | ✅ | |
| Tools: HTTP tools and built-ins `record_outcome`, `transfer_to_human`, `search_knowledge` | 🧪 | Timeouts and logging; mock servers only. |
| Knowledge documents | 🧪 | Keyword search over assigned documents, no embeddings. Returns a source name, no link. |
| Interruption (barge-in) | 🧪 | Time to stop never measured. |
| Handoff to a human | 🧪 | Handoff event and transfer tool. |
| Outcomes, transcripts, signed events | ✅ | Each event stored exactly once (idempotent by event id). |

## Two-person translation (not the goal; keep working)

| Feature | Status | Notes |
| --- | --- | --- |
| Shareable-link calls (`/calls`, invite codes) | ✅ 2026-11-07, one call | Two phones, English → Korean, ElevenLabs STT/TTS + Google Translation. Call 1 worked; call 2 garbled (speakers used a language they had not picked); call 3 failed on the ElevenLabs free quota. The browser was not yet joining the listen room then (fixed 2026-11-08), so whether anyone *heard* the voice is unknown. See [Calls](calls.md). |
| Translation sessions (`POST /translations`) | 🧪 | Hidden system agent; cascaded, two-way, private listening. |
| Per-listener routing (each hears only the other's translation) | 🧪 | ✅ 2026-10-18 on older code (Korean ↔ English, 0.8–1.4 s); rebuilt since, not re-measured. |
| Latency, last spoken sound → first translated audio | ✅ 2026-11-06, n=37 recordings | Median 1,289 ms. About 460 ms is speech-to-text including the 200 ms pause; a further ~330 ms is transport and buffering. Small sample, an indication only. |
| Realtime engine (OpenAI `gpt-realtime-translate`) | ✅ 2026-10-18, n=3 | One direction, provider voice. |
| Glossary | 🧪 | Step-by-step engine only. |
| Cinematic dubbing | 🧪 | Off by default, never heard by listeners. See [Dubbing](dubbing.md). |

## Phone (SIP)

| Feature | Status | Notes |
| --- | --- | --- |
| Outbound calls, campaigns, bridge, transfer | 🧪 | Behind `LIVEKIT_SIP_TRUNK_ID`. |
| Any real phone call | 🔒 | Needs a SIP provider account and a number. |

## Platform and safety

| Feature | Status | Notes |
| --- | --- | --- |
| Database and migrations `0001`–`0014` | ✅ PostgreSQL 16.2, 2026-11-05 | Upgrade, downgrade, upgrade round trip; full Platform suite passed on PostgreSQL. |
| Authentication: superadmin keys, organization keys (`vlk_`) with roles, session tokens (`vls_`) | ✅ | Off when no keys are set. Keys and tokens stored hashed. |
| Runtime ↔ Platform authentication (`X-Runtime-Token`) | ✅ | |
| Fail-closed provider check | ✅ 2026-11-07 | New calls refused with 503 `provider_unavailable` when ElevenLabs ran out of quota. |
| Analytics summary (conversations, languages, p50/p95) | 🧪 | `GET /analytics/summary`. |
| `/docs` and `/openapi.json` public when auth is on | 🟡 | Decision needed. |

## Providers

Swappable through registries in `runtime/src/runtime/speech_providers.py`; the session contract
selects them by provider record.

| Slot | Registered | Real evidence |
| --- | --- | --- |
| STT | Deepgram nova-3 (default), ElevenLabs realtime | Deepgram ✅ 2026-10; ElevenLabs ✅ 2026-11-06/07 |
| LLM / translation | OpenAI and OpenAI-compatible (default), Gemini, Google Cloud Translation (translation sessions only, Gemini fallback) | OpenAI ✅ 2026-10; Gemini and Google ✅ 2026-11-06 outside a call, Google ✅ 2026-11-07 in a call |
| TTS | Cartesia sonic (default), ElevenLabs | Cartesia ✅ 2026-10; ElevenLabs ✅ 2026-11-05 (15 of 15 service checks), 2026-11-07 in a call |
| Voice cloning | Cartesia only | 🧪, no real clone made |

## Measuring and testing

- **Tests:** run the commands in the [README](../README.md#checks). Last recorded on 2026-11-05:
  Platform 734 passed, Runtime 866 passed, Evals 137 passed (re-run before quoting).
- **Evaluation set:** 140 synthetic items with WER, Korean CER, chrF and term recall in `evals/`.
  The reference translations still need native Korean and Russian reviewers.
- **Live latency runner** (`live_latency`) and `utterance_timeline` events give per-turn timings.
- **Real-provider checklist:** [Smoke test](smoke-test.md).

## Open risks

1. **No agent run on current code.** Every agent ✅ above predates the 2026-11 changes.
2. **Provider accounts are free tier.** The ElevenLabs quota ran out mid-test (resets about
   2026-12-05). Keys shared in chat must be rotated.
3. **Nothing is deployed.** Platform and Runtime run only on a developer machine.
4. **Voice id patterns disagree:** Platform accepts any `provider_voice_id` up to 200
   characters; Runtime refuses ids outside `[A-Za-z0-9_-]{1,128}` at session start.
5. **Default-provider assumption:** with no TTS provider set, Platform assumes Runtime's default
   `cartesia`; if that default changes, `health.DEFAULT_PROVIDERS` must change with it.

## Run history

The full logs of these runs were in earlier versions of this page and of the deleted
`latency-analysis.md`; read them with `git log -p -- docs/status.md docs/latency-analysis.md`.

- 2026-10-15 to 10-21: first real-provider runs (agent, interpreter engines, agent + translator,
  live latency benchmark).
- 2026-11-05: ElevenLabs service checks (15/15), PostgreSQL 16.2 migration and suite check,
  security review (four findings, all fixed).
- 2026-11-06: ElevenLabs STT, Gemini and Google Translation live checks; 37-recording latency
  benchmark.
- 2026-11-07: first two-phone shareable-link call.
- 2026-11-08: listen-room fix and browser latency instrument validated with synthetic audio.

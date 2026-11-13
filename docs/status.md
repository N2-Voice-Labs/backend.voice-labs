# Backend status

> **UNAI** (formerly Voice Labs) is a B2B AI communication platform: AI voice agents (primary), real-time translation, AI chatbots and developer APIs. First market: university admissions offices. First MVP: the AI Admissions Assistant for international students at Inha University. Calls run in the browser over WebRTC (LiveKit); telephone/SIP calling is deferred.

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
touches audio. UNAI’s first MVP is the Inha University AI Admissions Assistant ([Goal](goal.md)); browser voice is primary, with translated browser handoff and chat as planned channels.
`feature/voice-renovation` was merged into `main` on 2026-11-11 (PR #1).

## UNAI MVP tracker

Evidence: 🧪 = automated tests with stubbed providers; 🔌 = local run against a real Platform process; ✅ = real providers end to end (ElevenLabs STT/TTS, Gemini, Google Translate, LiveKit Cloud) driven by headless Chromium with a recorded voice as the microphone (`scripts/portal_voice_check.cjs`, `scripts/handoff_flow_check.cjs`). Local stack, SQLite; not yet a deployed server, a real phone, or a human speaker.

| Piece | Code | Evidence (2026-11-13 unless noted) |
| --- | --- | --- |
| Public demo agent sessions `POST /demo/agents/{key}/sessions`, `GET /demo/agents/{key}`, 5-min cap via reaper | Done | ✅ 10/10 consecutive AI voice calls answered (3 questions in rotation, incl. "Can I speak with an admissions officer?"). |
| AI reply latency (end-of-turn detection → first reply audio) | — | ✅ median **2.18 s** over the 10 calls (1.41–4.20 s). FAQ answers 1.4–2.2 s; escalations 2.8–4.2 s (two LLM round trips: tool call, then the spoken reply). Measured from the stored `SpeechStopped`→`AgentSpeechStarted` events: the Runtime does not emit `turn_to_first_audio` in agent mode (the final processor never sees `UserStoppedSpeakingFrame`). Measured from turn detection, not from the last syllable. |
| Session cleanup | Done | ✅ 10/10 sessions `ended` after "End call"; handoff ends the AI session; both hang-up paths end the call session. |
| Inline knowledge (≤ `PLATFORM_INLINE_KNOWLEDGE_MAX_CHARS`, default 60000) + Hangul bigram search | Done | 🧪 tests; ✅ inline (42 KB of Inha knowledge in the prompt, ~10.8k prompt tokens per turn). |
| Inha agent seed (`scripts/seed_demo_agent.py`, `agents/inha-admissions/`) | Done | 🔌 idempotent re-seeds (prompt updates in place, same agent id). |
| Inha admissions knowledge (12 files from the Spring/Fall 2027 English guidelines + apply.inha.ac.kr, retrieved 2026-11-12) | Done | 167 numbers/dates/emails automatically checked against the sources: all present. |
| 20-question FAQ eval (`evals/scripts/run_faq_eval.py`, through chat) | Done | ✅ graded by hand: **20/20 factually correct, 0 invented facts**; both unanswerable questions offer an officer (`handoff_suggested`). Strict completeness 18/20 (#14 omits "about four weeks", #17 omits the street address). The automatic `must_include` check under-counts (spoken numbers such as "five or six"). Median chat latency 1.5 s. |
| Handoff `POST /handoffs` + `/staff/handoffs` (list, accept, transcript) + `/staff` console | Done | ✅ student → "Connect to Admissions Officer" → queue → officer picks up → joined in 5.6 s. |
| Translated EN↔KO handoff call (reuses `/calls` + private listen rooms) | Done (reuse) | ✅ one 110 s call: **5 EN→KO and 6 KO→EN** translated turns; speech end → first translated audio median **844 ms** (785–917, n=11). One mistranslation from speech recognition ("아포스티유" heard as "아포스티오" → "apostolate"); no glossary is set on handoff calls. Officer hung up; ended for both. |
| Chat `POST /demo/agents/{key}/chat` | Done | 🔌 real Gemini (see FAQ eval). |
| Portal (frontend.voice-labs): `/` assistant (voice + chat), `/staff` console | Done | ✅ desktop and emulated iPhone 13 viewport (Chromium, not Safari): no horizontal overflow, voice call works. **A real phone browser is not tested yet** (needs HTTPS). |
| UNAI marketing site (frontend): Products, Solutions, Developers, Live demo, Contact | Done | Unit tests + build. |
| Deploy kit (`compose.yml`, `docs/deploy.md`, Dockerfiles, `.env.example`) | Done | `docker compose config` valid; images not built (no Docker socket access); not deployed. |

Findings from the 2026-11-13 real runs (fixed, with regression tests):

- **Voice escalation failed.** Gemini 3 requires the `thought_signature` of every replayed function call; Pipecat's OpenAI-compatible service drops it, so the follow-up request after `transfer_to_human` got 400 and the session went `failed`. The Runtime's Gemini service now adds Google's documented placeholder signature and drops the nameless phantom call Gemini's stream produces (`runtime/src/runtime/speech_providers.py`). Verified against the API (400 → 200) and in 3/3 escalation calls since.
- **Portal ended every call on the first reply.** `useEffect(() => el.scrollIntoView(...))` returned the Promise current Chromium gives back, and React called it as a cleanup ("i is not a function"), unmounting the panel. jsdom hid it; the test mock now returns a Promise.
- **Chat rate limit was 60× too strict** (minutes passed where seconds were expected): 20 per hour instead of 20 per minute.
- **Translation feedback loop on shared or speaker setups** (found in a manual test, one laptop and one mic for both people): the translation played on the speakers was transcribed again and re-translated, 12 times in 30 s. The Runtime now drops a transcript that closely matches translated speech it played in the last 10 s (`runtime/src/runtime/echo_guard.py`; short phrases under 8 letters, such as "Hello.", are never dropped; `RUNTIME_ECHO_GUARD=0` turns it off, `RUNTIME_ECHO_GUARD_SECS` sets the window). 🧪 tests, including a replay of the loop; not yet re-run with real audio.
- **Each person was locked to one language** (user feedback, 2026-11-13). Translated calls now default to `speaker_language: "auto"` (`PLATFORM_CALLS_SPEAKER_LANGUAGE`; `declared` keeps the old behaviour): each sentence's language is detected (speech recognizer, corrected by a clear Cyrillic/Latin/Hangul script majority), translated into the listener's language, or played unchanged when it is already in it. ✅ one EN/RU handoff call (headless, recorded voices): 6 sentences, both people switched language once, all 6 routed correctly (2 played unchanged, 4 translated). 🧪 tests. Known limits: very short words may be misdetected when both languages share a script; a person repeating the exact same sentence within 10 s while the other side heard it unchanged is dropped by the echo guard.
- **Demo sessions share the `/calls` per-IP limit (6 per hour by default).** A live demo where the audience shares one venue IP will hit it; raise `PLATFORM_CALLS_PER_IP_HOUR` for demo day.
- Prompt: the agent recited its own instruction ("when asking about deadlines…") and once converted won to US dollars; the prompt now forbids both (5/5 chat and 3/3 voice tuition answers in won only afterwards).

Findings from the 2026-11-12 smoke run (fixed):

- Provider records rejected `max_tokens` ("provider secrets must use secret_reference") because the key contains "token", so no LLM provider could set a reply length. Token-count keys are now exempt (`validation.NON_SECRET_KEYS`).
- `gemini-2.5-flash` and `gemini-3.5-flash` spend hidden reasoning tokens out of `max_tokens`. With 160 (voice) or 300 (chat), replies were cut off mid-sentence (`finish_reason: length`). `gemini-3.5-flash-lite` finished every reply (`stop`) in 1.15–1.43 s (n=2, direct API), so it is now the default for the Inha agent and the chat. `gemini-2.5-flash-lite` is no longer available to new API users.

## Voice agents (existing platform capability)

| Feature | Status | Notes |
| --- | --- | --- |
| Talk to an agent in a browser | ✅ 2026-11-13 | ElevenLabs STT → Gemini (`gemini-3.5-flash-lite`) → ElevenLabs TTS through LiveKit Cloud: median 2.18 s end-of-turn → first reply audio (n=10). Earlier (2026-10) Deepgram → OpenAI → Cartesia: about 1.65 s. |
| Agent + translator (English agent, Russian caller) | ✅ 2026-10-18, n=2 | About 1.9 s. |
| Agent settings: prompt, language, model, voice, temperature; templates | ✅ | |
| Tools: HTTP tools and built-ins `record_outcome`, `transfer_to_human`, `search_knowledge` | 🧪 | Timeouts and logging; mock servers only. |
| Knowledge documents | ✅ 2026-11-13 (inline) | Small knowledge bases (≤ `PLATFORM_INLINE_KNOWLEDGE_MAX_CHARS`) are inserted into the prompt; larger ones use `search_knowledge` keyword search (with Hangul bigrams, 🧪). No embeddings. |
| Interruption (barge-in) | 🧪 | Time to stop never measured. |
| Handoff intent | ✅ 2026-11-13 | `transfer_to_human` by voice (Gemini) marks the conversation escalated; the portal offers "Connect to Admissions Officer" and the officer joins through `/staff` (see the MVP tracker). |
| Outcomes, transcripts, signed events | ✅ | Each event stored exactly once (idempotent by event id). |

## Existing two-person translation capability (supports planned translated handoff)

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

## ElevenLabs credit ledger

Extra credit budget: **121k credits** (10k approved for MVP verification). Record billable runs here. Read the balance before and after each run (`GET https://api.elevenlabs.io/v1/user/subscription`); the counter lags by a few minutes, so small runs are summed. Gemini-only runs (FAQ eval, chat) are not listed.

| date | run | credits before | after | used | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-11-12/13 | test-question audio (TTS) + first debugging calls | 0 | 700 | 700 | includes 3 failed calls before the two fixes above |
| 2026-11-13 | 10 consecutive AI voice calls | 700 | 1,224 | 524 | |
| 2026-11-13 | translated handoff call (110 s, 11 turns) | 1,224 | 1,449 | 225 | plus ~250 for its two recorded tracks, inside the next row |
| 2026-11-13 | track audio, 3 tuition re-checks, phone-viewport calls | 1,449 | 1,680 | 231 | |
| 2026-11-13 | manual tests by the lead (one laptop, then phone + laptop), HTTPS tunnel check | 1,680 | 2,459 | 779 | includes the 30 s feedback loop |
| 2026-11-13 | mixed-language EN/RU handoff call (auto detection) + its tracks | 2,459 | 2,587 | 128 | track TTS mostly counted in the previous row |
| | **total** | | | **2,587** | of 10k approved |

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

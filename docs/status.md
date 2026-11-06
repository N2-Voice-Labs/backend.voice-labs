# Backend status: what we have and what is expected

Updated 2026-11-05 and reconciled against the code on the same day (first copy in
`backend.voice-labs/docs`; the original lives in `stack.voice-labs/docs/BACKEND_STATUS.md` and is
stale, see [Stale stack docs](stale-stack-docs.md)). Backend only; the web app is not covered here. For the architecture see
[System overview](../../stack.voice-labs/docs/SYSTEM_OVERVIEW.md), and for the direction see the
[North Star plan](north-star.md).

**Legend.** Three levels, each one includes the one before it. A row gets the highest level
that is actually evidenced, never the one we expect.

| Mark | Level | Meaning |
| --- | --- | --- |
| ✅ | **Real-provider-verified** | Run against the real external service (Deepgram, OpenAI, Cartesia, LiveKit, ...) and the result observed. The date and numbers are in the notes. For rows with no external provider (database, auth, API rules) it means exercised through the real code path, not only unit-mocked. |
| 🧪 | **Mock-tested** | Implemented and covered by automated tests that use fakes or mocked HTTP/WebSocket/Pipecat services. Says nothing about how the real provider behaves. |
| 🛠 | **Implemented** | Code exists but no automated test exercises the whole path (or only parts). |
| 🟡 | Partly done | Some of the row is missing. |
| ⏳ / 🔒 | Planned / blocked | Not built / waiting on something outside the code. |

**Dating rule.** A ✅ on a provider row is a historical result from the dated 2026-10 runs named in the notes (see [Translate plan](translate-plan.md): 2026-10-15 to 2026-10-21), on code that has since changed. A row without a date or number in its notes cannot be ✅ and is marked 🧪.

**Real-provider evidence so far, plainly.** On 2026-11-05 the **ElevenLabs layer** was run against
the real ElevenLabs API with a free-tier key: 15 of 15 automated checks passed (see
[ElevenLabs real-provider run](#elevenlabs-real-provider-run-2026-11-05)). **`/translations` is not
real-provider-verified**, and neither is the full two-way call with ElevenLabs (it also needs
Deepgram, OpenAI, LiveKit and two participants); those parts of the [Smoke test](smoke-test.md) are
pending. Rows verified on 2026-10-15 to 2026-10-21 were verified on
earlier code; later changes (per-listener routing, system agents, the TTS refactor) have only the
mocked test evidence unless a row says otherwise. [Smoke test](smoke-test.md) is the checklist
for closing that gap; its automated ElevenLabs part has been run (results recorded there), the
manual two-person parts have not.

## In one paragraph

Voice Labs has two backend services. The **Platform** (FastAPI, PostgreSQL) stores agents,
voices, conversations and settings, and hands each call a "session contract". The **Runtime**
(FastAPI, Pipecat, LiveKit) runs the live audio: it listens, translates, speaks and reports
events back. The Platform never touches audio. The product is a **translator between two
people** (see the [North Star](north-star.md)); start one with `POST /translations`. The same
layer can also run an AI voice agent, or an agent that translates for its caller, as a secondary
product. Phone calls are written but have not been tried with a real phone line.

## What works today

### Voice agents

| Feature | Status | Notes |
| --- | --- | --- |
| Talk to an agent in a browser (speech in, speech out) | ✅ (2026-10, undated here; stack docs say 2026-10-18) | Deepgram → OpenAI → Cartesia through LiveKit. About 1.65 s from the end of speech to the first reply audio. |
| Interruption (barge-in) in agent calls | 🧪 | The agent stops when the caller speaks (stack docs claim it was observed; no date or measurement recorded). Time to stop not measured. |
| Agent settings: prompt, language, model, voice, temperature | ✅ | Agent templates for quick starts. |
| Tools the agent can call (HTTP tools) | 🧪 | Run by Runtime with timeouts and logging; verified with mock servers. |
| Knowledge documents attached to an agent | 🧪 | Text is added to the agent's context. |
| Conversation outcomes, actions and filters | ✅ | Stored per conversation and filterable. |
| Transcripts and events | ✅ | Versioned, repeat-safe events from Runtime to Platform. |

### Live translation (interpreter mode)

| Feature | Status | Notes |
| --- | --- | --- |
| **Translation sessions** (`POST /translations`, `/invite`, `GET /translations`, `GET /translations/{id}`): two people, no agent to configure | 🧪 | Added 2026-11-05. Backed by a hidden system agent (`agents.system`, migration `0014`) that public agent routes treat as missing; always cascaded, two-way, private listening. Platform tests with a mocked Runtime only; never run with real providers or through the web app. Optional `tts_provider_id` selects the TTS provider record (so ElevenLabs is possible); without it Runtime's default (Cartesia) is used. Per-organization bound on hidden configurations (`PLATFORM_MAX_TRANSLATION_CONFIGS`, default 50). See [Contracts](contracts.md#translation-sessions-translations). |
| One speaker, translated speech back (step-by-step engine) | ✅ (2026-10-18, EN↔RU, n=4) | Deepgram → OpenAI → Cartesia. 1.47 s after speech ends. |
| Realtime engine (OpenAI `gpt-realtime-translate`) | ✅ (2026-10-18, RU→EN, n=3, 0.96 s from speech start) | One direction only, 13 output languages, provider voice. |
| Two people, each hearing only the other's translation | 🧪 current routing; historical ✅ 2026-10-18 for two speakers, Korean ↔ English, 0.8–1.4 s, before the routing was rebuilt | Per-listener routing was rebuilt after that real-provider check and has not been re-measured with real providers; no date or measurement exists for the current routing. Treat it as mock-tested. Each person has their own speech lane, and the original microphones stay private. `hear_original` lets listeners also hear the original quietly. Not re-run after the 2026-11 changes; see [Smoke test](smoke-test.md). |
| Direction: `one_way` or `auto` | ✅ | `auto` works for the languages Deepgram can detect together (English, Spanish, French, German, Hindi, Russian, Portuguese, Japanese, Italian, Dutch). Korean pairs need `one_way`. |
| Latency modes: instant, balanced, accurate | 🧪 (the 2026-10-21 benchmark used them, history only) | Different pause, chunk size and model. Instant uses the realtime engine for one-way calls. |
| Agent + translator (an English agent talking to a Russian caller) | ✅ (2026-10-18, n=2) | The agent works in one language and the caller speaks another. About 1.9 s. |
| Glossary (terms that must be translated a certain way, or kept) | 🧪 (done 2026-10-21; no real-provider run recorded here) | Per agent. "residence permit" now becomes "вид на жительство". Also sent to Deepgram to hear the terms better. Step-by-step engine only. |
| Voice cloning with consent | 🧪 | Cartesia only. Only https samples, with checks against internal addresses, a 10 MB limit and a consent record. Tested with mocks, not with a real clone. ElevenLabs voices cannot be cloned or deleted (`422 unsupported_tts_provider`, `404 unknown_provider`). |
| Voice preview | ✅ Cartesia · ✅ ElevenLabs (2026-11-05, REST layer) | Cartesia checked against the real API. ElevenLabs REST preview en/ru/ko on `eleven_flash_v2_5` with `language_code` returned audible clips (1.67 to 2.46 s, RMS about 3200 to 3800); `pcm_24000` is observed to be headerless raw 16-bit PCM; `eleven_multilingual_v2` without `language_code` works for ko. Measured through the runtime's own synthesis functions, not through the Platform `/voices/{id}/preview` route (mock-tested). Audio quality not judged. |
| Live ElevenLabs voices in calls | ✅ service layer (2026-11-05) · 🧪 inside a call | Provider `elevenlabs` in the Runtime registry (Pipecat WebSocket service built by the runtime's own factory, model `eleven_flash_v2_5` by default, no cloning). **Real:** first audio 0.2 to 0.3 s at 24 kHz, voice and language switch with reconnect (0.22 s to first audio), in-flight interruption stops audio (2 frames before, 0 after), a new utterance works on the same service, clean shutdown, invalid key fails fast as an auth error. **Not real:** use inside a full interpreter session (Deepgram, OpenAI, LiveKit, two participants), per-participant voices there, full-pipeline latency. Selected through an agent's TTS provider record or `tts_provider_id` on `/translations`; those routes are mock-tested. Voice ids are provider-specific; Platform answers `409 voice_provider_mismatch` when both the voice's and the session's providers are known and differ (skipped otherwise). See [Contracts](contracts.md#elevenlabs-live-tts). |
| Failure-apology clips rendered with ElevenLabs (`render_apologies.py --elevenlabs --confirm-billable`) | 🧪 | Guards (no request without the confirm flag, key, a voice per language, a valid rate) tested; no real render made. The committed clips are macOS placeholders. |

### Phone calls (SIP)

| Feature | Status | Notes |
| --- | --- | --- |
| Outbound calls, campaigns with targets, dial-next | 🧪 | Behind `LIVEKIT_SIP_TRUNK_ID`. Mocks only. |
| Bridge call and transfer to another number | 🧪 | Bridge needs the step-by-step engine with `auto`. |
| Phone ↔ browser and phone ↔ phone interpreter | 🔒 | Needs a SIP provider account (Twilio or Telnyx) and a phone number. |

### Platform API and safety

| Feature | Status | Notes |
| --- | --- | --- |
| Agents, providers, voices, deployments, conversations | ✅ PostgreSQL 16.2 (2026-11-05) · 🧪 tests on SQLite | Migrations `0001` to `0014` (organizations `0008`-`0010`, voice tombstones, problem reports, session liveness, system agents) ran on a real PostgreSQL 16.2; details in [PostgreSQL check](#postgresql-check-2026-11-05). |
| Authentication: superadmin keys (`PLATFORM_API_KEYS`), organization keys (`vlk_…`) with roles, per-session tokens (`vls_…`) | ✅ | Off when no keys are set, so local development is unchanged. Keys and tokens are stored hashed. Every row belongs to one organization; see [Contracts](contracts.md#authentication). |
| Runtime authentication (`X-Runtime-Token`) | ✅ | Platform and Runtime only accept each other. |
| CORS for the web app, localhost and the deployed site | ✅ | `PLATFORM_WEB_ORIGIN` takes a comma-separated list. |
| Analytics summary: conversations, languages, p50/p95 latency, timeline per engine | 🧪 | `GET /analytics/summary`. |
| `/docs` and `/openapi.json` stay public when auth is on | 🟡 | Decision needed. |

### Provider flexibility

| Slot | Today | Swappable? |
| --- | --- | --- |
| Speech recognition (STT) | Deepgram nova-3 | 🧪 registry; a second provider is NS-07 |
| Language model | OpenAI (gpt-4.1, mini, nano) and OpenAI-compatible APIs | 🧪 registry |
| Voice output (TTS) | Cartesia sonic-3.6 (default) · ElevenLabs (`elevenlabs`) | 🧪 registry. ElevenLabs is wired into live calls, preview and the apology renderer; its service layer was verified against the real API on 2026-11-05 (above), the apology renderer and in-call use are mock-tested only (NS-07 code done, full real verification open). Selectable through `/translations` by `tts_provider_id` |
| Translation engine | step-by-step, realtime | 🧪 registry (both engines ✅ individually above) |
| Realtime translation | OpenAI `gpt-realtime-translate` | 🧪 same registry; Qwen is NS-07 |

The Cartesia key and voice fields keep their `cartesia_*` / `CARTESIA_*` names on purpose
(stored contracts and constructors do not change); ElevenLabs has its own `ELEVENLABS_API_KEY` /
`ELEVENLABS_VOICE_ID`, and `PipelineConfig.tts_api_key` / `tts_voice_id` follow the selected TTS
provider. Cloning and deletion are Cartesia-only. Voice preview works for both. Still missing for
ElevenLabs end to end: a full two-way call with real Deepgram and OpenAI. Voice/provider compatibility is checked only when
both providers are known to Platform.

### Measuring and testing

| Feature | Status | Notes |
| --- | --- | --- |
| One timeline per translated utterance (speech start, voice detected, first transcript, translation start, first audio, stable playback) | ✅ (2026-10-21 live runs) | Emitted as `utterance_timeline` events. |
| Evaluation set: 140 synthetic items (EN↔KO, RU↔EN, mixed languages, names and numbers, domain terms, phone quality, interruptions) | ✅ | The reference translations still need a native Korean and Russian reviewer. |
| Quality scoring: word error rate, Korean character error rate, chrF, term recall | ✅ | `evals/`. |
| Live latency runner (`live_latency`) | ✅ (2026-10-21, 30 sentences per row) | One command, p50/p95 per engine, mode and language pair, with a "where the time goes" table. |
| Tests | see below | Fresh results 2026-11-05 in [Test results](#test-results-2026-11-05). Earlier counts in the stack docs and in older versions of this page are out of date. |

## Test results (2026-11-05)

Final runs on 2026-11-05 on branch `feature/voice-renovation` (uncommitted), in an environment built
**from the lockfile**: `UV_PROJECT_ENVIRONMENT=$S/lockenv uv sync --locked --all-packages --all-extras`
run from the repository root (uv 0.13.7; it succeeded, so the regenerated `uv.lock` files are consistent
with the `pyproject.toml` files; resolved fastapi 0.143.0, starlette 1.7.0, httpx 0.28.1). `$S` is the
session scratchpad directory, outside the repository. An earlier set of runs used a plain `pip` venv with
newer, unpinned FastAPI/Starlette; the counts matched.

| Suite | Command | Result |
| --- | --- | --- |
| Platform | `cd platform && $S/lockenv/bin/python -m pytest -q -p no:cacheprovider -W ignore` | 734 passed, 1 skipped |
| Runtime | `cd runtime && $S/lockenv/bin/python -m pytest -q -p no:cacheprovider -W ignore` | 866 passed, 6 skipped |
| Evals | `cd evals && $S/lockenv/bin/python -m pytest -q -p no:cacheprovider -W ignore` | 137 passed |
| Stack acceptance, before the fix | `cd runtime && $S/lockenv/bin/python -m pytest -q -p no:cacheprovider -W ignore ../../stack.voice-labs/tests` | 1 failed, 5 passed, 3 skipped |
| Stack acceptance, after the fix in [stack-acceptance-patch.md](stack-acceptance-patch.md) (applied in place, uncommitted, in `stack.voice-labs/tests/test_voice_platform_acceptance.py`) | same command | 6 passed, 3 skipped, 0 failed |

- **The one acceptance failure is a stale test, not a regression.** It fails identically on a pristine
  export of `HEAD`. Cause (confirmed by capturing the exception): the test stubs HTTP by replacing
  `platform_main.httpx`, but the session routes live in `sessions.py`, which has its own `import httpx`,
  so the Runtime handoff does a real DNS lookup of `runtime.test` and Platform answers 503
  `runtime_unavailable`. The fix is three lines in a sibling-repo test file; it was applied in place on 2026-11-05 with the
  user's authorization and is uncommitted (see the patch page).
- **Skipped by design or by environment:** the two LiveKit tests need a local LiveKit server (Docker's
  socket is permission-denied for this user, so none could be started); the real-provider test is opt-in
  and billable.

- All of these are mock-level evidence. The tests build the schema with `create_all`, so they
  do **not** exercise migration `0014` itself. A migration round trip (upgrade, downgrade,
  upgrade) was run on SQLite and, later the same day, on a real PostgreSQL 16.2; see
  [PostgreSQL check](#postgresql-check-2026-11-05).
- `uv` rewrote the lock `revision` 3 to 5 (produced with uv 0.13.7). Contributors and CI need a uv
  that understands revision 5; an older uv may refuse the lock or re-resolve it. The minimum version
  is not verified.
- The earlier stack-acceptance failure was `test_platform_hands_off_deferred_session_and_recovers_runtime_state_loss`
  (`ensure-running` returned 503). It failed identically on a pristine export of `HEAD` (a stale test,
  diagnosed above) and passes after the patch.
- The LiveKit-dependent tests are skipped because Docker's socket was permission-denied for the
  test user, so no local LiveKit could be started. The real-provider tests are skipped by design
  (opt-in, billable).
- Skips: Platform skips its PostgreSQL concurrency test without `PLATFORM_TEST_POSTGRES_URL`.

## ElevenLabs real-provider run (2026-11-05)

Run by the lead with the user's authorization, using `runtime/scripts/elevenlabs_smoke.py` against the
real ElevenLabs API with a **free-tier** key. 15 checks, **15 passed**, 429 characters used in
total across all runs (read from the account afterwards; free tier, 10,000-character limit). The key expires; it is not stored in the repository. Only
objective measurements were made (durations, levels, timings): **audio quality was not judged by
anyone**; sample WAVs were saved to an ephemeral scratch directory for the user to listen to.
Full table: [Smoke test, section 11](smoke-test.md#11-record-results-here).

| What | Observed (2026-11-05) |
| --- | --- |
| Key and account | valid, tier `free`; 21 voices available |
| REST preview, `eleven_flash_v2_5` with `language_code` | en 1.67 s, ru 2.46 s, ko 2.32 s of audio; RMS 3189 / 3267 / 3846; not silent; request about 0.35 to 0.4 s |
| `pcm_24000` body | headerless raw 16-bit PCM (first bytes `FE FF 00 00`, not `RIFF`; 80,248 bytes for 1.67 s). Previously an unverified assumption; now **observed, not documented** |
| `eleven_multilingual_v2` without `language_code`, ko | works (2.18 s of audio) |
| Live websocket via the runtime's own `build_tts` factory | first audio 0.29 s, 24 kHz |
| Voice and language switch (voice B, ru) | reconnects and delivers audio; first audio 0.22 s including the reconnect |
| Interruption of an open (in-flight) context | audio had started (2 frames before), 0 frames after 0.5 s; a new utterance worked on the same service (first audio 0.17 s) |
| Shutdown | `EndFrame` ends the run and the socket is cleared |
| Invalid key, REST | `401` in about 0.2 s |
| Invalid key, websocket | **ElevenLabs accepts the handshake and then closes with `1008 (policy violation) Invalid API key`** (see the defect below) |

`pcm_*` is evidently accepted on the websocket too (audio was produced at 24 kHz through Pipecat
using it): observed working, not documented.

**Defect found by this run, and fixed.** Before the fix, an invalid or expired key made Pipecat
reconnect three times; Voice Labs ended the call after about 3.3 s classified as `other` (after a
first non-fatal "completed with no audio" note at about 1.2 s). Fix: `provider_failures.classify_error`
maps a `ConnectionClosed` whose text has credential wording to `auth` (`OSError` and
`PermissionError` are excluded), and `speech_providers._elevenlabs_service_class` subclasses Pipecat's
service so an auth-classified close is reported at once as a permanent error. Measured afterwards
against the real API through `watch_provider_failures`: one report `('tts', 'elevenlabs', 'auth',
fatal=True)` at about 1.3 s. An expired key mid-deployment now fails the call fast with an auth
`ProviderError` instead of silent audio loss. Regression tests:
`test_in_band_key_rejection_ends_the_session_at_once_as_an_auth_failure` (with `RejectingSocket`) in
`runtime/tests/test_elevenlabs_tts.py` and the classify tests in `runtime/tests/test_provider_failures.py`.
The older mocked test `test_rejected_api_key_on_connect_...` models a handshake `401` (a REST-style
rejection) that the real websocket does **not** do; it is kept, but it is not the observed behaviour.
*Report-file note:* the saved JSON report records the invalid-key websocket check as classified `auth`
with `fatal: False` at 12.07 s, i.e. the classification before the service-class fix; the `fatal=True`
at about 1.3 s figure is the lead's later measurement and is not in that file.

**Still not real-verified (pending):** the full two-way `/translations` flow (needs Deepgram and
OpenAI keys, LiveKit and two participants); per-participant voices inside a real interpreter session;
latency of the full pipeline; audio quality (unjudged); PostgreSQL behind a production-like server; whether an ElevenLabs voice id is
scoped to the account (across accounts); behaviour beyond the free tier and paid-tier limits; close
codes other than `1008`.

## PostgreSQL check (2026-11-05)

Run by the lead on a real **PostgreSQL 16.2** (embedded through the pip package `pgserver` 0.1.4 in a
scratch directory outside the repository; no Docker; driver psycopg 3 from the lock-pinned environment).
Recipe: [Smoke test, PostgreSQL check](smoke-test.md#12-postgresql-check).

| Check | Result |
| --- | --- |
| `alembic upgrade head` from an empty database | the full chain `0001` to `0014` ran, rc 0 |
| `alembic downgrade 0013_session_last_event`, then `upgrade head` | succeeded; `alembic current` = `0014_system_agents` (head) |
| Resulting schema (queried) | `agents.system` boolean NOT NULL DEFAULT false; `agents.system_key` varchar NULL; constraint `uq_agents_org_system_key` UNIQUE (`organization_id`, `system_key`) |
| `platform/tests/test_postgres_concurrency.py` (skipped unless `PLATFORM_TEST_POSTGRES_URL` is set) against the migrated database | 1 passed |
| Translation tests (`test_translations.py`, `_hardening.py`, `_safety.py`) on PostgreSQL | 52 passed (they wrote there: 41 hidden agents, 67 sessions). `test_system_agent_race_real_threads_file_sqlite` uses its own file-backed SQLite engine, so it is **not** a PostgreSQL race test |
| Scratch concurrency script through the real FastAPI app: 12 threads released by a barrier POST `/translations` for the same brand-new configuration | all 12 returned 201; exactly 1 new hidden agent; 12 distinct new sessions; 0 thread errors |
| Refused request (unknown `voice_profile_id`, `409 selected_voice_unavailable`) | 0 new hidden agents and 0 new sessions: the flush-not-commit rollback works on PostgreSQL |
| Full Platform suite on PostgreSQL (fresh database, schema by `create_all`) | **717 passed, 17 failed, 1 skipped** (251 s) |

The scratch script is not committed; the method is the table row above (barrier-released threads, one
shared new configuration, then counting rows).

**The 17 failures are pre-existing.** The identical 17 tests fail on a pristine export of `HEAD` under
PostgreSQL (the two failing-test lists have an empty difference both ways): 14 in
`tests/test_sip_dialing.py`, 2 in `tests/test_auth_orgs.py`, 1 in `tests/test_tenancy_isolation.py` (an
audit-log `before=` ISO timestamp whose `+` is not URL-escaped, so `02:23:17+09:00` becomes a space and
the API returns 422). They are test-suite issues under PostgreSQL, not caused by this branch.

**Diagnosed (2026-11-05, by reading the failing tracebacks and test code; the tests were not changed).**
All 17 are test-fixture artifacts, not production defects. SQLite does not enforce foreign keys by
default and PostgreSQL does:
- 14 in `tests/test_sip_dialing.py`: the test double for `open_voice_session` (`tests/test_sip_dialing.py`
  line 138) returns `"conversation_id": str(uuid4())` with no `conversations` row, so
  `campaigns.process_target` hits `campaign_targets_conversation_id_fkey` on commit. The production code
  receives the id it is given.
- 2 in `tests/test_auth_orgs.py` (lines 338-347 and 364-373): the tests insert a `VoiceSession` whose
  `conversation_id` is a random uuid with no `conversations` row (`voice_sessions_conversation_id_fkey`).
- 1 in `tests/test_tenancy_isolation.py`: the unescaped `+` described above.
**Fixed (2026-11-05), test code only:** the SIP double now creates a real `Conversation` row; the two
`test_auth_orgs.py` tests create an agent and conversation first (`make_conversation`); the audit cursor is
URL-encoded in `test_tenancy_isolation.py`. Re-run on a fresh PostgreSQL 16.2 database: the **whole
Platform suite passes, 734 passed, 1 skipped** (219 s), and 734 passed, 1 skipped on SQLite (118 s).

**Still not verified on PostgreSQL:** a separate production-like server (pooling, replicas); lock
contention while the provider preflight is awaited (D3); long-running soak.

## Security review (2026-11-05)

An independent review (an Opus 5.5 agent) of this branch's changes found the following; the lead
fixed all four and added mock tests.

| Finding | Fix |
| --- | --- |
| Preview path traversal: a tenant-supplied voice id such as `../../v1/user` is interpolated into the ElevenLabs URL path (confirmed with httpx to collapse to another endpoint, with the API key header attached) | Voice ids must fullmatch `^[A-Za-z0-9_-]{1,128}$` in `/tts/preview`, the apology renderer, the live factory, and for every participant voice in Runtime's `ParticipantDirectory` (`422 invalid_participant_voice`, Cartesia too) |
| Orphan and unbounded hidden agents: a refused `/translations` request left a committed agent; the number of configurations per organization was unlimited | Agent is flushed, committed together with the session; rolling 24 h bound on NEW configurations per organization (`PLATFORM_MAX_TRANSLATION_CONFIGS`, default 50, `0` disables) with `429 too_many_translation_configs`; glossary order canonicalised for the hash |
| Voice/provider mismatch accepted silently | `409 voice_provider_mismatch`; falls back to Runtime's default provider (`cartesia`) when no TTS provider is set; skipped when the voice has no `tts_provider_id` or its provider row is unresolvable |
| Fail-open TTS credentials: a TTS provider without declared credential fields could start without a key | `register_tts_credentials`: an unmapped provider fails at session start |

**Still untested** (the rollback on refusal and the same-configuration race have since run on
PostgreSQL; see [PostgreSQL check](#postgresql-check-2026-11-05)). Not measured: the configuration bound
under concurrent creators (it can overshoot by design), lock contention while the provider preflight is
awaited, a separate production-like server (pooling, replicas), soak runs; ElevenLabs
behaviour mid-call beyond the short checks of 2026-11-05 (network drops during a long call, quota
exhaustion, paid-tier limits); WebSocket close codes other than the observed `1008` for an invalid
key. (Headerless `pcm_24000` and the invalid-key behaviour were settled by the real run below.)

## Latest live measurements (2026-10-21, 30 sentences per row, one speaker)

*History, not current performance.* These were measured on 2026-10-21 with Cartesia, before the
per-listener routing, system-agent, and TTS-provider changes of 2026-11. They do not apply to the
new code and nothing in this table involves ElevenLabs. Re-measure with the same command after the
[Smoke test](smoke-test.md).

Times are seconds after the speaker started, typical case (p50) and slow case (p95). Sentences
are about 3.3 s long, so step-by-step times include the speech itself.

| Engine and mode | Pair | Translation heard, p50 / p95 | After speech ends, p50 / p95 |
| --- | --- | --- | --- |
| Realtime, instant | KO→EN | 3.98 / 5.16 | 0.67 / 1.79 |
| Realtime, instant | EN→KO | 3.87 / 5.15 | 0.31 / 1.93 |
| Realtime, instant | RU→EN | 4.17 / 5.88 | 0.90 / 1.96 |
| Realtime, instant | EN→RU | 4.10 / 6.21 | 0.98 / 3.08 |
| Step-by-step, instant | KO→EN | 5.24 / 6.19 | 2.02 / 2.48 |
| Step-by-step, balanced | KO→EN | 4.86 / 6.42 | 1.96 / 2.45 |
| Step-by-step, accurate | KO→EN | 5.71 / 6.67 | 2.28 / 2.70 |

Other pairs are within about half a second of these. Full tables are in the run reports.

**What this tells us**

- Nothing meets the targets yet (realtime ≤ 1.2 s from speech start, step-by-step ≤ 1.5 s
  after speech ends). Korean is not specifically worse than Russian.
- **Step-by-step:** the translation request starts about 0.3–0.5 s after the speaker
  finishes, and the first audio arrives about 1.0–1.3 s after that. More time goes to
  waiting for the end of the sentence and to translation plus voice than to recognising speech.
- **Realtime:** the first audio arrives at roughly the end of the sentence, not during it. An
  earlier hand test reached 0.79 s from speech start. The two disagree, and the difference
  (speech style, how the audio is paced, or the model's behaviour) is not explained yet. Find
  out why before tuning anything.
- **Accurate mode lost most Russian sentences:** 26 of 30 for RU→EN and all 30 for EN→RU
  were never heard. The cause is not found. It is a real bug or limit to investigate, not a
  result.
- These came from synthetic voices played into a LiveKit room on this machine, so they include
  local network and playout time. Real microphones and phone audio will differ.

## What is expected next

Order follows the [North Star plan](north-star.md). The first proof is a Korean speaker
in a browser and an English speaker on a phone, talking to each other live.

| Next | What it adds | Needs |
| --- | --- | --- |
| **Investigate the benchmark** | Explain the realtime gap and the accurate-mode misses; rerun with the same command. | nothing |
| **NS-02 Interruption policy** | When someone keeps talking, the old translation finishes and the new one queues. When the listener starts talking, the translation they hear gets quieter instead of stopping. Measure time to stop in agent calls. | nothing |
| **NS-03 Audio quality** | Check recorded translations for gaps, clipping and loudness. | nothing (recordings already saved) |
| **NS-04 Provider failure handling** | If realtime fails, fall back to step-by-step; reconnect dropped speech or voice connections; end a direction cleanly instead of hanging. | nothing |
| **NS-05 Rejoin** | Someone who drops and returns within 60 s keeps their settings and routing. | nothing |
| **NS-07 Second providers** (ElevenLabs layer verified 2026-11-05, in-call use pending) | ElevenLabs live TTS, preview and apology renderer are implemented with the Pipecat extra and regenerated locks; the service layer passed 15 of 15 real checks; **open:** the full two-way call ([Smoke test](smoke-test.md)). Still planned: Qwen for translation and realtime, one more speech-recognition provider with strong Korean. | API keys |
| **RM-013 → RM-020 Phone** | SIP trunk, then browser ↔ phone interpreter (the first proof), then phone ↔ phone. | SIP account and a number |
| **RM-022 Provider benchmark** | Same call with each provider mix; compare speed, quality, cost per minute. Pick the default stack. | NS-01, NS-03, NS-07 |
| **Human ↔ AI calls, AI phone calls, an AI joining a two-person call** | The same layer used by agents. | phone line working |
| **Later:** mid-sentence translation stability, translation in the speaker's own voice, live self-cloning | Research-level features. | consent flows, provider support |

## Open decisions and risks

1. **Phone proof is blocked** until a SIP account and number exist.
2. **The web app must play any subscribed audio track** (`translation-for-<identity>`), not
   only `pipecat-audio`, or two-person calls are silent in the browser.
3. **API keys were shared in chat**: rotate them.
4. **Reference translations** in the evaluation set need a native Korean and Russian reviewer
   before the quality scores can be trusted.
5. **Realtime and glossary:** the realtime engine cannot use a glossary, so the terms feature
   works only with the step-by-step engine.
6. **Voice cloning and SIP** have only been tested with mocks.
7. **Join-time leak:** a participant can hear up to about 0.1 s of another's original voice
   when joining, and a client that subscribes by itself can override server-side routing.
8. **This branch's work is uncommitted** (2026-11-05): the modified and untracked files under
   `platform/`, `runtime/` and `docs/` are not in a commit.
9. **Voice/provider check is conservative:** `409 voice_provider_mismatch` is skipped when the
   voice has no `tts_provider_id` or its provider row cannot be resolved; with no TTS provider set
   it compares against Runtime's default (`cartesia`).
10. **The two-person real-provider smoke test has never been run** for `/translations` (the automated ElevenLabs-layer script has);
    see [Smoke test](smoke-test.md).
11. **Hidden-agent row lock (D3):** the flushed hidden-agent row stays locked until
    `open_voice_session` commits, which first awaits the provider preflight (HTTP timeout up to about
    5 s). Concurrent creators of the same new configuration can therefore block on PostgreSQL and
    take SQLite's write lock. On 2026-11-05, 12 concurrent creators of one new configuration on a real
    PostgreSQL all succeeded with one hidden agent, but the blocking time while the preflight is awaited
    was not measured (the test had no `RUNTIME_API_URL`, so no preflight ran).
12. **Voice id patterns disagree (D4):** Platform accepts any `provider_voice_id` up to 200
    characters (no pattern, `voices.py`) while Runtime refuses ids outside `[A-Za-z0-9_-]{1,128}`.
    Such a voice saves fine and then fails at session start with `422 invalid_voice_id` (Runtime
    `POST /sessions`, ElevenLabs) or `422 invalid_participant_voice` (participant add/update).
13. **Default-provider assumption:** with no TTS provider set, Platform assumes Runtime speaks
    with its default `cartesia`, so an ElevenLabs-registered voice there is
    `409 voice_provider_mismatch`. If Runtime's default ever changes, `health.DEFAULT_PROVIDERS`
    must change with it.

## Open source and services we use

| Part | What we use |
| --- | --- |
| Realtime audio and rooms | LiveKit (self-hosted server, SIP, server API) |
| Audio pipeline | Pipecat 1.12 |
| Voice activity detection | Silero VAD |
| Speech recognition | Deepgram nova-3 |
| Language model and translation | OpenAI gpt-4.1 family, `gpt-realtime-translate` |
| Voice output and cloning | Cartesia sonic-3.6 (cloning); ElevenLabs via Pipecat (voice output only; service layer verified 2026-11-05) |
| API and data | FastAPI, SQLAlchemy, Alembic, PostgreSQL, Pydantic |
| Docs | Zensical (MkDocs-compatible), built with `--strict` |

## Two-person browser workflow: release readiness (2026-11-05)

- **Invite recovery:** idempotent by `Idempotency-Key`; an invite never displaces a seated guest
  unless `replace: true` is sent. Covered by Platform tests (SQLite).
- **Private listening** is on by default; with `PLATFORM_PRIVATE_LISTENING=0` a session that hides
  the original microphones (`hear_original=false`) is refused (`409 private_listening_required`).
- **PostgreSQL:** concurrent `POST /translations` create one hidden configuration; a slow provider
  check no longer blocks the API (`tests/test_postgres_translations.py`).
- **Browser audio: BLOCKED.** This repository has no web client and nothing here joins
  `connection.listen`. The Korean to English and English to Korean call in two browsers with real
  providers (zero original/self audio on join and rejoin, both translations audible) has not been run.
  Until it passes, do not claim the workflow is verified end to end. Note that a client which ignores
  `connection.listen` gets a session that starts but plays no translation.
- **SIP is out of the release scope.** The phone path (`/voice/sessions/{id}/phone-participants`) still
  tells the Runtime before dialling and withdraws best-effort on a failed dial; a failed withdraw can
  leave Platform and Runtime disagreeing. Tracked separately; make no SIP claims for this release.

## Provider stack: ElevenLabs STT + Gemini translation + ElevenLabs voice (2026-11-06)

- **Added:** `elevenlabs` STT provider (Pipecat `ElevenLabsRealtimeSTTService`, fast auth failure),
  `gemini` llm provider and translator (OpenAI-compatible endpoint, own base URL), `stt_provider_id`
  and `llm_provider_id` on `POST /translations`, per-provider credential requirements, provider
  health for `stt/elevenlabs` and `llm/gemini`. Deepgram, OpenAI and Cartesia remain the defaults.
- **No model is locked in code.** Gemini needs a configured model; ElevenLabs uses the SDK default
  unless the record sets one.
- **Verified:** Runtime and Platform tests with mocks only (see contracts for the list).
  Checked offline against the real Pipecat class: the language codes map correctly
  (`ko` to `kor`, `en` to `eng`, `multi` to automatic detection).
  **Live checks, 2026-11-06 (free ElevenLabs account, one Gemini key):** the ElevenLabs realtime STT
  WebSocket accepted the key and committed correct Korean and English transcripts of speech
  synthesized by ElevenLabs TTS (about 0.25 s from commit to final text); Gemini streamed
  Korean to English translations through the OpenAI-compatible endpoint with `max_tokens` honoured,
  using `gemini-3.5-flash-lite` (`gemini-2.5-flash-lite` is no longer available to new accounts,
  which is why no model is built in). These ran the Gemini translator class directly and the
  ElevenLabs endpoints with a raw WebSocket, not the full Runtime pipeline. Not verified: the
  pipeline with these providers inside a LiveKit session, translation quality and end-to-end
  timing in a real call, and anything with the web app. Waiting on LiveKit and the web app.
- **Repeatable check:** `cd runtime && python scripts/stack_smoke.py --confirm-billable` runs
  Korean to English and English to Korean through ElevenLabs STT, the Runtime's Gemini translator and
  ElevenLabs voice, with timings per stage (no LiveKit needed; reads `.env` and never prints keys).
  First run 2026-11-06: both directions returned correct text at every stage.
  Gemini time to first text (a 14-token Korean translation, from the dev machine, one key that
  appears to be on a free tier): `gemini-3.5-flash-lite` typically 0.8 s to 1.6 s (median about 1 s),
  the same through the OpenAI-compatible endpoint and Google's native API, with no thinking tokens
  reported; occasional spikes of 10 s to 12 s and some 429/503 answers. `gemini-3.1-flash-lite`
  about 2.9 s; `gemini-3.5-flash` 10 s to 13 s because it spends about 60 thinking tokens first.
  `reasoning_effort="none"` is rejected (400) by the compatible endpoint. About 6 requests per
  model, so treat these as rough; a paid tier or another translation model may behave differently.

## Google Cloud Translation option

`llm` provider `google-translate` (key `GOOGLE_TRANSLATE_API_KEY`) swaps the LLM translator for the
classic machine-translation API. Live check (Korean and English sentences, 3 repeats each): about 0.15 to
0.5 s per sentence against about 1 s to first text for Gemini lite, with usable quality. Trade-offs: the
whole sentence arrives at once (no clause-by-clause speech start), and conversation history and glossary
are not used. Translation sessions only; selecting it for an agent LLM fails at session start. Not yet
measured inside a real call (waiting for LiveKit).

Backup: when `GEMINI_API_KEY` and a model (`provider_settings.llm.translation_model` or
`GEMINI_TRANSLATION_MODEL`) are set, `google-translate` is wrapped in a `FallbackTranslator`. The LLM
translates sentences that need the glossary, and any sentence where Google fails or returns nothing; after
a failure Google is skipped for 30 s. Without a model nothing is guessed and Google runs alone.

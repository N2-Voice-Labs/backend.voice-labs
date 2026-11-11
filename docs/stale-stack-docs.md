# Stale pages in `stack.voice-labs/docs`

Written 2026-11-05. **`stack.voice-labs/docs` was not modified** and neither was `frontend.voice-labs`. The one change in the
`stack.voice-labs` repository is the stale acceptance test
`tests/test_voice_platform_acceptance.py`, patched in place on 2026-11-05 with the user's
authorization and uncommitted (see [Stack acceptance patch](stack-acceptance-patch.md)). This page only
lists what should change in the stack docs, with text you can paste. The maintained backend docs are
the pages next to this one: [Contracts](contracts.md), [Status](status.md),
[North Star](north-star.md), [Translate plan](translate-plan.md), [Smoke test](smoke-test.md).

I read each page named below and checked the claims that matter against the code in this
repository (migrations, `sessions.py`, `translations.py`, `speech_providers.py`, `pipeline.py`,
`auth.py`). "Stale" means a statement contradicts today's code or docs; a page that is merely
older but not wrong is marked **history** or **current**.

## Summary

| Page (in `stack.voice-labs/docs`) | Verdict | Action |
| --- | --- | --- |
| `contracts.md` | **Superseded** by [Contracts](contracts.md) | Replace with pointer |
| `BACKEND_STATUS.md` | **Superseded** by [Status](status.md) | Replace with pointer |
| `NORTH_STAR_PLAN.md` | **Superseded** by [North Star](north-star.md) | Replace with pointer |
| `BACKEND_TRANSLATE_PLAN.md` | **Superseded** by [Translate plan](translate-plan.md) | Replace with pointer |
| `SYSTEM_OVERVIEW.md` | Partly stale | Edit the listed paragraphs |
| `index.md` | Partly stale | Edit the listed sections |
| `concepts.md` | Partly stale | Edit the listed paragraphs |
| `quickstart-voice.md` | Partly stale | Edit the listed paragraphs |
| `development.md` | Partly stale | Edit the listed paragraphs |
| `run-locally.md` | Mostly current | Add ElevenLabs and `/translations` notes |
| `troubleshooting.md` | Mostly current | One paragraph |
| `BACKEND_ARCHITECTURE.md` | **History** (proposal) | Add a banner |
| `BACKEND_TASKS.md` | **History** (first-milestone backlog) | Add a banner |
| `quickstart-api.md` | Current for what it covers | Optional note |
| `evaluation.md` | Current as dated history | Optional note |
| `.env.example` (repo root) | Missing keys | Apply the patch below |

## Fully superseded pages: replace with a pointer

Keep the file (so links do not break), delete the body, and paste the matching text. The
backend copies are the maintained ones; the old text stays in git history.

### `contracts.md`

Why: it predates organization keys and session tokens (`vlk_`, `vls_`, roles, per-organization
scoping), the whole `/translations` API, hidden system agents, and the ElevenLabs provider. It
still says authentication is "admin API keys" only (`PLATFORM_API_KEYS`) and lists session-token
routes without `/translations/{id}`, `/invite` and `phone-participants`. About 290 lines differ
from the maintained copy.

```markdown
# Contracts

This page moved. The API and event contracts are maintained next to the code in
`backend.voice-labs/docs/contracts.md` and are updated in the same commit as the route they
describe. The copy that used to be here was stale (it predated organizations, session tokens,
translation sessions and ElevenLabs).

Start there for: authentication, voices, deployments, interpreter mode, translation sessions
(`/translations`), ElevenLabs (live TTS), glossary.
```

### `BACKEND_STATUS.md`

Why: dated 2026-10-22. It predates `POST /translations`, authentication (it does not know
organization keys), ElevenLabs, per-listener routing status, and the 2026-11-05 test results.

```markdown
# Backend status

This page moved. Current status, with a legend that separates implemented, mock-tested and
real-provider-verified, is in `backend.voice-labs/docs/status.md`. The 2026-10-22 version that
used to be here is stale.
```

### `NORTH_STAR_PLAN.md`

Why: status line "NS-06, NS-08 and NS-01 (runner) done" with no `/translations` product
surface section and the old NS-07 text ("live calls are not wired").

```markdown
# North Star plan

This page moved to `backend.voice-labs/docs/north-star.md` (updated 2026-11-05). Edit it there.
```

### `BACKEND_TRANSLATE_PLAN.md`

Why: it stops at stage 3b; it has no stage 3c (`TR-050`, `TR-051`).

```markdown
# Backend Milestone 10: live translation

This page moved to `backend.voice-labs/docs/translate-plan.md` (updated 2026-11-05, includes
stage 3c: translation sessions). Edit it there.
```

Pointer pages have no headings, so any link elsewhere in the stack docs of the form
`contracts.md#authentication` (or another anchor) will break a `--strict` docs build. Repoint
those links to the backend copy (`backend.voice-labs/docs/contracts.md#authentication`).

## Partly stale pages: exact changes

### `SYSTEM_OVERVIEW.md` (header says "Status as of 2026-10-18")

| Where | What is wrong | Replace with |
| --- | --- | --- |
| First line "Status as of 2026-10-18." | Out of date | "Status as of 2026-11-05. Current status lives in `backend.voice-labs/docs/status.md`." |
| "Voices" table, "Voice preview" row: "Type text, hear the voice (Cartesia)" | Preview also works for ElevenLabs (real API checked 2026-11-05 at the Runtime level) | "Type text, hear the voice (Cartesia and ElevenLabs verified against the real APIs)" |
| "Live translation" table | No row for the product entry point | Add: `Translation sessions (POST /translations)` , 🧪 , "Two people, no agent to configure; hidden system agent; Cartesia by default; never run with real providers" |
| Alembic row "Database migrations (`0001`–`0006`)" | Chain now ends at `0014_system_agents` | "(`0001`–`0014`)" |
| "External services" table, Cartesia row | ElevenLabs missing | Add row: "ElevenLabs, text-to-speech (live, preview), `eleven_flash_v2_5` by default, service layer verified against the real API on 2026-11-05 (15 of 15 checks), in-call use not yet verified, no cloning" |
| "Quality and testing" table: "Platform 235 passed / Runtime 293 passed" | Out of date | Link to `status.md#test-results-2026-11-05` (Platform 734 passed, 1 skipped; Runtime 858 passed, 6 skipped; evals 137 passed on 2026-11-05) |
| "Not done yet": "User authentication and authorization (required before any public deployment)" | Authentication exists: superadmin keys, organization keys with roles, session tokens, per-organization scoping | Delete the bullet, or reword to "Login/user accounts for the web app" |
| "Not done yet": "Per-listener audio routing in two-person calls (today everyone hears all audio)" | Per-listener private listening exists in code (RM-001/002) | Delete the bullet |
| Features table "User accounts and login: ⏳" | Partly done (API keys and roles); no login UI | Reword to "Login UI ⏳; API keys with roles ✅" |
| First paragraph: "You create a voice, give an agent instructions ..." and the diagram's "interpreter" as an agent | Translation is now the primary product surface | Add one sentence: "The primary product is translation between two people (`POST /translations`); agents are secondary." |

### `index.md`

| Where | What is wrong | Replace with |
| --- | --- | --- |
| "Current status": "Last progress update: 2026-10-14", the Verified / Not yet verified / Deferred lists | Several "Deferred" items exist now: authentication and multi-tenancy, voice previews, outbound dialing, SIP transfer, sending tools and knowledge to Runtime | Replace the whole section with: "Current status: see `backend.voice-labs/docs/status.md` (3-level legend: implemented, mock-tested, real-provider-verified)." |
| "Initial language work covers English, Russian ... Korean ... deferred" | Korean ↔ English two-person mode was verified on 2026-10-18, before per-listener routing was rebuilt; the current routing is mock-tested only | "English, Russian and Korean are supported; see `status.md` for what was verified and when." |
| "Start here" table | No link to the backend docs | Add rows: "Backend docs (contracts, status, plans) -> `backend.voice-labs/docs/README.md`" and "Real-provider smoke test -> `backend.voice-labs/docs/smoke-test.md`" |
| "The three repositories": "`backend.voice-labs` ... in one uv workspace" | Correct | no change |

### `concepts.md`

| Where | What is wrong | Replace with |
| --- | --- | --- |
| Line ~259 "**Not yet:** ... Per-listener audio routing (only the other person hears the translation) is future work; everyone in the room hears all audio." | Contradicts the code: private listening exists | "**Per-listener routing:** with `private_listening`, each person publishes in the session room and hears only their own translation in a separate listen room (`translation-for-<identity>`). Mock-tested; see `status.md` for the verification state." |
| Line ~225 "Interpreter mode (`mode=interpreter`): the agent repeats..." and the Creation bullet | The recommended entry point is now `POST /translations` | Add after the Creation bullet: "To translate between two people, prefer `POST /translations`; it creates a hidden interpreter agent for you. Such agents do not appear in `GET /agents` and cannot be read or edited (404 `agent_not_found`). An interpreter agent's `system_prompt` is ignored by the Runtime." |
| Line ~69 "Adding a provider (for example ElevenLabs TTS) is one ..." | ElevenLabs is already registered | "ElevenLabs TTS (`elevenlabs`) is registered next to Cartesia; selecting it is a provider record." |
| Line ~316 "Platform has no authentication or multi-tenancy (deferred)." | Wrong since migrations `0007`-`0010` | "Authentication (API keys, roles, session tokens) and per-organization scoping exist; see Contracts. A login UI does not." |
| Line ~253 "No barge-in: A translation is never cut off by the other person starting to speak" | The agreed policy (NS-02) is not implemented; the actual behaviour has not been measured | Prefix with "Intended:" and link `north-star.md` NS-02 |

### `quickstart-voice.md`

| Where | What is wrong | Replace with |
| --- | --- | --- |
| Lines ~30-32 "Runtime supports only these providers today: `deepgram` ... `cartesia` for TTS. Agents that select any other provider fail at session start." | `elevenlabs` is registered for TTS | "Runtime supports `deepgram` (STT), `openai` / `openai-compatible` (LLM) and `cartesia` or `elevenlabs` (TTS). ElevenLabs needs `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`; its service layer was checked against the real API on 2026-11-05, a full two-person call was not." |
| Lines ~222-225 "**Platform has no authentication yet.** Anyone who learns a public URL can ..." | Authentication exists but is OFF unless `PLATFORM_API_KEYS` is set | "Platform authentication is **off unless `PLATFORM_API_KEYS` is set**. If you expose Platform through a tunnel, set it first; anyone who learns the URL can otherwise create sessions that spend your credit." |
| Verification-status box at the top | Dated and still accurate in spirit (it says no complete spoken turn is recorded) but conflicts with `status.md`, which records real-provider verification of agent calls | Replace with a link to `status.md` for the verification state |

### `development.md`

| Where | What is wrong | Replace with |
| --- | --- | --- |
| Lines ~113-114 "`python -m pytest -q   # 99 passed, 1 skipped`" and "`# 44 passed, 6 skipped`" | Counts from an early run | Remove the numbers; link `backend.voice-labs/docs/status.md#test-results-2026-11-05` |
| Line ~169 "The current chain ends at `0004_voice_studio`." and the downgrade example `downgrade 0003_conversation_indexes` | Chain ends at `0014_system_agents` | "The chain ends at `0014_system_agents` (run `alembic heads`)." Keep the round-trip procedure; use a revision you actually want to test, for example `downgrade 0013_session_last_event` |
| Same section | The test-suite builds the schema with `create_all`; it does not run migrations | Add: "The Platform suite does not execute migrations; exercise `0014` with the upgrade/downgrade/upgrade round trip against a disposable database." |
| Variables table | Missing ElevenLabs and limits | Add rows for `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` (Runtime; only needed when a session selects the `elevenlabs` TTS provider), `PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG` (default 10), `PLATFORM_SESSION_CREATE_RATE` (default 30 per minute) and `PLATFORM_MAX_TRANSLATION_CONFIGS` (default 50, `0` disables) |
| Layout table | "(deterministic checks only so far)" for `evals` | evals now includes a live latency runner (`live_latency`); reword |

### `run-locally.md` (mostly current)

- Section 3 key list and the template block: add `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`
  (see the `.env.example` patch below), described as "only for sessions that select the
  ElevenLabs TTS provider".
- Section 10 ("Try live translation"): add a subsection "d. Translation between two people",
  with `curl -X POST localhost:8000/translations -H 'content-type: application/json' -d
  '{"languages":["ko","en"],"creator":{"language":"ko"}}'` and the invite call; mention the
  optional `tts_provider_id` (an ElevenLabs provider record). Do not claim it has been tried with
  real providers.
- Section 8's remark "If you assign a voice profile, its provider voice ID must be a real
  Cartesia voice" should read "... a real voice of the agent's TTS provider. Platform answers
  `409 voice_provider_mismatch` when the voice's and the agent's provider records differ, but it
  never checks that the id exists at the provider".

### `troubleshooting.md` (mostly current)

- Add to the "silent agent" row: "Voice ids are provider-specific. A mismatch is rejected with
  `409 voice_provider_mismatch` only when both the voice and the agent name a TTS provider;
  otherwise it is accepted and then silent or failing. With no TTS provider set the comparison is
  against Runtime's default (`cartesia`). A voice id with characters other than letters, digits,
  `_` and `-` (or longer than 128) is refused by Runtime: `422 invalid_voice_id` at session start
  for ElevenLabs, `422 invalid_participant_voice` for participants; Platform itself accepts the
  voice when it is saved."
- Add a row for `429 too_many_active_sessions` and `429 rate_limited` (object body with a
  `code`, `Retry-After` header on the latter); `troubleshooting.md` does not mention either
  (checked with grep).

### `BACKEND_ARCHITECTURE.md` (history)

Status "proposed". It describes the first slice: a workspace with git submodules under `src/`,
"Do not add ... organizations, roles", Qwen/faster-whisper/Chatterbox as first provider
candidates. Today there are no submodules (`AGENTS.md` forbids adding them), organizations and
roles exist, and the registered providers are Deepgram, OpenAI(-compatible), Cartesia and
ElevenLabs. Paste at the top:

```markdown
> **History (2026-11-05).** This is the original proposal for the first backend slice. It is not
> the current architecture: organizations and roles now exist, there are no submodules, and the
> provider registry and ElevenLabs are described in `backend.voice-labs/docs/status.md` and
> `contracts.md`.
```

### `BACKEND_TASKS.md` (history)

54 unchecked boxes, "Status: planned", tasks such as bootstrapping from a template and adding
submodules under `src/platform`. None of it reflects the delivered work. Paste at the top:

```markdown
> **History (2026-11-05).** This was the first-milestone backlog; the checkboxes were never
> updated and do not show what is built. For what exists and what is next see
> `backend.voice-labs/docs/status.md` and `north-star.md`.
```

## Lockfile revision note for `development.md` / `run-locally.md`

The Python locks (`runtime/uv.lock`, root `uv.lock`) now have `revision = 5` (produced with uv
0.13.7; the diff is only the extras line and the revision). Add to the Python-environments
section: "Use a uv that understands lock revision 5. An older uv may refuse the lock or
re-resolve it; the minimum version has not been verified." The plain-`venv` route does not use
the locks.

## `.env.example` patch

`stack.voice-labs/.env.example` ends with `CARTESIA_API_KEY=` and `CARTESIA_VOICE_ID=`. Runtime
reads `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` (`runtime/.env.example` already has them).
Suggested append (note the optional comment line):

```diff
 CARTESIA_API_KEY=
 CARTESIA_VOICE_ID=
+# Only needed when a session selects the elevenlabs TTS provider (provider record "tts" = elevenlabs).
+# Voice ids are provider-specific: a Cartesia voice id does not work with ElevenLabs.
+ELEVENLABS_API_KEY=
+ELEVENLABS_VOICE_ID=
```

Also consider adding commented examples of the optional limits
(`# PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG=10`, `# PLATFORM_SESSION_CREATE_RATE=30`).

## Not stale or not reviewed in depth

- `quickstart-api.md`: key-less API tour; I did not find statements contradicting the code. If
  authentication is turned on it will need an `Authorization` header (the page assumes it is off).
- `evaluation.md`: dated benchmark history plus a description of the `live_latency` runner;
  add one line, "numbers predate the 2026-11 changes".
- `PROGRESS.md`, `TEST_PLAN.md`, `DATA_SOURCES.md`, `PROJECT_REPORT.md`, `project-overview.md`,
  `architecture.md`, `PROJECT_STRUCTURE.md`, `BACKEND_MVP.md`, `BACKEND_ROADMAP.md`,
  `BACKEND_LIVE_PLAN.md`, `BACKEND_STUDIO_PLAN.md`: not read in detail (outside the requested
  list). Treat the plans as history, as stated in [README](README.md).

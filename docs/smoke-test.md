# Real-provider smoke test (human-run)

Written 2026-11-05. **Status:** the *automated ElevenLabs-layer script* (section 1a) has been run; everything else on
this page (Parts A and B, the checks, section 8) was derived from reading the code and has not been
run. Section 11 holds the recorded results.

> **The automated ElevenLabs-layer script was run on 2026-11-05 (15 of 15 passed, free tier); the manual two-person parts below have not been run.** See section 1a and section 11.
>
> **This test is billable and needs your own keys.** It calls Deepgram, OpenAI, and
> Cartesia and/or ElevenLabs, and uses a LiveKit server. Use keys you are willing to spend on, keep
> them in your shell environment or an untracked `.env`, and never paste them into this page, a
> ticket, or a chat. Recordings of people speaking stay out of the repository.

It exists to close the gap recorded in [Status](status.md): ElevenLabs live TTS and the
`/translations` API are implemented and mock-tested but never real-provider-verified. It also
re-checks per-listener audio routing after the 2026-11 changes (task TR-051 in
[Translate plan](translate-plan.md)).

## What this tests

| Question | Route to use |
| --- | --- |
| Does `/translations` work with real providers on the default TTS (Cartesia)? | Part A: `POST /translations` without `tts_provider_id` |
| Does ElevenLabs speak in a live two-person call? | Part B: `POST /translations` with `tts_provider_id` set to an ElevenLabs provider record |

The manual two-person Parts A and B have not been run: they are pending Deepgram and OpenAI keys,
LiveKit and two participants.

Run Part A only if you have Cartesia keys, Part B only if you have ElevenLabs keys. Voices are
provider-specific: keep each part's voices on that part's provider record. Platform answers
`409 voice_provider_mismatch` if a voice's provider record differs from the session's TTS provider.
With no `tts_provider_id` the session is compared against Runtime's default, `cartesia`, so an
ElevenLabs-registered voice on a default `/translations` is also `409 voice_provider_mismatch`
(only a voice with no provider record skips the check). That makes the mismatch negative test
(section 8, item 5) easy.

## 1. Prerequisites

- A machine that can run Docker (for the LiveKit server from `stack.voice-labs`), Python 3.11+,
  `curl`, `jq`.
- Two people (or two browser profiles and one person), each with a microphone and headphones.
  Headphones matter: with speakers, one person's translation can be picked up by the other's
  microphone and the test becomes unreadable.
- Optional but recommended: a Korean speaker and an English speaker. The test script below uses
  `ko` and `en`.
- A way to join a LiveKit room with the tokens Platform returns and to **publish a microphone**
  and **subscribe to a second "listen" room** (see step 5). `frontend.voice-labs` contains
  private-listening playback code (`components/features/call/use-call-room.ts`), but I found no
  call to `/translations` in its `app`, `components` or `lib` directories, so whether the web app can
  start this flow was not verified. If it cannot, use any LiveKit-capable client that accepts a
  URL and token.

### 1a. ElevenLabs layer: automated script

`runtime/scripts/elevenlabs_smoke.py` checks the ElevenLabs layer against the real API with no
Deepgram, OpenAI or LiveKit: key and voices, REST preview in en/ru/ko, whether `pcm_24000` is
headerless raw PCM, `eleven_multilingual_v2` without `language_code`, the live Pipecat WebSocket
service built by the runtime's own factory (first audio, 24 kHz, voice and language switch,
interruption of an in-flight context, a second utterance, clean shutdown) and the invalid-key
behaviour. **It is billable** (a few hundred characters) and refuses to run without `--confirm-billable`.

```bash
cd runtime && python scripts/elevenlabs_smoke.py --confirm-billable --out-dir /tmp/el-smoke --report /tmp/el-smoke.json
```

Key lookup, first match wins: the `ELEVENLABS_API_KEY` environment variable; the file named by
`--key-file` or `ELEVENLABS_API_KEY_FILE`; `~/Developer/11labs-api-key.txt`. The key is never printed
(output is redacted) and never written by the script. **To rotate the key:** replace the contents of
the key file or export a new `ELEVENLABS_API_KEY`; nothing is stored in the repository. The script
only measures objective things (durations, levels, timings); listen to the files in `--out-dir`
to judge quality. It needs the Runtime `voice` extra (Pipecat with ElevenLabs).

### Environment

All of these are read from the process environment. The local stack template is
`stack.voice-labs/.env.example` (copy to `.env`, which is gitignored). Values shown are
placeholders.

| Variable | Process | Purpose |
| --- | --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Platform and Runtime | LiveKit; the compose dev server uses `ws://localhost:7880`, `devkey`, `secret` |
| `PLATFORM_RUNTIME_TOKEN` | Platform | Token Platform sends to Runtime |
| `RUNTIME_PLATFORM_TOKEN` | Runtime | **Must equal** `PLATFORM_RUNTIME_TOKEN` |
| `RUNTIME_API_URL` | Platform | e.g. `http://localhost:8001`; if unset, Platform does not start a Runtime session |
| `RUNTIME_PLATFORM_URL` | Runtime | e.g. `http://localhost:8000`, where events are posted |
| `PLATFORM_DATABASE_URL` | Platform | PostgreSQL URL; without it Platform falls back to a local SQLite file |
| `PLATFORM_API_KEYS` | Platform | Leave empty for local use (authentication off). If you set it, send `Authorization: Bearer <key>` (the curl set below does this through `AUTH`) |
| `DEEPGRAM_API_KEY` | Runtime | Speech-to-text (default provider; not needed when the session selects `elevenlabs` STT) |
| `OPENAI_API_KEY` | Runtime | Translation (default provider; not needed when the session selects `gemini`) |
| `GEMINI_API_KEY` | Runtime | Translation through Gemini's OpenAI-compatible endpoint, when the session selects the `gemini` llm provider. Set the llm record's `model` or `GEMINI_TRANSLATION_MODEL` (no model is built in); `GEMINI_BASE_URL` optionally overrides the endpoint |
| `GOOGLE_TRANSLATE_API_KEY` | Runtime | Google Cloud Translation (key restricted to the Cloud Translation API), when the session selects the `google-translate` llm provider. Translation sessions only, not agent mode. Sent as the `x-goog-api-key` header. Measured ~0.15-0.5 s per sentence, whole sentence at once (no streaming), no conversation context or glossary. `GOOGLE_TRANSLATE_TIMEOUT_S` (default 2) is the per-sentence deadline after which the Gemini backup answers; `GOOGLE_TRANSLATE_KEEPALIVE_S` (default 30) is how long an idle connection stays open between sentences (5 = old behaviour); `GOOGLE_TRANSLATE_BASE_URL` redirects it for failure tests |
| `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID` | Runtime | Needed for Part A (and Part B's baseline); `CARTESIA_VOICE_ID` is the default voice for a participant without a voice profile |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | Runtime | Needed for Part B; required when a session selects the `elevenlabs` TTS provider, and `ELEVENLABS_API_KEY` also when it selects `elevenlabs` for STT (`stt_provider_id`). `ELEVENLABS_API_KEY_FILE` is read only by `scripts/elevenlabs_smoke.py`, not by the Runtime |

`stack.voice-labs/.env.example` does not yet list `ELEVENLABS_*`; add them to your untracked
`.env` (see [Stale stack docs](stale-stack-docs.md#envexample-patch)).

## 2. Start everything

Commands are taken from `stack.voice-labs/docs/run-locally.md` and checked against the module
paths in this repository (`platform/src/platform/main.py`, `runtime/src/runtime/api.py`); I did not
run them. Use your own virtualenv.

```bash
# Terminal 0: LiveKit, PostgreSQL, Valkey
cd stack.voice-labs && docker compose up -d && docker compose ps

# Install once (voice extra now includes elevenlabs)
python3.11 -m venv ~/.venvs/voice-labs
~/.venvs/voice-labs/bin/pip install -e "backend.voice-labs/platform[test]" -e "backend.voice-labs/runtime[voice,test]"

# Terminal 1: Platform
set -a; source stack.voice-labs/.env; set +a; export PYTHONDONTWRITEBYTECODE=1
cd backend.voice-labs/platform
~/.venvs/voice-labs/bin/python -m alembic upgrade head
~/.venvs/voice-labs/bin/python -m uvicorn src.platform.main:app --port 8000

# Terminal 2: Runtime
set -a; source stack.voice-labs/.env; set +a; export PYTHONDONTWRITEBYTECODE=1
cd backend.voice-labs/runtime
~/.venvs/voice-labs/bin/python -m uvicorn src.runtime.api:app --port 8001
```

Sanity checks (no provider calls):

```bash
curl -s localhost:8000/health
curl -s localhost:8001/health
# Provider reachability as seen by Runtime (token header required; this does call the providers'
# lightweight status endpoints for the keys that are set):
curl -s "localhost:8001/health/providers?refresh=true" -H "X-Runtime-Token: $RUNTIME_PLATFORM_TOKEN" | jq
```

Expected: both `/health` answer `{"status":"ok", ...}`; the providers report lists each provider
with a status. A `down` provider makes Platform refuse sessions with `503 provider_unavailable`.

## 3. curl set (placeholders only)

Set these once in the shell you use for the steps. Never put real keys in files you commit.

```bash
API=http://localhost:8000
AUTH=()                                   # authentication off locally
# AUTH=(-H "Authorization: Bearer $PLATFORM_API_KEY")   # if PLATFORM_API_KEYS is set
api() { curl -sS -H 'content-type: application/json' "${AUTH[@]}" "$@"; }

EL_VOICE_A="<elevenlabs-voice-id-for-person-A>"   # Part B only; must be usable through your API plan
EL_VOICE_B="<elevenlabs-voice-id-for-person-B>"
CA_VOICE_A="<cartesia-voice-id-for-person-A>"     # Part A only (optional; omit to use CARTESIA_VOICE_ID)
CA_VOICE_B="<cartesia-voice-id-for-person-B>"
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)              # used to filter analytics later
```

## 4. Part A: `/translations` with Cartesia (baseline)

A1. (Optional) create Cartesia voices, one per person. Without them each person is spoken for with
`CARTESIA_VOICE_ID`, so you cannot tell the voices apart; use two different ones to check voice
per participant.

```bash
CA_TTS=$(api -X POST $API/providers -d '{"kind":"tts","provider":"cartesia"}' | jq -r .id)
VA=$(api -X POST $API/voices -d "{\"name\":\"Smoke A\",\"tts_provider_id\":\"$CA_TTS\",\"provider_voice_id\":\"$CA_VOICE_A\"}" | jq -r .id)
VB=$(api -X POST $API/voices -d "{\"name\":\"Smoke B\",\"tts_provider_id\":\"$CA_TTS\",\"provider_voice_id\":\"$CA_VOICE_B\"}" | jq -r .id)
api -X POST $API/voices/$VA/preview -d '{"text":"Hello, this is a preview."}' -o /tmp/preview-a.wav   # plays the voice, billable
```

A2. Create the translation (person A speaks Korean, voice A).

```bash
T=$(api -X POST $API/translations -d "{\"languages\":[\"ko\",\"en\"],\"creator\":{\"language\":\"ko\",\"voice_profile_id\":\"$VA\",\"display_name\":\"A\"}}")
echo "$T" | jq 'del(.session_token)'          # do not paste the token anywhere
SID=$(echo "$T" | jq -r .session_id); CID=$(echo "$T" | jq -r .conversation_id)
TOKEN_A=$(echo "$T" | jq -r .session_token)
```

Expected: `201`; `connection.url`, `connection.token`, and (private listening) `connection.listen`
with `track` equal to `translation-for-browser-<session_id>`.

A3. Invite person B (English, voice B). Use A's session token or an admin key.

```bash
G=$(api -X POST $API/translations/$SID/invite -H "Authorization: Bearer $TOKEN_A" \
  -d "{\"language\":\"en\",\"voice_profile_id\":\"$VB\",\"display_name\":\"B\"}")
echo "$G" | jq 'del(.session_token)'
```

(When authentication is off, drop the `-H Authorization` argument.) A third invite must return
`409 session_full`; an invite with `"language":"ko"` must return `422 participant_languages_must_differ`.

A4. Inspect.

```bash
api $API/translations/$SID | jq          # participants never show provider voice ids
api $API/translations | jq '.[0]'
```

A5. Both people join (step 5 below), speak (step 6), then check (step 7) and end:

```bash
api -X POST $API/voice/sessions/$SID/end
```

## 5. Part B: `/translations` on ElevenLabs

B1. Record the provider, then two ElevenLabs voices.

```bash
EL_TTS=$(api -X POST $API/providers \
  -d '{"kind":"tts","provider":"elevenlabs","configuration":{"model":"eleven_flash_v2_5"}}' | jq -r .id)
EA=$(api -X POST $API/voices -d "{\"name\":\"EL A\",\"tts_provider_id\":\"$EL_TTS\",\"provider_voice_id\":\"$EL_VOICE_A\"}" | jq -r .id)
EB=$(api -X POST $API/voices -d "{\"name\":\"EL B\",\"tts_provider_id\":\"$EL_TTS\",\"provider_voice_id\":\"$EL_VOICE_B\"}" | jq -r .id)
api -X POST $API/voices/$EA/preview -d '{"text":"Hello, this is a preview."}' -o /tmp/el-preview-a.wav
file /tmp/el-preview-a.wav ; # should say RIFF WAVE audio, 16 bit, mono 24000 Hz; play it and listen
```

The automated script observed on 2026-11-05 that ElevenLabs' `pcm_24000` response is headerless
16-bit PCM, so this preview should play as speech; if it plays as loud noise, record it under Results. Voice ids must match `^[A-Za-z0-9_-]{1,128}$`; anything else is refused
before a request is made. Cloning and deleting ElevenLabs voices are not supported.

B2. Create the translation on the ElevenLabs record (person A speaks Korean, voice A).

```bash
T=$(api -X POST $API/translations -d "{\"languages\":[\"ko\",\"en\"],\"tts_provider_id\":\"$EL_TTS\",\"creator\":{\"language\":\"ko\",\"voice_profile_id\":\"$EA\",\"display_name\":\"A\"}}")
echo "$T" | jq 'del(.session_token)'
SID=$(echo "$T" | jq -r .session_id); CID=$(echo "$T" | jq -r .conversation_id); TOKEN_A=$(echo "$T" | jq -r .session_token)
G=$(api -X POST $API/translations/$SID/invite -H "Authorization: Bearer $TOKEN_A" \
  -d "{\"language\":\"en\",\"voice_profile_id\":\"$EB\",\"display_name\":\"B\"}")
echo "$G" | jq 'del(.session_token)'
```

Expected: `201` both times, and the session contract sent to Runtime carries
`providers.tts = "elevenlabs"` with `provider_settings.tts.model = "eleven_flash_v2_5"`. If Runtime
lacks `ELEVENLABS_API_KEY` or `ELEVENLABS_VOICE_ID`, session start fails with a message that names
only the ElevenLabs variables (not `CARTESIA_*`); that is correct behaviour. Try also a
nonexistent or non-tts `tts_provider_id`: expect `422 invalid_tts_provider_id`.

## 6. Join and speak

Each person joins with their own token: the creator uses `connection` from the create response,
the guest uses `connection` from the invite/participants response. With private listening each
connection has two parts:

- `connection.url` + `connection.token`: the **session room**. It lets the person publish their
  microphone but not subscribe, so nobody's original microphone can reach another person.
- `connection.listen`: `url`, `token`, `room_name`, `track`. A second, subscribe-only connection to
  a room only Runtime publishes to; the person hears **their translation** on the audio track named
  `translation-for-<their identity>`.

A client that only subscribes to the session room will hear nothing; that is expected and is the
reason the web app must play any subscribed audio track.

Speak a fixed script so runs are comparable, with a pause of about 2 seconds between utterances.
Suggested minimum: 5 utterances each way, short sentences (10 to 20 words), including one name
and one number. Example (adapt freely, keep a record):

- A (Korean): a greeting, a question about an appointment time, a phone number read digit by digit.
- B (English): an answer, a clarification, an address.

## 7. What to check

1. **Transcripts.** Original and translation, per speaker, with languages:

   ```bash
   api $API/conversations/$CID/messages | jq '.[] | {role, primary_language, content}'
   ```

   Check meaning, names, numbers. Expect user messages in the speaker's language and assistant
   messages in the listener's language.
2. **Per-listener isolation.** With headphones: A must hear only B's translation (Korean) and never
   B's original English voice or A's own voice; B the mirror. Also try `hear_original: true` on a
   separate `/translations` run (Part A only) and confirm both then hear the original quietly.
   Note: a participant can hear up to about 0.1 s of another's original voice on joining (known issue).
3. **Voice per participant.** The translation heard by B is spoken in A's voice profile (voice A),
   and the one heard by A in B's. If both people sound the same, the per-speaker voice is not
   applied (or both voices fell back to the default): record it.
4. **Latency.** Platform stores one `utterance_timeline` event per translated utterance. The raw
   events have no read API (`/conversations/{id}/runtime-state` omits `LatencyMeasured`). Read the
   aggregate p50/p95 after the run, filtered to the time of your run:

   ```bash
   api "$API/analytics/summary?since=$START" | jq .latency
   ```

   Expected key: `utterance_timeline` (value in milliseconds from speech start to first audible
   translated audio) and the other metric names listed in [Contracts](contracts.md#latencymeasured-metrics-for-translation).
   This has not been run; if `latency` is `{}`, record it as a finding. Also note by ear the delay
   from end of speech to first heard word. Do not compare to the 2026-10-21 benchmark
   tables as if they applied; they are a different build and provider mix.
5. **Runtime events.** `api $API/conversations/$CID/runtime-state | jq '[.events[].event_type]'`
   should show `RuntimeStarted`, `SpeechStarted`/`TranscriptFinal`-type events, and no
   `SessionFailed`, `RuntimeFailed` or `ProviderError`.
6. **Hidden-agent isolation (Part A).** `api -o /dev/null -w '%{http_code}\n' $API/agents/<agent_id of the conversation>` where
   the id is `jq -r .agent_id` of `GET /conversations/$CID`: expect `404`; `GET /agents` must not list
   it.

## 8. Interruption and provider-failure manual checks

These record what happens; they are not pass criteria for a feature that exists. The agreed
interruption policy (NS-02) is not implemented yet, so the observed behaviour is the baseline.

1. **Same speaker continues.** While A's first translation is still playing to B, A speaks again.
   Record: does B hear the first translation finish, get cut, or overlap? Are both translated?
2. **Listener interrupts.** While B hears A's translation, B speaks. Record: is A's translation to
   B cut, ducked or unaffected? Is B's speech translated normally?
3. **Wrong TTS key.** Start a Part B session in a Runtime whose `ELEVENLABS_API_KEY` is
   deliberately set to a bogus placeholder (never a real key in the log). Expected from the mocked
   test only: the session ends as an authentication failure; record the HTTP status of session
   creation, the `runtime-state` events (`SessionFailed`/`ProviderError`) and whether an apology
   clip plays (the committed clips are macOS placeholders).
4. **Provider unreachable mid-call.** Start a call, then block outbound traffic to the TTS
   provider with your firewall tool (do not stop the whole network). Record whether the call
   ends cleanly, an apology clip plays, the other person is told, and what events appear. Restore
   the network afterwards.
5. **Voice/provider mismatch (negative test).** In Part B, create a translation with
   `tts_provider_id` = the ElevenLabs record but `creator.voice_profile_id` = a Cartesia voice
   (`$VA`). Expected from the code: `409 voice_provider_mismatch`, and no session is created.
   Also try the ElevenLabs voice (`$EA`) on a `/translations` without `tts_provider_id`: expect
   `409 voice_provider_mismatch` (default fallback to `cartesia`). Then repeat with a voice that has
   no `tts_provider_id` (create one without the field): the check is skipped, Platform accepts it,
   and a Cartesia id sent to ElevenLabs should fail or be silent. Record what you get and hear.
6. **Hostile voice id.** `POST /voices` with `provider_voice_id` `../../v1/user` on the
   ElevenLabs record, then preview it: expect a failure (`502 voice_preview_failed`) and no request
   to ElevenLabs. Platform accepts any `provider_voice_id` up to 200 characters, so the voice saves; using it in a
   session as a creator or guest voice should give `422 invalid_participant_voice` from Runtime
   (`422 invalid_voice_id` applies to the session-level `voice_id` of an ElevenLabs session, which
   a hidden translation agent does not set; it is reachable with an agent-first session whose
   agent has an ElevenLabs voice profile).

## 9. Pass/fail table

Fill this in. Leave a row blank if you did not run it. "Evidence" is a file name, a log line,
or a short note, never a secret.

| # | Check | Part | Pass criterion | Result (pass / fail / not run) | Evidence / notes |
| --- | --- | --- | --- | --- | --- |
| 1 | LiveKit, Platform, Runtime start | both | `/health` ok on both | | |
| 2 | Provider health report | both | no required provider `down` | | |
| 3 | `POST /translations` | A | 201 with `connection.listen` | | |
| 4 | Invite, third person, same language | A | 201 / 409 `session_full` / 422 `participant_languages_must_differ` | | |
| 5 | ElevenLabs preview audible | B | plays as speech (not noise) | | |
| 5a | `tts_provider_id` rejected when invalid | B | `422 invalid_tts_provider_id` | | |
| 6 | ElevenLabs session starts | B | Runtime joins; no `SessionFailed` | | |
| 7 | Transcripts both directions | both | meaning, names and numbers correct | | |
| 8 | Per-listener isolation | both | no original voice, no self-echo | | |
| 9 | Voice per participant | both | two distinguishable voices as assigned | | |
| 10 | Latency recorded | both | p50/p95 or per-utterance times written down | | |
| 11 | Hidden agent invisible | A | `404` on `/agents/{id}`, not listed | | |
| 12 | Same-speaker interruption (observation) | both | behaviour written down | | |
| 13 | Listener interruption (observation) | both | behaviour written down | | |
| 14 | Bogus TTS key | B | clean failure, events recorded | | |
| 15 | Provider unreachable mid-call | B | clean end, events recorded | | |
| 16 | Voice/provider mismatch | B | `409 voice_provider_mismatch`; skipped-case behaviour written down | | |
| 17 | Hostile voice id | B | refused, no provider request | | |

## 10. Cost estimate

Estimate before you start and compare with the provider dashboards afterwards; the dashboards are
the source of truth.

- **TTS characters.** Every translated utterance is synthesised once per listener. With two people
  that is one synthesis per utterance. Count characters of the translations (from
  `GET /conversations/$CID/messages`, assistant rows) and any previews you made:
  `characters x price per 1,000 characters / 1,000`.
  Example of the arithmetic only: 2,000 characters at $0.04 per 1K = $0.08.
- **Rates.** Per the ElevenLabs pricing page (official docs fetched twice on 2026-11-05): $0.04
  per 1K characters for v4 Turbo, v3 Conversational and Flash/Turbo; $0.08 per 1K for v4, v3 and
  v2 Multilingual. **Re-check <https://elevenlabs.io/pricing/api> and your plan (free tiers may exclude
  library voices or API use) before relying on these numbers.** Cartesia, Deepgram and OpenAI: read
  your own plan; the rough constants in `evals/src/evals/live_latency.py` (marked there as list
  prices for estimation only) are not authoritative.
- **STT.** Deepgram streaming is billed per connected audio minute; each person has their own
  lane for the whole call, so budget two lanes times the call length.
- **LLM.** Count the translation requests (one per utterance or clause) times your model's price.
- Add a safety margin for retries, previews and re-runs.

## 11. Record results here

Date, git commit or branch, who ran it, provider plans (not keys), and the filled pass/fail table.

### 2026-11-05: automated ElevenLabs-layer script (run by the lead, free-tier key)

`runtime/scripts/elevenlabs_smoke.py --confirm-billable`. **15 of 15 checks passed.** 429
characters used in total across all runs, read from the account afterwards (free tier, 10,000-character limit). Objective measurements only; audio
quality was **not** judged by anyone (sample WAVs are in an ephemeral scratch directory for the user).
The key expires and is not in the repository.

| # | Check | Result | Measured |
| --- | --- | --- | --- |
| 1 | key works (subscription) | pass | tier `free` |
| 2 | voices available to the key | pass | 21 voices |
| 3 | REST preview en (`eleven_flash_v2_5`, `language_code`) | pass | 1.67 s audio, RMS 3189, 0.35 s request |
| 4 | REST preview ru | pass | 2.46 s, RMS 3267, 0.37 s |
| 5 | REST preview ko | pass | 2.32 s, RMS 3846, 0.40 s |
| 6 | `pcm_24000` body is headerless raw PCM | pass | first bytes `FE FF 00 00` (not `RIFF`), 80,248 bytes for 1.67 s |
| 7 | REST `eleven_multilingual_v2` ko without `language_code` | pass | 2.18 s, RMS 2295 |
| 8 | live websocket en, voice A | pass | first audio 0.29 s, 24 kHz, 1.68 s audio |
| 9 | live voice + language switch (voice B, ru) | pass | first audio 0.22 s including reconnect, 1.87 s audio |
| 10 | interruption of an open context | pass | 2 frames before, 0 frames after 0.5 s |
| 11 | new utterance after interruption | pass | first audio 0.17 s |
| 12 | clean shutdown (`EndFrame`) | pass | socket cleared |
| 13 | REST rejects an invalid key quickly | pass | `401` in about 0.2 s |
| 14 | websocket handshake with an invalid key | pass (info) | accepted; the key is checked after connecting (`1008 Invalid API key`) |
| 15 | live invalid key classifies as auth | pass | classified `auth` (this run recorded `fatal: False` at 12.07 s; after the fix in `speech_providers.py` the lead measured `fatal=True` at about 1.3 s) |

Finding of this run: the in-band `1008` close for a bad key was classified `other` and took about 3.3 s
to end the call; fixed (see [Status](status.md#elevenlabs-real-provider-run-2026-11-05)).

### Manual two-person parts (Parts A and B, section 8): pending

*(empty: not run. Needs Deepgram and OpenAI keys, LiveKit, and two participants. Fill the pass/fail
table in section 9 when run.)*

## 12. PostgreSQL check

Needs no provider keys and costs nothing. Reproduces the 2026-11-05 PostgreSQL results in
[Status](status.md#postgresql-check-2026-11-05) with an embedded PostgreSQL (no Docker). `$S` is any
scratch directory outside the repository. Written from the lead's description; I did not run it.

```bash
export S=<scratch dir>
python -m venv $S/pgenv && $S/pgenv/bin/pip install pgserver
# long-running python process that keeps the server alive and prints the URI (unix-socket host dir):
$S/pgenv/bin/python - <<'PY'
import os, time, pgserver
srv = pgserver.get_server(os.environ["S"] + "/pgdata", cleanup_mode=None)
srv.psql("create database vl;")
print(srv.get_uri())      # note the socket directory
time.sleep(10**6)
PY
```

In another terminal, with the lock-pinned environment (psycopg 3):

```bash
export PLATFORM_DATABASE_URL="postgresql+psycopg://postgres:@/vl?host=<socket dir>"
cd platform
alembic upgrade head                       # full chain 0001 -> 0014
alembic downgrade 0013_session_last_event && alembic upgrade head
alembic current                            # 0014_system_agents (head)
export PLATFORM_TEST_POSTGRES_URL="$PLATFORM_DATABASE_URL"   # on a migrated database
python -m pytest -q -p no:cacheprovider -W ignore tests/test_postgres_concurrency.py
python -m pytest -q -p no:cacheprovider -W ignore tests/test_translations.py tests/test_translations_hardening.py tests/test_translations_safety.py
```

Expected (2026-11-05): 1 passed; 52 passed. For the full suite use a fresh database with schema by
`create_all`: expect 717 passed, 17 failed, 1 skipped, the 17 being the pre-existing failures listed in
Status. A concurrency probe (not in the repository): release about 12 threads with a barrier to
`POST /translations` for the same new configuration, then count hidden agents and sessions (expect 1 and 12);
send one request with an unknown `voice_profile_id` and confirm no new agent or session appears.


## Latency benchmark (translation call)

`runtime/scripts/translation_benchmark.py` replays the same recorded sentences (200, alternating
ko/en, about 35% with digits, source speech from the free `edge-tts` voices) through a live
Platform + Runtime + LiveKit session and writes one line per utterance to `raw.jsonl`
(monotonic timestamps, server-side stage timings, recognised text, translation). Latency is
measured from the **last spoken sound** (the clips end with silence, so the file end would
overstate it by ~0.9 s) to the first translated audio. `report` prints P50/P90/P95/P99 per
stage, premature turn splits, number preservation, silent drops and fallback activations.
It is billable (about 22 ElevenLabs characters per utterance, ~4.4k per 200-utterance run).

```bash
cd runtime
python scripts/translation_benchmark.py prepare --out bench/samples
python scripts/translation_benchmark.py run --samples bench/samples --profile balanced --out bench/balanced --runtime-log <runtime.log> --confirm-billable
python scripts/translation_benchmark.py run --samples bench/samples --profile instant  --out bench/instant  --runtime-log <runtime.log> --confirm-billable
python scripts/translation_benchmark.py report bench/balanced bench/instant
```

Normal benchmarks run against the real Google endpoint (leave `GOOGLE_TRANSLATE_BASE_URL` unset).
Fault injection is a separate configuration: start `scripts/fake_google.py`, point the Runtime at
it with `GOOGLE_TRANSLATE_BASE_URL`, and pass `--fake-google http://localhost:9000 --google-mode slow1000`.

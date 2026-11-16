# Contracts

> **UNAI** (formerly Voice Labs) is a B2B AI communication platform: AI voice agents (primary), real-time translation, AI chatbots and developer APIs. First market: university admissions offices. First MVP: the AI Admissions Assistant for international students at Inha University. Calls run in the browser over WebRTC (LiveKit); telephone/SIP calling is deferred.

Runtime events carry `event_id`, `event_type`, `schema_version`, session/conversation/
agent IDs, timestamp, optional trace ID, and payload. Platform authenticates Runtime
requests and ingests events idempotently by event ID.

Platform-to-Runtime session handoff includes the selected agent/conversation/session
context and LiveKit room details. Runtime must join the requested room and report
lifecycle changes; the durable record remains owned by Platform.

## MVP endpoints (implemented 2026-11-12; tested with stubs, not yet with real voice providers)

All are generic: the university lives only in agent data. They run in the calls organisation (`PLATFORM_CALLS_ORG_ID`, default org) like `/calls`.

| Route | Auth | Body → reply |
| --- | --- | --- |
| `GET /demo/agents/{key}` | public, rate limited; 404 `demo_disabled` unless `PLATFORM_DEMO_AGENTS` is set | → `{key, name, description, enabled_languages}` |
| `POST /demo/agents/{key}/sessions` | public, rate limited | `{name?}` → same as `POST /voice/sessions` (incl. `session_token`). Reaped after `PLATFORM_DEMO_MAX_SECONDS` (300). |
| `POST /demo/agents/{key}/chat` | public, per-IP limit (`PLATFORM_CHAT_PER_IP_MINUTE`) | `{message, conversation_id?}` → `{conversation_id, reply, handoff_suggested}`. 30 user messages per conversation (429 `chat_limit`), 60-minute window; 503 `chat_unavailable` without a key, 502 `chat_failed` upstream. LLM: `PLATFORM_CHAT_LLM_MODEL` (default `gemini-3.5-flash-lite`), key `PLATFORM_CHAT_LLM_API_KEY` or `GEMINI_API_KEY`. |
| `POST /handoffs` | `Authorization: Bearer vls_…` of the AI session | `{session_id, language="en", staff_language=PLATFORM_HANDOFF_STAFF_LANGUAGE or "ko", name?, reason?}` → ends the AI session, opens a translated call + invite, returns the `POST /calls` reply plus `handoff_id`. One handoff per AI session (409 `handoff_exists`). Requires `PLATFORM_PUBLIC_CALLS=1` for the staff join. |
| `GET /staff/handoffs` | `X-Staff-Key` = `PLATFORM_STAFF_KEY` (404 when unset, 401 when wrong) | → waiting + accepted handoffs, newest first; never the invite code |
| `POST /staff/handoffs/{id}/accept` | staff key | → `{invite_code, call_session_id}`; the officer then uses `POST /calls/invites/{code}/join` (the portal opens `/join#c=<code>`). 409 `handoff_not_waiting`. |
| `GET /staff/handoffs/{id}/transcript` | staff key | → the AI conversation's messages |

Knowledge: when an agent's linked documents total ≤ `PLATFORM_INLINE_KNOWLEDGE_MAX_CHARS` (default 60000; 0 disables), the Platform appends them to the session's `system_prompt` and sends `knowledge_enabled: false` (no `search_knowledge` tool). Larger sets keep the search tool.

## Authentication

Updated 2026-11-05 to match `platform/src/platform/auth.py` and `context.py`.

Credentials are tried in this order for every non-public, non-runtime route:

1. **Superadmin key**: a value from `PLATFORM_API_KEYS` (comma-separated). It acts in the
   organization named by the `X-Organization-Id` header, or the default organization.
2. **Organization API key** (`vlk_...`): created with `POST /api-keys`, bound to one
   organization and one role (`owner` > `admin` > `member`). Stored as a SHA-256 digest;
   the plaintext is shown once.
3. **Session token** (`vls_...`): see below. Valid only on session-scoped routes.
4. Anything else: `401 invalid_api_key`.

**When `PLATFORM_API_KEYS` is unset or empty, authentication is disabled** (principal `dev`,
default organization, owner) and Platform logs one startup warning. This is the local
development default. A `vlk_` key presented anyway is still validated. Never expose the API
publicly without setting it.

When enabled, every endpoint requires `Authorization: Bearer <key>`, and a route is open only
if explicitly allowlisted. Every table row is scoped to one organization; another
organization's id is indistinguishable from a missing one (`404`, never `403`).

| Route | Credential |
| --- | --- |
| `GET /health` | None (public) |
| `POST /runtime/events`, `POST /runtime/agents/{agent_id}/knowledge/search` | Runtime token (`X-Runtime-Token`); no tenant credential is accepted here |
| `GET /voice/sessions/{id}`, `POST /voice/sessions/{id}/end`, `.../ensure-running`, `.../participants`, `.../phone-participants`, `GET /translations/{id}`, `POST /translations/{id}/invite`, `GET /conversations/{id}`, `.../messages`, `.../runtime-state`, `POST /conversations/{id}/reports` | Member+ key, or a session token for that session (creator-only where noted below) |
| Creating things (`POST /voice/sessions`, `POST /translations`) and reading collections | Member+ |
| Creating or editing agents, providers, voices, deployments, API keys | Admin+ |
| Organization management (`/organizations`) | Superadmin |

Errors: `401 authentication_required` (no `Authorization` header), `401 invalid_api_key`
(wrong key or not a `Bearer` credential; the response carries `WWW-Authenticate: Bearer`),
`403 insufficient_role`, `403 forbidden` (a session token used on another session, on a
creator-only route as a guest, or on a route that is not session-scoped). CORS allows the
`authorization` and `x-organization-id` headers.

Admission limits: at most `PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG` (default 10) active sessions
per organization (`429 too_many_active_sessions`) and `PLATFORM_SESSION_CREATE_RATE` (default
30/min, per key, in process memory) session creations (`429 rate_limited`). Both 429 bodies are objects, not strings:
`{"detail": {"code": "too_many_active_sessions" | "rate_limited", "message": "..."}}`;
`rate_limited` also carries a `Retry-After` header. A value of `0` disables either limit. The
rate limiter is a token bucket (burst = the per-minute rate) keyed by API key id, or by
principal and organization when there is no key id; it is per Platform process.

### Session tokens

`POST /voice/sessions` returns an extra `session_token` (prefix `vls_`) for the creator. Adding
a participant with `POST /voice/sessions/{id}/participants` returns one for that guest as
`session_token`. Tokens are random, shown once, and only their SHA-256 digests are stored
(`voice_sessions.session_token_hash` and `guest_token_hashes`, migration `0007_session_tokens`).
A token works only on the session-scoped routes above and only for its own session (and its
conversation). Only the creator's token may add participants; a guest's token may not.
A browser can therefore be handed a session token instead of an admin key. Tokens are returned
even while authentication is disabled but are not required.

## Milestone 8: Voice Studio, Deployments, and Calls

*Reconciled 2026-11-05:* the sentence below that lists what Platform does not do ("no ... TTS
preview, PSTN dialing, SIP transfer ...") describes Milestone 8 only. Preview, provisioning,
dialing and transfer exist now and are described further down.

Backend-only contracts for the **Voice → Agent → Deployment → Call** model. Platform stores durable metadata only: no
audio, no raw voice samples, no provider provisioning, TTS preview, PSTN dialing, SIP
transfer, embeddings, or LLM summaries. Error bodies are `{"detail": "<code>"}`; request
shape/format failures (for example a non-E.164 phone number or a secret-looking
`configuration` key) return FastAPI's standard `422` validation `detail` list instead of a
code. All `configuration`/`metadata` objects reject secret-looking keys (`secret`, `password`,
`token`, `api_key`, `apikey`, matched case-insensitively at any depth); use references.
Phone numbers must match E.164 (`^\+[1-9]\d{6,14}$`). Unknown IDs on `GET`/`PATCH`/`DELETE`
return `404` with `<object>_not_found`.

### Voices

`POST /voices` (library), `POST /voices/clone`, `POST /voices/design`, `GET /voices`
(filters `status`, `source`), `GET /voices/{id}`, `PATCH /voices/{id}` (`name`, `languages`,
`style`; `null` is rejected), `POST /voices/{id}/status`, `POST /voices/{id}/provision`,
`POST /voices/{id}/preview`, `DELETE /voices/{id}` (`204`, sets `status=disabled`). Voice
fields: `id`, `name`, `source` (`library|clone|design`), `status`, `tts_provider_id`,
`provider_voice_id`, `languages` (default `["en","ru"]`), `style`, `design_prompt`,
`sample_reference`, `consent`, `failure_reason`, timestamps.

| Create | Required | Initial status |
| --- | --- | --- |
| `POST /voices` | `name`, `tts_provider_id`, `provider_voice_id` | `ready` |
| `POST /voices/clone` | `name`, `sample_reference`, `consent` | `pending` |
| `POST /voices/design` | `name`, `design_prompt` (≤2000 chars) | `pending` |

`style` accepts only `energy`, `warmth`, `speed` (floats in `[0, 1]`) and `style` (label,
≤50 chars); other keys are rejected. `tts_provider_id`, when given, must be an enabled `tts`
provider (`422 invalid_tts_provider_id`).

Clone consent is mandatory and stored with the voice, with a server `recorded_at` added:

```json
{
  "name": "Support voice",
  "sample_reference": "https://example.com/anna.wav",
  "consent": {
    "confirmed": true,
    "subject_name": "Anna Petrova",
    "statement": "I consent to cloning my voice for this agent.",
    "method": "self"
  }
}
```

`method` is `self` or `authorized`. Clone errors: `422 voice_consent_required` (consent
missing or `confirmed` is not `true`), `422 invalid_sample_reference` (not a
`scheme://...` reference without whitespace, or longer than 1024 chars).

`POST /voices/{id}/status` takes `{"status", "provider_voice_id?", "failure_reason?"}`.
Transitions are validated (`409 invalid_voice_status_transition`); moving to `ready`
requires a provider voice ID in the request or already on the voice
(`422 provider_voice_id_required`). `failure_reason` is kept only for `failed` and cleared
otherwise.

| From | Allowed next |
| --- | --- |
| `pending` | `processing`, `failed` |
| `processing` | `ready`, `failed` |
| `failed` | `pending` |
| `ready` | `disabled` |
| `disabled` | `ready` (only if a `provider_voice_id` was already set; otherwise `409`) |

#### Voice cloning provisioning

`POST /voices/{id}/provision` triggers the provisioning of a cloned voice: Platform sends
a request to Runtime, which downloads the consented `sample_reference` (HTTPS only,
≤10 MB, audio content-type, host must be public IPv4/IPv6), validates it, and uploads it
to the TTS provider's clone API. Platform records the `provider_voice_id` on success or a
`failure_reason` on error. The voice must have `source=clone` and `status` in
`{"pending", "failed"}` (`409 invalid_voice_status_transition` otherwise); it must carry
valid `consent` with `confirmed: true` (`409 voice_consent_required` otherwise);
`sample_reference` must be HTTPS (`422 unsupported_sample_url` otherwise).

Request timeout: 60 seconds. The voice is moved to `processing` before the request and to
`ready` on success or `failed` with a `failure_reason` on error.

Failure reasons (safe to expose):
- `unsupported_sample_url`: sample URL is not HTTPS
- `sample_unavailable`: sample host did not resolve, returned HTTP error, is private/loopback, or is too large (>10 MB)
- `clone_provider_error`: the TTS provider (Cartesia) call failed. Cloning is Cartesia-only; a voice whose provider record is `elevenlabs` fails with `unsupported_tts_provider` (see [ElevenLabs](#elevenlabs-live-tts))
- `tts_not_configured`: provider credentials not set in Runtime
- `unsupported_tts_provider`: provider name is not supported by Runtime
- `invalid_runtime_token`: Platform runtime token is invalid
- `runtime_error_NNN`: Runtime returned HTTP NNN without a recognized error code
- `runtime_unreachable`: network error reaching Runtime
- `runtime_invalid_response`: Runtime response was malformed or had no provider voice ID

**Status:** Implemented and tested with mocks. Not exercised against a real Cartesia
trunk; sample download and clone upload shape follow
`https://docs.cartesia.ai/api-reference/voices/clone`.

#### Voice preview

`POST /voices/{id}/preview` with `{"text"}` (≤300 chars) returns audio (MIME type
`audio/wav`) synthesized by the voice's TTS provider. The voice must have `status=ready`
and a `provider_voice_id` (`409 voice_not_ready` otherwise). Platform forwards the request
to Runtime's `POST /tts/preview` endpoint with the voice provider, ID, and text.

Errors: `409 voice_not_ready`, `503 runtime_not_configured`, `503 runtime_auth_not_configured`,
`502 voice_preview_failed` (Runtime error or timeout).

**Status:** Cartesia preview verified against the real API. Runtime's `/tts/preview` also
accepts `provider: "elevenlabs"` (key `ELEVENLABS_API_KEY`, default model `eleven_flash_v2_5`,
PCM from ElevenLabs wrapped in a WAV by Runtime); the Runtime synthesis function was run against
the real ElevenLabs API on 2026-11-05 (en/ru/ko, audible clips; see
[ElevenLabs (live TTS)](#elevenlabs-live-tts)); the Platform route is mock-tested. The provider name comes from the voice's `tts_provider_id` record
(`cartesia` when the voice has none); only `model` and `language` of that record's
`configuration` are forwarded. Runtime sends `language_code` to ElevenLabs only for the models
in a local table (`eleven_flash_v2_5`, `eleven_turbo_v2_5`, `eleven_v3`,
`eleven_v3_conversational`) and only when the language is listed for that model; it never sends
one for `eleven_multilingual_v2` or an unknown model. Observed 2026-11-05 (not documented by ElevenLabs): `output_format=pcm_24000` returns headerless
16-bit mono PCM, so the WAV wrapping is correct.
The voice id must match `^[A-Za-z0-9_-]{1,128}$` or Runtime refuses it before any request.
Live calls use the provider selected by the agent's TTS provider record; see
[ElevenLabs (live TTS)](#elevenlabs-live-tts).

**Agent voice.** Agents gain `voice_profile_id` (nullable) on create/update/read. It must
reference an existing, non-disabled voice (`422 invalid_voice_profile_id`); it need not be
`ready` yet. The legacy `voice_id` string stays supported.

### Deployments

`POST /deployments`, `GET /deployments` (filters `agent_id`, `channel`, `status`),
`GET /deployments/{id}`, `PATCH /deployments/{id}` (`name`, `status`, `phone_number`,
`configuration`), `DELETE /deployments/{id}` (`204`, archives). Fields: `id`, `agent_id`,
`name`, `channel`, `status`, `phone_number`, `configuration`, timestamps.

| Rule | Detail |
| --- | --- |
| `channel` | `web`, `sip`, `api`; fixed at creation |
| `status` | `active` (default), `paused`, `archived`; archived is terminal |
| Phone | `sip` requires E.164 `phone_number` (`422 phone_number_required`); other channels must not have one (`422 phone_number_only_for_sip`) |
| Agent | must exist and not be `disabled` (`422 invalid_agent_id`); not changeable after creation |
| Archived | `PATCH` returns `409 deployment_archived` |

Phone numbers are not checked for uniqueness across deployments.

**Sessions.** `POST /voice/sessions` accepts `{"agent_id", "deployment_id?"}`. With a
`deployment_id` it returns `404 deployment_not_found`, `409 deployment_not_active`, or
`409 deployment_agent_mismatch` (these follow the existing `409 agent_not_enabled`). The
session response shape is unchanged; the deployment is recorded on the conversation:
`deployment_id` and `channel` (the deployment's channel, or `web` when none is given).

**Voice resolution.** The runtime handoff `voice_id` is `agent.voice_id` unless the agent has
a `voice_profile_id`; then the voice must be `status=ready` with a `provider_voice_id`
(otherwise `409 selected_voice_unavailable`) and its `provider_voice_id` is sent as
`voice_id`.

### SIP and outbound calls

`POST /deployments/{id}/bridge-call` bridges two phone numbers into a single LiveKit
room with an interpreter agent. The deployment must have `channel=sip` and an agent with
`mode=interpreter` using the `cascaded` engine and `direction: "auto"`; otherwise the call is
refused with `422 bridge_requires_two_way_interpreter` before anything is dialed (the two
legs aren't registered as separate participants yet, so the language has to be detected per
utterance). Both legs join the same room and the agent hears and speaks
translations. Request: `{"from_number", "to_number"}` (both E.164). The `from` party is
dialed first, then `to`; if either fails, both calls are hung up and the session is ended.
Response: `session_id`, `conversation_id`, `room_name`, and `participants` list with
`role` (`from` or `to`), phone number, SIP identity, and LiveKit participant ID.

Errors:
- `422 deployment_not_sip`: deployment channel is not `sip`
- `422 agent_not_interpreter`: agent mode is not `interpreter`
- `409 deployment_not_active`: deployment status is not `active`
- `422 bridge_numbers_must_differ`: `from_number` equals `to_number`
- `409 sip_not_configured`: LiveKit SIP trunk not configured
- `502 session_create_failed`: could not create voice session
- `502 sip_dial_failed`: could not dial one or both numbers

#### Outbound campaign dialing

`POST /campaigns/{id}/dial-next` (for demos and tests) dials exactly one pending target
and waits for the attempt to resolve (answered or failed), then returns once the result
is known. The campaign must be `running` and SIP must be configured. Returns
`{"result": "completed" | "failed" | "no_answer", "target": {...}}`.

Errors:
- `409 sip_not_configured`: LiveKit SIP trunk not configured
- `409 campaign_not_running`: campaign status is not `running`
- `409 no_pending_targets`: no pending targets remain

When a campaign's status is set to `running` and `LIVEKIT_SIP_TRUNK_ID` is configured,
an in-process dispatcher starts dialing pending targets. For each attempt it opens a voice
session, dials the target into that room, and records whether the call was answered,
failed, or got no answer. Targets transition: `pending → calling → completed | failed |
no_answer`. Failed or unanswered attempts retry if `attempts < max_attempts`, else move
to their final status. The dispatcher runs one process per Platform instance (not safe
for multi-replica deployments) and does not honour `schedule_start`/`schedule_end` or
`retry_delay_*` configuration. A crash leaves the in-flight target in `calling`.

**Configuration and environment:**
- `LIVEKIT_SIP_TRUNK_ID`: LiveKit outbound SIP trunk ID (required to enable dialing)
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`: LiveKit credentials (required)
- `LIVEKIT_SIP_RINGING_TIMEOUT_S`: seconds to let callee ring (default 30)
- `LIVEKIT_SIP_KRISP`: `1` to enable Krisp noise cancellation on the SIP leg
- `PLATFORM_API_URL`: where Platform reaches itself to open voice sessions (default
  `http://localhost:8000`)

#### SIP transfer

When the LLM calls `transfer_to_human` in a SIP call (a call from a `sip` deployment),
Runtime transfers the SIP participant to the configured destination. The destination is
resolved from the session's `transfer_to` (set by Platform from
`deployment.configuration.transfer_to`), or from `SIP_TRANSFER_TO` environment variable.
Bare E.164 numbers are converted to `tel:+...` URIs; `tel:`, `sip:`, and `sips:` URIs
pass through as-is.

Web sessions (no SIP participant) record the handoff intent without transferring.

**Configuration and environment:**
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`: required; the transfer handler is
  installed whenever these are set (no trunk ID needed for transfers)
- `SIP_TRANSFER_TO`: default destination for transfers (E.164 or tel:/sip:/sips: URI);
  overridden by `deployment.configuration.transfer_to` per session

`deployment.configuration.transfer_to` is validated on create and update: `+E.164` or a
`tel:`/`sip:`/`sips:` URI without spaces, at most 255 characters (`422`, message
`invalid_transfer_to`).

Tool errors: `transfer_not_configured` (a SIP caller but no destination; the agent is told no
human is being connected), `transfer_failed` (LiveKit returned a transfer failure reason),
`transfer_not_configured` (no destination resolved), `livekit_api_not_installed`,
plus generic `TransferError` on network or API failure.

**Status:** Implemented and tested with mocks. Not verified against a real SIP trunk.

### Campaigns and targets

When `LIVEKIT_SIP_TRUNK_ID` is configured, moving a campaign to `running` starts an
in-process dispatcher that dials pending targets. Otherwise campaigns are durable records
only and setting to `running` places no calls.

`POST /campaigns`, `GET /campaigns` (filters `agent_id`, `status`), `GET /campaigns/{id}`,
`PATCH /campaigns/{id}`, `POST /campaigns/{id}/status`, `POST /campaigns/{id}/dial-next`,
`POST /campaigns/{id}/targets`, `GET /campaigns/{id}/targets`. Create body: `name`,
`agent_id`, `deployment_id`, `schedule_start?`, `schedule_end?` (naive datetimes are
treated as UTC), `timezone` (IANA, default `UTC`), `max_attempts` (1-10, default 1),
`configuration`. Reads add `status` (starts `draft`) and `target_counts`
(`{"<target status>": n}`). The `configuration` may include `max_concurrent_calls` (1-10,
default 1) to limit concurrent outbound dials.

| Error | Cause |
| --- | --- |
| `422 invalid_agent_id` / `invalid_deployment_id` | unknown agent / deployment |
| `409 deployment_agent_mismatch` | deployment belongs to another agent |
| `422 campaign_requires_sip_deployment` | deployment channel is not `sip` |
| `422 invalid_timezone` / `invalid_schedule` | bad IANA zone / `schedule_end <= schedule_start` |
| `409 campaign_not_editable` | `PATCH` outside `draft`/`scheduled`; add targets outside `draft`/`scheduled`/`paused` |
| `409 invalid_campaign_status_transition` | transition not in the table below |
| `409 deployment_not_active` | moving to `scheduled` or `running` with a non-active deployment |
| `409 campaign_has_no_targets` | moving to `scheduled` or `running` with zero targets |

| From | Allowed next |
| --- | --- |
| `draft` | `scheduled`, `cancelled` |
| `scheduled` | `draft`, `running`, `cancelled` |
| `running` | `paused`, `completed`, `cancelled` |
| `paused` | `running`, `cancelled` |
| `completed`, `cancelled` | none |

`POST /campaigns/{id}/targets` takes `{"targets": [{"phone_number", "name?",
"variables?"}]}` (1-1000 per request) and dedupes by phone number within the campaign and
within the request:

```json
{ "added": 2, "skipped_duplicates": 1 }
```

`GET /campaigns/{id}/targets` accepts `status`, `limit` (1-500, default 100), `offset`.
Target fields: `id`, `campaign_id`, `phone_number`, `name`, `variables`, `status`
(`pending | calling | completed | failed | no_answer`), `attempts`, `conversation_id`,
timestamps. A concurrent insert race that persists returns `409 target_conflict`.

### Tools and knowledge

**Tools.** `POST /tools`, `GET /tools` (filter `enabled`), `GET`/`PATCH /tools/{id}`,
`DELETE /tools/{id}` (`204`, sets `enabled=false`). Fields: `id`, `name`
(`^[a-z][a-z0-9_]{0,63}$`, unique, else `409 tool_name_conflict`), `description`,
`tool_type` (`http` default, or `builtin`), `input_schema` (JSON Schema object: top-level
`"type": "object"`, `properties` an object if present), `configuration`, `enabled`.

`PUT /agents/{id}/tools` with `{"tool_ids": [...]}` **replaces** the agent's tool set
(duplicates collapsed; `[]` clears) and returns the tools sorted by name. Unknown or disabled
tools return `422 invalid_tool_id`. `GET /agents/{id}/tools` lists the assigned tools
(including any disabled after assignment). Unknown agent: `404 agent_not_found`.

**Knowledge documents.** `POST /knowledge/documents`, `GET /knowledge/documents` (filter
`enabled`; summaries omit `content`), `GET`/`PATCH /knowledge/documents/{id}`,
`DELETE` (`204`, sets `enabled=false`). Fields: `id`, `title` (≤300), `source` (≤1024),
`content` (1 to 1,000,000 chars), `metadata`, `enabled`, `version`, `content_length`,
timestamps. `version` starts at `1` and increments only when a `PATCH` changes `content` to
a different value; metadata, title, and `enabled` edits do not bump it.

`PUT /agents/{id}/knowledge` with `{"document_ids": [...]}` replaces the agent's document
set (same semantics as tools; `422 invalid_knowledge_document_id` for unknown or disabled
documents) and returns summaries sorted by title; `GET` lists them.

*Reconciled 2026-11-05 against `sessions.py`:* the Platform to Runtime session handoff does
carry them: `tools` (the agent's enabled tools, as LLM function definitions) and a
`knowledge_enabled` flag, and Runtime reaches knowledge through
`POST /runtime/agents/{agent_id}/knowledge/search` (runtime token). The earlier text of this
paragraph, "stored and assignable only ... not included in the handoff", described the first
milestone. Retrieval is keyword search over whole documents; there is no chunking or embeddings.
The hidden `/translations` agents have no tools or knowledge, and the Runtime never uses them in
interpreter mode.

### Agent templates

Static personas: `customer_support`, `sales`, `receptionist`, `booking`, `restaurant`,
`travel`. `GET /agent-templates` and `GET /agent-templates/{key}` return `key`, `name`,
`description`, `system_prompt`, `enabled_languages`, and `suggested_tools` (names only; no
tools are created or assigned). Unknown key: `404 agent_template_not_found`.

`POST /agent-templates/{key}/agents` (`201`) creates an `enabled` agent from the template.
The optional body overrides `name`, `description`, `system_prompt`, `enabled_languages`; the
response is `{id, name, description, system_prompt, enabled_languages, status,
template_key}`. No voice, provider, or tool is set.

### Conversation outcomes, actions, and filters

Conversation reads (`GET /conversations`, `GET /conversations/{id}`) gain:

| Field | Meaning |
| --- | --- |
| `deployment_id`, `channel` | origin (`channel` defaults to `web`) |
| `outcome` | one of the values below, or `null` until recorded |
| `summary`, `intent` | free text; `intent` ≤200 chars |
| `actions` | list of objects (`[]` by default) |
| `outcome_recorded_at` | time of the last outcome write |

Outcomes: `resolved`, `booked`, `callback_requested`, `no_answer`, `declined`, `escalated`,
`failed`, `other`.

`GET /conversations` filters: `agent_id`, `status`, `outcome`, `channel` (combinable),
newest first.

`PUT /conversations/{id}/outcome` with `{"outcome", "summary?", "intent?", "actions?"}`
records an outcome and returns the conversation. Omitted optional fields keep their existing
values (they are not cleared); `actions` replaces the list when sent. Errors:
`422 invalid_outcome`, `404 conversation_not_found`.

### New Runtime events

`ToolCalled`, `HandoffRequested`, and `OutcomeRecorded` are accepted by
`POST /runtime/events` and declared in Runtime's `RuntimeEventType`. They use the standard
envelope (schema version `1`, idempotent by `event_id`) and appear in
`GET /conversations/{id}/runtime-state`. Payloads are validated before the duplicate check;
a duplicate event causes no second effect.

| Event | Payload | Effect on conversation | Errors |
| --- | --- | --- | --- |
| `ToolCalled` | `tool_name` (non-empty string, required), `arguments?` (object), `status?`, `error?` (strings) | none beyond storing the event | `422 invalid_tool_event` |
| `HandoffRequested` | `reason?`, `target?` (strings) | sets `outcome=escalated`, `outcome_recorded_at`, appends `{"type":"handoff_requested","reason","target"}` to `actions` | `422 invalid_handoff_event` |
| `OutcomeRecorded` | `outcome` (required, valid value), `summary?`, `intent?` (≤200), `actions?` (list of objects) | same as `PUT .../outcome` | `422 invalid_outcome` (bad outcome), `422 invalid_outcome_event` (other fields) |

Later writes win: an `OutcomeRecorded` after a handoff replaces `escalated`, and a handoff
overwrites any earlier outcome. *Reconciled 2026-11-05:* the event itself only records intent;
the actual SIP transfer is performed by Runtime's `transfer_to_human` tool, see
[SIP transfer](#sip-transfer) (mock-tested only).

### Analytics

`GET /analytics/summary?agent_id=<uuid>&since=<ISO datetime>` (both optional; `since`
filters on conversation `started_at`, naive values are UTC). Missing values are counted
under `"unknown"`.

```json
{
  "conversations": {
    "total": 12,
    "by_status": { "ended": 10, "failed": 2 },
    "by_outcome": { "resolved": 6, "booked": 2, "unknown": 4 },
    "by_channel": { "web": 9, "sip": 3 }
  },
  "languages": { "en": 40, "ru": 22, "unknown": 3 },
  "latency": { "stt": { "count": 25, "p50": 120.0, "p95": 310.0 } }
}
```

`languages` counts messages by `primary_language`. `latency` is keyed by metric name with
nearest-rank `p50`/`p95` over `LatencyMeasured` events.

**Provisional:** *Reconciled 2026-11-05:* Runtime does emit `LatencyMeasured` now (per-utterance
`utterance_timeline`, realtime and translation metrics; see below), so the "`{}` in practice"
statement that followed is out of date, but the payload shape is still not a stable contract. Platform currently accepts a
metric name in the first string among `metric`, `stage`, `name` and a number (not
boolean) in the first of `value_ms`, `latency_ms`, `duration_ms`, `value`; other events are
skipped. Runtime must fix the real payload before this is relied upon.

### Interpreter mode

An agent with `mode=interpreter` acts as a translator between two people (two browsers, or a
browser and a phone). Instead of a personality, it carries an `interpreter` configuration with the language
pair and translation engine:

```json
{
  "mode": "interpreter",
  "interpreter": {
    "languages": ["en", "ru"],
    "direction": "auto",
    "engine": "cascaded"
  }
}
```

| Field | Values | Detail |
| --- | --- | --- |
| `languages` | list of 2 distinct IANA codes | Required; supported: `en`, `ru`, `ko`, `ja`, `zh`, `es`, `fr`, `de`, `it`, `pt`, `nl`, `hi`, `tr`, `pl` |
| `direction` | `auto` (default) or `one_way` | `auto` requires both languages in Deepgram's code-switching set (`en`, `es`, `fr`, `de`, `hi`, `ru`, `pt`, `ja`, `it`, `nl`) |
| `engine` | `cascaded` (default) or `realtime` | `cascaded`: separate STT → LLM → TTS pipeline; `realtime`: OpenAI gpt-realtime-translate (one_way only, provider voices, target language must be in its output set: `es`, `pt`, `fr`, `ja`, `ru`, `zh`, `de`, `ko`, `hi`, `id`, `vi`, `it`, `en`) |

Creating or updating an agent requires `interpreter` if and only if `mode=interpreter`;
omitting it when mode is `interpreter` raises `422 interpreter_config_required`, and
including it when mode is `agent` raises `422 interpreter_config_not_allowed`.

Invalid combinations raise `422 invalid_interpreter_config`:
- `engine=realtime` with `direction != one_way`
- `engine=realtime` with target language (second in `languages`) not in the realtime
  output set

**Status:** `direction=auto` and `engine=cascaded` verified 2026-10-18 with real Deepgram/OpenAI/Cartesia (1.47 s end-to-speech p50). `engine=realtime` verified with real OpenAI gpt-realtime-translate (0.96 s p50 from speech start); transcription text occasionally misses spaces between streamed pieces. Bridge-call endpoint dials both parties, handled by the interpreter. SIP transfer and dialing implemented with mocks only.

#### Two-participant sessions

An interpreter agent with `engine=cascaded` can conduct a two-person call with per-participant language and voice profiles. Both parties join the same LiveKit room; each hears the other in their chosen voice and declared language.

**Create a session with a participant:**

`POST /voice/sessions` takes an optional `participant` in the body:

```json
{
  "agent_id": "...",
  "deployment_id": "...",
  "participant": {
    "language": "en",
    "voice_profile_id": "...",
    "display_name": "Alice"
  }
}
```

- `language` (required): one of the agent's `interpreter.languages`.
- `voice_profile_id` (optional): a `status=ready` voice profile with a `provider_voice_id`. Without it, the agent's voice is used.
- `display_name` (optional, ≤100 chars): how the interpreter names this speaker in transcripts.

Returns session response with `participants` object in `runtime_contract` containing the first participant's entry: `{language, voice_id, display_name}`.

**Add the second participant:**

`POST /voice/sessions/{session_id}/participants` takes the same `ParticipantIn` body.

Returns:

```json
{
  "session_id": "...",
  "conversation_id": "...",
  "room_name": "...",
  "participant": {
    "language": "ko",
    "voice_id": "...",
    "display_name": "Bob"
  }
}
```

Validation and errors:
- `404 voice_session_not_found`: session does not exist.
- `409 session_not_active`: session is not `created` or `running`.
- `409 participants_not_enabled`: no first participant; call `POST /voice/sessions` with a `participant` first.
- `409 session_full`: session already has 2 participants (max is 2).
- `422 invalid_participant_language`: participant language not in `interpreter.languages`.
- `422 participants_require_cascaded_interpreter`: agent is not an interpreter or uses `engine=realtime`.
- `409 selected_voice_unavailable`: voice profile is not `status=ready` or lacks a `provider_voice_id`.
- `409 voice_provider_mismatch`: the voice's TTS provider record names a different provider than the session's TTS provider (checked only when both are known; see [Translation sessions](#choosing-the-tts-provider)).
- Runtime-side `422 invalid_participant_voice`: the provider voice id is not `^[A-Za-z0-9_-]{1,128}$`.

**Status:** Verified 2026-10-18 with Korean ↔ English (0.8–1.4 s end-to-speech) on Cartesia. Fixed Pipecat shared-resampler crosstalk. *Note 2026-11-05:* per-listener routing (RM-001/002: a private listen room per person, `connection.listen`, track `translation-for-<identity>`, and one TTS per listener) landed after this check, so the line "everyone hears all audio" no longer describes the code. The rebuilt routing is mock-tested only (see [Status](status.md)); `smoke-test.md` re-checks it.

#### Agent translation (assistant + translator)

An agent with `mode=agent` can work in its own language while callers speak others via a `translation` field. The agent's reasoning, tools, and knowledge stay in the working language; inbound and outbound speech are automatically translated.

`POST /agents` or `PATCH /agents/{id}` with `translation` (on a non-interpreter agent):

```json
{
  "translation": {
    "working_language": "en",
    "caller_languages": ["ru"]
  }
}
```

- `working_language` (required): agent's language for LLM, tools, and knowledge; must be in `INTERPRETER_LANGUAGES`.
- `caller_languages` (required): list of 1–4 distinct codes callers speak; must all be in `INTERPRETER_LANGUAGES`, cannot include `working_language`.

Validation rules:
- `translation` requires `mode=agent` (`422 translation_not_allowed_for_interpreter` if `mode=interpreter`).
- If all caller languages and the working language are in Deepgram's code-switching set (`en`, `es`, `fr`, `de`, `hi`, `ru`, `pt`, `ja`, `it`, `nl`), automatic per-utterance language detection is enabled. Otherwise, exactly one (fixed) caller language is allowed; multiple without automatic detection raises `422 invalid_translation_config`.
- All language codes must be distinct and in the supported set.

Errors:
- `422 translation_not_allowed_for_interpreter`: agent mode is `interpreter`.
- `422 invalid_translation_config`: field types invalid, unknown language, >4 caller languages, duplicates, working language in caller list, or multiple fixed languages without automatic detection capability.

The LLM model for translation defaults to `gpt-4.1-nano`; override with `provider_settings.llm.translation_model` in Runtime.

**Status:** Verified 2026-10-18 with English receptionist agent and Russian caller (n=2); end-to-speech p50 1.9 s.

#### TranscriptFinal payload fields for interpreter and translator modes

For interpreter and agent+translator modes, `TranscriptFinal` events emit language and translation metadata in the `payload`:

**User utterances in interpreter mode (role=user):**
- `speaker_user_id`: LiveKit participant ID of the speaker.
- `primary_language`: spoken language (detected or declared).
- `target_language`: language into which the translator repeats.
- `speaker_name`: optional display name from the session.

**Agent translations in interpreter mode (role=assistant):**
- `speaker_user_id`: whom the interpreter is repeating.
- `source_language`: the speaker's language.
- `target_language`: language of the spoken translation.
- `primary_language`: language of the output translation.
- `speaker_name`: optional display name from the session.

**Inbound user speech in agent+translator mode (role=user):**
- `translated_text`: text in the working language (after inbound translation).
- `primary_language`: caller's language.
- `working_language`: agent's language.
- `user_id`: LiveKit participant ID.
- `speaker_name`: optional.

**Agent response in agent+translator mode (role=assistant):**
- `original_text`: agent's response in the working language.
- `translated_text`: response in the caller's language (if translation succeeded; original if it failed).
- `primary_language`: language of the spoken output.
- `working_language`: agent's language.
- `speaker_name`: optional.

#### LatencyMeasured metrics for translation

Interpreter and agent+translator pipelines emit additional metrics in `LatencyMeasured` events:

| Metric | Stage | Meaning |
| --- | --- | --- |
| `translation_ms` | cascaded | End-to-end: speech end → translated audio (STT → LLM → TTS) |
| `translation_first_chunk_ms` | cascaded | Time to first translated text output (STT → LLM) |
| `realtime_first_audio_ms` | realtime | Time from speech start to first audible output |
| `inbound_translation_ms` | agent+translator | Caller speech → working language (LLM translation step) |
| `outbound_translation_first_chunk_ms` | agent+translator | Agent reply start → first translated text (LLM translation step) |

**Status:** Runtime emits these in TR-040 (agent+translator), TR-010 (realtime engine), and TR-004 (cascaded). Platform stores and reports them in analytics. Payload shape is provisional pending Runtime emission.

#### Audio ownership

Every audio track in an interpreter room has one owner and an explicit audience. Enforcement is
server-side (Runtime and the LiveKit server API), so it holds for any client (browser, phone, API)
without trusting it.

| Track | Published by | Who may hear it |
| --- | --- | --- |
| A's microphone | Participant A | Runtime only |
| B's microphone | Participant B | Runtime only |
| A's words in B's language (`translation-for-<B>`) | Runtime | B only |
| B's words in A's language (`translation-for-<A>`) | Runtime | A only |
| Agent replies (single-caller modes) | Runtime | the caller |

1. A participant never hears their own microphone or their own translation.
2. Participants do not hear each other's original audio unless `hear_original` (or `dubbing`) is set.
3. Single-participant sessions (agent, agent + translator) publish one track that the caller hears.

### Translation sessions (`/translations`)

Reviewed against the code 2026-11-05 (`platform/src/platform/translations.py`, `sessions.py`,
`tenancy.py`, `limits.py`). **Implemented and tested with mocks; not run with real providers**
(see [Status](status.md) and [Smoke test](smoke-test.md)).

The primary way to start a conversation between two people. No agent, prompt or LLM to
configure: the Platform keeps a hidden, system-managed interpreter agent per organization and
exact interpreter configuration, and reuses the normal session path. The Runtime therefore
receives `mode="interpreter"`, an empty `system_prompt`, no tools and no knowledge.

```http
POST /translations
```

```json
{
  "languages": ["ko", "en"],
  "creator": {"language": "ko", "voice_profile_id": null, "display_name": "Min"},
  "latency_mode": "balanced",
  "hear_original": false,
  "dubbing": null,
  "glossary": null,
  "stt_provider_id": null,
  "llm_provider_id": null,
  "tts_provider_id": null
}
```

`tts_provider_id` (optional) selects an organization's enabled `tts` provider record (for
example ElevenLabs); omitted, Runtime uses its default TTS provider (Cartesia). See
[Choosing the TTS provider](#choosing-the-tts-provider). `latency_mode` is `instant`, `balanced` (default) or `accurate` (otherwise `422` validation
error); `hear_original` defaults to `false`; `dubbing` (optional, default off) keeps the other person's real
voice quietly under the translation: `true` or a preset name (`subtle`, `balanced`, `cinematic`, `present`), see
[dubbing](dubbing.md); it replaces `hear_original` (both set is `422 invalid_interpreter_config`) and, unlike
`hear_original`, keeps private listening, because the Runtime mixes the voice in and nobody receives
the other person's microphone; `glossary` follows the [glossary](#glossary-interpreterglossary-translationglossary)
rules, but any invalid glossary or other config failure surfaces here as the generic
`422 invalid_interpreter_config`. The response is the same as `POST /voice/sessions`
(`session_id`, `conversation_id`, `connection`, `session_token`, ...); the creator's token is a
session token (`vls_`). Member+ credential required.

| Rule | Error |
| --- | --- |
| `languages`: exactly two distinct supported codes, and a valid glossary/latency config | `422 invalid_interpreter_config` |
| `creator.language` is one of the pair | `422 invalid_participant_language` |
| `creator.voice_profile_id` is a `ready` voice of the caller's organization with a `provider_voice_id` | `409 selected_voice_unavailable` |
| `tts_provider_id` is not an enabled `tts` provider record of the caller's organization (foreign, missing, wrong kind or disabled: one identical answer) | `422 invalid_tts_provider_id` |
| `stt_provider_id` / `llm_provider_id` likewise (an enabled `stt` / `llm` record of the organization) | `422 invalid_stt_provider_id` / `422 invalid_llm_provider_id` |
| The creator's voice belongs to a different TTS provider than the selected one | `409 voice_provider_mismatch` |
| More than `PLATFORM_MAX_TRANSLATION_CONFIGS` (default 50, `0` disables) NEW configurations created by the organization in the last 24 h | `429` `too_many_translation_configs` (object body) |
| Organization already has `PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG` active sessions | `429` `too_many_active_sessions` (object body, see [Authentication](#authentication)) |
| More than `PLATFORM_SESSION_CREATE_RATE` creations per minute for this key | `429` `rate_limited` with `Retry-After` |
| `RUNTIME_API_URL` set but `PLATFORM_RUNTIME_TOKEN` empty | `503 runtime_auth_not_configured` |
| A required provider reported `down` by Runtime's health check | `503` `provider_unavailable` |

The rate limit is a dependency, so it is evaluated before the body is validated. Then, in
order: pair and creator language, the provider record, the active-session cap, the configuration
bound, and the shared session path. The hidden agent is only flushed, never committed on its
own: it is committed together with the session, so a refused request (any of the errors above)
leaves no agent, conversation or session row behind. Verified on SQLite and, on 2026-11-05, on a real
PostgreSQL 16.2 (a refused request left 0 new hidden agents and 0 new sessions).

- The engine is always `cascaded`, set explicitly, so `latency_mode: "instant"` never switches
  to the one-speaker `realtime` engine. Only the cascaded engine takes a language and a voice
  per speaker.
- `direction` is `auto` when both languages are in Deepgram's code-switching set, otherwise
  `one_way` (for example Korean). Each person has their own speech lane in their declared
  language, so detection is not used to decide who is speaking.
- Each person hears only the other's translation (private listening: a separate listen room
  `connection.listen`, track `translation-for-<identity>`), unless `hear_original`.
- A creator without `voice_profile_id` is spoken for with the Runtime's default voice of the
  selected TTS provider (`CARTESIA_VOICE_ID` or `ELEVENLABS_VOICE_ID`).

#### Choosing the providers (STT, translation, voice)

`stt_provider_id`, `llm_provider_id` and `tts_provider_id` each select an organization's enabled
provider record for that slot; an omitted slot uses Runtime's default (`deepgram`,
`openai-compatible`, `cartesia`). The supported non-default stack is **ElevenLabs STT
(`elevenlabs`), Gemini translation (`gemini`) and ElevenLabs voice (`elevenlabs`)**:

- Runtime needs only `ELEVENLABS_API_KEY` (one key for STT and voice) and `GEMINI_API_KEY`;
  `DEEPGRAM_API_KEY` and `OPENAI_API_KEY` are not required for such a session. Each selected
  provider's credentials are required, and a provider without a credential mapping is refused at
  session start.
- Gemini is called through its OpenAI-compatible endpoint
  (`GEMINI_BASE_URL`, default `https://generativelanguage.googleapis.com/v1beta/openai/`);
  `OPENAI_BASE_URL` is not used for it.
- **No model is built in for Gemini.** Set the `llm` record's `model` setting or
  `GEMINI_TRANSLATION_MODEL`; without one the session fails to start. ElevenLabs STT uses
  Pipecat's own default model unless the `stt` record sets `model`. Latency modes keep setting turn
  pause and chunking only.
- ElevenLabs STT has no code-switching language: each person's lane uses their declared
  language, which is what `/translations` already does.
- Provider health reports `stt/elevenlabs` (same result as `tts/elevenlabs`, one key) and
  `llm/gemini`; the preflight checks the selected providers.
- Mock-tested only; not run against the real ElevenLabs STT or Gemini services.

Each id is part of the hidden agent's key (only when set), so different providers are different
hidden agents.

**TTS.** With `tts_provider_id` set, Platform stores it on the hidden agent and the normal session path
sends `providers.tts` (for example `elevenlabs`) plus the record's `model` and `language`
settings in `provider_settings.tts`, so **`/translations` can run on ElevenLabs**. Without it
no `providers` block is sent and Runtime defaults to `deepgram`, `openai-compatible`,
`cartesia`. The provider id is part of the hidden agent's key, so a different provider is a
different hidden agent (an omitted provider hashes as it did before this field existed).
Mock-tested only; never run against the real ElevenLabs service.

**Voice/provider check.** The effective TTS provider is the agent's (or hidden agent's) provider
record; when there is none, Platform compares against Runtime's default, `cartesia`
(`health.DEFAULT_PROVIDERS["tts"]`). When a voice carries a `tts_provider_id` whose provider name differs, Platform
answers `409 voice_provider_mismatch`. It applies to the agent voice profile and the creator's
voice when a session opens, to `POST /translations/{id}/invite` and
`POST /voice/sessions/{id}/participants`, and to phone participants. So an ElevenLabs-registered
voice on a session without a TTS provider is `409 voice_provider_mismatch`. It is skipped only
when the voice has no `tts_provider_id` or its provider row cannot be resolved in the organization;
there a wrong voice id still shows up only as failed or silent synthesis. It
compares provider names, and does not check that the id exists at the provider.

#### Hidden-agent bound and concurrency

The number of NEW hidden configurations an organization may create in a rolling 24 hours is bounded
by `PLATFORM_MAX_TRANSLATION_CONFIGS` (default 50, `0` disables); one more gives
`429` `{"code": "too_many_translation_configs", ...}`. Reusing an existing configuration always
works, and rows older than 24 h do not count. The count and the insert are not atomic, so
concurrent creators can overshoot the bound. The provider preflight (a network call) runs before
the hidden agent is flushed, so no row lock is held while it waits. The flushed row stays
uncommitted until `open_voice_session` commits, with no await in between. Checked on PostgreSQL
16.2 (`tests/test_postgres_translations.py`): eight simultaneous creates produce one hidden
configuration, and a one-second provider check leaves `/health` responsive.
The glossary is hashed in a canonical (sorted) order, so a reordered glossary reuses the same
hidden agent while the stored and sent config keeps the caller's order. Two concurrent first uses
of one configuration are resolved by the unique `(organization_id, system_key)` constraint: the
loser rolls its transaction back and re-selects the winner's row. **Tested on SQLite
(file-backed threads) and on a real PostgreSQL 16.2 on 2026-11-05: 12 barrier-released concurrent
`POST /translations` for one new configuration all returned 201 and created exactly one hidden agent
and 12 sessions.** Migration `0014` (`agents.system` boolean NOT NULL DEFAULT false, `agents.system_key`
varchar NULL, unique `uq_agents_org_system_key` on `(organization_id, system_key)`) ran on that
PostgreSQL with upgrade, downgrade to `0013_session_last_event` and upgrade. See
[Status](status.md#postgresql-check-2026-11-05).

#### Hidden agents are invisible and immutable through the public API

The hidden agent has `agents.system = true` and a `system_key` (`translator:` plus a 32-character
SHA-256 prefix of the sorted interpreter JSON), unique per `(organization_id, system_key)`
(migration `0014_system_agents`). The same configuration reuses the same row; a concurrent first
use is race-free (the unique constraint makes the loser's insert fail, it rolls back to a
savepoint and re-selects the winner's row). Public agent routes treat a system agent as missing:

| Request | Result for a system agent |
| --- | --- |
| `GET /agents` | not listed |
| `GET`, `PATCH`, `DELETE /agents/{id}` | `404 agent_not_found` |
| `GET`/`PUT /agents/{id}/tools`, `GET`/`PUT /agents/{id}/knowledge` | `404 agent_not_found` |
| `POST /voice/sessions` with its `agent_id` | `404 agent_not_found` |
| `POST /deployments`, `POST /campaigns` with its `agent_id` | `422 invalid_agent_id` |

The responses are the same as for a random unknown id, so the id's existence is not leaked. Only
`POST /translations` may open a session on a system agent (internally `allow_system=True`).
Another organization's id is also `404`. Two observations from the code: conversation reads
(`GET /conversations`, `/conversations/{id}`) still carry the hidden agent's `agent_id`, and
resolving it through `/agents/{id}` gives `404`; and an `agent_id` filter on `GET /conversations`
is not blocked for system agents.

**Legacy interpreter agents with prompts.** An ordinary agent with `mode=interpreter` may still
carry a `system_prompt`. Platform still sends it in the session contract (a test pins this), but
the Runtime's interpreter pipelines never use it: `build_livekit_pipeline` returns the
interpreter pipeline before any agent-prompt code, and the translation prompt is built into
`interpreter.py`. Do not expect a prompt on an interpreter agent to change behaviour.

#### Inviting the second person and reading sessions

```http
POST /translations/{id}/invite
```

Body: `{"language": "en", "voice_profile_id": null, "display_name": null, "replace": false}`.
Seats the second person as a browser guest and returns `participant_identity`, `connection`, a
guest `session_token` and `replaced_participant_identity` (null unless `replace` was used). It
calls the same function as `POST /voice/sessions/{id}/participants`.

**Retry safety.** Send an `Idempotency-Key` header (8 to 128 characters, generated once per
invite attempt by the client). The seat is stored before the Runtime is told, so a lost response,
a double-click or a Runtime error leaves exactly one guest seat. A retry with the same key and
language returns the same `participant_identity` with fresh LiveKit tokens and a new
`session_token`; every credential issued earlier stays valid. The same key with another language
is `409 idempotency_key_conflict`.

**No silent replacement.** Once a guest holds the seat, an invite without the matching key is
`409 guest_already_invited` and changes nothing. `replace: true` is the deliberate way to swap
the guest: it revokes the previous guest's session tokens, frees their Runtime slot, removes
them from the LiveKit session and listen rooms (best effort; their LiveKit token otherwise
expires after 10 minutes) and seats a new identity.

| Condition | Error |
| --- | --- |
| Guest already seated, key missing or different, no `replace` | `409 guest_already_invited` |
| `Idempotency-Key` reused with another language | `409 idempotency_key_conflict` |

| Condition | Error |
| --- | --- |
| Session is not a translation (an ordinary voice session, a foreign organization's, unknown) | `404 translation_not_found` |
| Session not `created`/`running` | `409 session_not_active` |
| Two people already seated and the second seat is not a browser guest (e.g. a phone) | `409 session_full` (maximum is 2) |
| Same language as the first person | `422 participant_languages_must_differ` |
| Language not in the pair | `422 invalid_participant_language` |
| Voice not ready | `409 selected_voice_unavailable` |
| Voice belongs to a different TTS provider than the session's | `409 voice_provider_mismatch` |
| Runtime cannot be told about the new participant | `503` with body `{"code": "runtime_unavailable", "message": ...}` |

Credentials: the **creator's session token** or a member+ key. A guest's token gets
`403 forbidden` (the route is creator-only). To reach the second person by phone use
`POST /voice/sessions/{id}/phone-participants`.

`GET /translations` (member+, newest first, `?limit=`, clamped to 1 to 200, default 50) and
`GET /translations/{id}` (member+ key, or the creator's or a guest's token for that session)
return `id`, `conversation_id`, `status`, `started_at`, `ended_at`, `languages`,
`latency_mode` and `participants` (`identity`, `language`, `display_name`; never provider voice
ids). `GET /translations/{id}` returns `404 translation_not_found` unless the session belongs to
a system agent of the caller's organization. Transcripts are read through
`GET /conversations/{conversation_id}/messages`.

#### Runtime: explicit `mode`

A `POST /sessions` to the Runtime that omits `mode` is logged as a warning
(`session <id> started without an explicit mode; defaulting to 'agent'`), because an omitted
mode silently becomes an answering AI agent. Platform always sends it. The request is not
rejected.

### ElevenLabs (live TTS)

Reviewed against the code 2026-11-05. Mock-tested, and the **ElevenLabs layer was run against the
real API on 2026-11-05** (free-tier key, 15 of 15 automated checks passed, objective measurements only,
audio quality not judged; table in [Smoke test](smoke-test.md#11-record-results-here), summary in
[Status](status.md#elevenlabs-real-provider-run-2026-11-05)). **Observed working:** REST preview in
en/ru/ko; headerless 16-bit PCM from `pcm_24000`; `eleven_multilingual_v2` without `language_code` for
ko; live WebSocket TTS (first audio 0.2 to 0.3 s, 24 kHz, `pcm_24000` evidently accepted on the
WebSocket although the docs do not state it); voice and language switch with reconnect; interruption of
an in-flight context; clean shutdown; invalid key. **Not real-verified:** a full two-way `/translations`
call (needs Deepgram, OpenAI, LiveKit, two participants), per-participant voices inside a real session,
full-pipeline latency, audio quality, voice-id account scoping, behaviour beyond the free tier.

**Invalid or expired key (found by the real run).** ElevenLabs accepts the WebSocket handshake for an
invalid key and then closes with `1008 (policy violation) Invalid API key`. Runtime classifies a
`ConnectionClosed` with credential wording as `auth` (`provider_failures.classify_error`) and its
ElevenLabs service subclass reports it at once as a permanent error, so the call ends with an auth
`ProviderError` after about 1.3 s instead of reconnecting three times and ending as `other` after
about 3.3 s. The older mocked test `test_rejected_api_key_on_connect_...` models a handshake `401`,
which the real websocket does not do; it stands for a REST-style rejection only.

- **Selection.** Runtime has a TTS provider named `elevenlabs` (`speech_providers.py`), built on
  Pipecat's `ElevenLabsTTSService` (WebSocket). It is chosen per agent by creating a provider
  record `{"kind": "tts", "provider": "elevenlabs", "configuration": {...}}` (`POST /providers`,
  admin) and setting the agent's `tts_provider_id`; Platform sends `providers.tts = "elevenlabs"`
  in the session contract. Without a TTS provider record the default is `cartesia`.
  `POST /translations` takes the same record as `tts_provider_id`; the older agent-first path
  (`POST /agents` with `mode=interpreter` and `engine: "cascaded"` stated explicitly, then
  `POST /voice/sessions`) also works.
- **Settings.** The record's `configuration` may carry `model` and `language` (strings); other
  keys are dropped by Platform, and Runtime rejects an unknown key at session start. Default
  model `eleven_flash_v2_5`. The per-call language is applied by Pipecat only when the model
  takes a `language_code` and lists the language; otherwise it is logged and dropped.
- **Capabilities.** `streaming` and `transport_destination`. **No `voice_clone`.**
- **Environment (Runtime).** `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` (default voice, used
  for a participant without a voice). `PipelineConfig.tts_api_key` and `tts_voice_id` follow
  `providers["tts"]`. Only the selected provider's credentials are required: a Cartesia session
  does not need ElevenLabs variables and an ElevenLabs session does not need `CARTESIA_*`. The
  `CARTESIA_*` variables and `cartesia_*` fields are unchanged. A session `voice_id` replaces the
  default voice of the selected provider only. Keys are never stored in Platform records.
  `scripts/elevenlabs_smoke.py` additionally reads the key from `ELEVENLABS_API_KEY_FILE` / `--key-file` / `~/Developer/11labs-api-key.txt` (never printed; rotate by replacing the file or exporting a new key; nothing is stored in the repository). The Runtime itself reads only the environment variables.
- **Voices are provider-specific.** A voice record carries a `provider_voice_id` that only means
  something to one provider. A Cartesia voice id will not work with ElevenLabs and vice versa.
  Platform now rejects a voice whose `tts_provider_id` names a different provider than the
  effective TTS provider with `409 voice_provider_mismatch` (see above), but only when both are
  known, and it never checks that the id exists at the provider. Create voices with
  `tts_provider_id` pointing at the ElevenLabs record so previews and the check use the right
  provider.
- **Voice id validation.** Provider voice ids are interpolated into provider URL paths (a
  `../../v1/user` id was confirmed to collapse to another ElevenLabs endpoint with httpx). They must
  match `^[A-Za-z0-9_-]{1,128}$` (full match) before use: Runtime `POST /sessions` answers
  `422 invalid_voice_id` when `providers.tts == "elevenlabs"` and `voice_id` fails it (checked after
  the token check, before any state is stored; Cartesia or unset providers are unaffected). Platform
  itself accepts any `provider_voice_id` up to 200 characters, so such a voice saves fine and fails
  at session start. Also in `/tts/preview` (a bad id gives
  `502 tts_provider_error`, no request is made), in the apology renderer (exit `2`), in the live
  service factory (the default voice; session start fails), and for every participant voice at
  Runtime's participant directory (`422 invalid_participant_voice`, for Cartesia voices too, on
  session start, participant add and update).
- **Fail closed.** Every registered TTS provider must declare its credential fields
  (`register_tts_credentials` in `pipeline.py`; `()` means none needed). An unmapped provider fails
  at session start instead of running without a key.
- **No cloning or deletion.** `POST /voices/clone` in Runtime answers `422 unsupported_tts_provider`
  for any provider other than `cartesia`, and `DELETE /voices/elevenlabs/{id}` answers
  `404 unknown_provider`, both before any provider call. `POST /voices/{id}/provision` on a voice
  whose provider is ElevenLabs therefore fails with `unsupported_tts_provider`.
- **Reconnects.** Changing the voice or language of a live ElevenLabs service reconnects its
  WebSocket (both are part of the URL; stated in the factory docstring and pinned by a mocked
  test). The two-person pipeline keeps one TTS per listener (fixed language and output
  destination) and a listener always hears the same other speaker, so the voice normally
  changes at most once per lane, on its first utterance when that speaker has a non-default voice
  (read from `interpreter.py`; the cost of that reconnect was never measured).
- **Preview.** `POST /voices/{id}/preview` works for ElevenLabs voices through Runtime's
  `/tts/preview`; see [Voice preview](#voice-preview) for the language-code rules and the
  unverified PCM assumption.
- **Apology clips.** `runtime/scripts/render_apologies.py --elevenlabs --confirm-billable`
  renders failure-apology WAVs (billable). It exits `2` before any request when
  `--confirm-billable` is missing, `ELEVENLABS_API_KEY` is unset, a language has no voice
  (`--voice-id` or `--voice LANG=ID`), the rate is not one of 8000, 16000, 22050, 24000, 44100, or
  a `--voice` value is not `LANG=ID`.
- **Lockfiles and uv.** uv rewrote the lock `revision` from 3 to 5, so contributors and CI need
  a uv that understands lock revision 5 (the locks were produced with uv 0.13.7). An older uv may
  refuse the file or re-resolve it; which minimum version works is not verified.
- **Dependency.** The `voice` extra of `runtime/pyproject.toml` now lists
  `pipecat-ai[livekit,deepgram,openai,cartesia,elevenlabs]`. `runtime/uv.lock` and the root `uv.lock`
  were regenerated with uv 0.13.7; the diff is that extras line plus uv's lock `revision` 3 to 5
  and no package version changed.

**Per the official ElevenLabs docs, fetched twice on 2026-11-05** (two independent fetches agree):
- TTS model ids: `eleven_v4`, `eleven_v4_turbo`, `eleven_v3`, `eleven_v3_conversational`,
  `eleven_multilingual_v2` (29 languages), `eleven_flash_v2_5` (32 languages, includes ko, ru, en),
  `eleven_flash_v2` (English only). `eleven_turbo_v2_5` and `eleven_turbo_v2` are deprecated.
- REST `POST /v1/text-to-speech/{voice_id}`; header `xi-api-key` (shown in code examples only);
  `model_id` defaults to `eleven_multilingual_v2`.
- `output_format` allows `alaw_8000`, `mp3_*`, `opus_*`, `pcm_8000/16000/22050/24000/32000/44100/48000`,
  `ulaw_8000`, `wav_*`; 44.1 kHz PCM/WAV needs Pro tier or above.
- REST `language_code`: "If the model does not support the provided language code, it will be
  ignored. Not supported for multilingual_v2 models."
- WebSocket: `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input` with
  `close_context`, `close_socket` and `flush` messages.
- Pricing page (check <https://elevenlabs.io/pricing/api> before relying on any number): $0.08 per
  1K characters for v4, v3 and v2 Multilingual; $0.04 per 1K for v4 Turbo, v3 Conversational and
  Flash/Turbo.

**Still not stated by any official page:** that `pcm_*` REST responses are headerless raw PCM
(**observed** 2026-11-05: they are); that `pcm_*` is accepted on the WebSocket endpoint (its
`output_format` is untyped "Any type"; **observed working** 2026-11-05 through Pipecat, at 24 kHz). Still
unverified: whether a voice id is scoped to an account (across accounts); and
latency figures (the pages disagree: Flash about 100 ms vs about 75 ms). Library voices are
reportedly not available through the API on the free tier (from the earlier fetch only). Note that
`eleven_v4` and `eleven_v4_turbo` are absent from Pipecat 1.12.0's language table, so Pipecat and
our preview never send `language_code` for them. Runtime's default model and language table come
from Pipecat 1.12.0, not from these docs.

### Glossary (`interpreter.glossary`, `translation.glossary`)

An optional per-agent list of terms that translation must respect. It is stored inside the
agent's existing `interpreter` or `translation` JSON (no migration) and reaches Runtime in the
session contract.

```json
"glossary": [
  {"source": "residence permit",
   "translations": {"ru": "вид на жительство", "ko": "체류 허가"},
   "note": "optional hint"},
  {"source": "Voice Labs", "keep": true}
]
```

- At most 200 entries. `source` and each translation are 1–100 characters, `note` at most
  200, with no control characters; sources are unique ignoring case. Language codes use the
  interpreter language set. An entry has either `translations` or `keep: true`.
- `source` is the term in any language without a `translations` entry (usually the working
  language). en → ru uses the Russian term; ru → en maps it back to `source`. `keep: true`
  terms (names, brands) are never translated.
- Runtime puts the relevant entries into the translation prompt: all of them when 6 or fewer
  apply, otherwise only those whose term appears in the utterance, at most 12 per call.
- The same terms are sent to Deepgram as `keyterm` boosting (at most 100) so speech
  recognition hears them correctly.
- Applies to the cascaded engine and to agent + translator mode. The realtime engine
  (gpt-realtime-translate) has no instruction or glossary field and ignores it.
- An invalid glossary is rejected with 422 (`invalid_glossary: <reason>` for `translation`).

## Shareable-link calls (`/calls`)

Opt-in (`PLATFORM_PUBLIC_CALLS=1`, otherwise 404). Full description, configuration and security model: [calls](calls.md).

| Route | Auth | Body → response |
|---|---|---|
| `POST /calls` | none (rate limited per address) | `{languages: [mine, theirs], name?, voice_profile_id?, latency_mode?, experience?: "ai_only"\|"cinematic"}` → the same reply as `POST /translations`: `session_id`, `conversation_id`, `connection {url, token}`, `session_token` |
| `POST /calls/{session_id}/invite` | creator's session token | → `{code, expires_at, ttl_seconds}`; a new code replaces the old one; seats nobody |
| `GET /calls/invites/{code}` | none (rate limited) | → `{from, me, them, expires_at}`; 404 `invite_not_found`, 410 `invite_expired` / `call_ended` |
| `POST /calls/invites/{code}/join` | none (rate limited) | `{name?}` → `{participant_identity, connection, session_token, session_id, conversation_id, me, them, from}`; repeating it with the same code returns the same seat; 409 `guest_already_invited` for any other code |
| `POST /calls/{session_id}/end` | either person's session token | → `{status}`; ends the call for both (idempotent). `POST /voice/sessions/{id}/end` stays creator-only |

`experience: "cinematic"` sets `interpreter.dubbing = "cinematic"` (the `balanced` profile with the original at -24 dB while the translation speaks).
Providers come from `PLATFORM_CALLS_PROVIDERS`, never from the request.

Two-person calls use `interpreter.speaker_language`: `declared` fixes recognition and
translation to the speaker’s declared language (Runtime default when omitted); `auto`
resolves each utterance from STT detection, with script correction and declared-language
fallback. Speech already in the listener’s declared language is played as-is, without
translation. `/calls` and handoffs default to `auto`; set
`PLATFORM_CALLS_SPEAKER_LANGUAGE=declared` to restore fixed recognition. Invalid environment
values fall back to `auto`. Listener languages and voice routing stay unchanged.

### Demo chat retrieval and citations

`POST /demo/agents/{key}/chat` additionally returns `sources`, an array of
`{n, title, heading, url, snippet}`. Only numbered sources cited in `reply` appear;
`snippet` is at most 240 characters. Unknown numeric citation markers are removed.
Existing `conversation_id`, `reply`, and `handoff_suggested` fields are unchanged.
`PLATFORM_CHAT_KNOWLEDGE_MODE=rag` (default) retrieves using the last two user turns;
`inline` retains the full-knowledge prompt and returns an empty sources array.
Embedding failures degrade to keyword retrieval.

`POST /agents/{agent_id}/knowledge/reindex` requires admin permission, as does
setting agent knowledge. It refreshes stale or missing chunks for enabled assigned
documents and returns `{documents, chunks, embedded}` (current index counts).
`embedded` counts chunks with stored embeddings; unavailable providers leave
keyword-searchable chunks. Disabled/unassigned documents are excluded at query time.

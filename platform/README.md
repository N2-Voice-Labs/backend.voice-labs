# platform.voice-labs

The Voice Labs control plane. It owns durable agent, provider, conversation, message,
and voice-session state. Realtime audio stays in `runtime.voice-labs`.

Run locally with `python -m alembic upgrade head` followed by
`uvicorn src.platform.main:app --reload`.

## Security-relevant settings

| Variable | Default | Effect |
| --- | --- | --- |
| `PLATFORM_API_KEYS` | unset | Comma-separated admin keys. Unset (and no flag below): every authenticated route answers 401 `auth_not_configured`; public routes, org `vlk_` keys and session tokens still work. |
| `PLATFORM_AUTH_DISABLED` | unset | `1` turns authentication off when no admin keys are set: every caller is the `dev` owner/superadmin. Local development only; logged as a warning at startup. |
| `PLATFORM_DIAL_PER_ORG_HOUR` | `30` | Outbound phone dials per organisation per hour (in-memory, per process; `0` disables) on `POST /voice/sessions/{id}/phone-participants` and `POST /deployments/{id}/bridge-call`. Over the limit: 429 with `Retry-After`. |
| `PLATFORM_PUBLIC_CALL_MAX_SECONDS` | `1800` | Hard duration cap for calls opened through public `/calls` and handoffs; the reaper fails them with reason `public_call_max_duration`. |
| `LIVEKIT_PUBLIC_URL` | `LIVEKIT_URL` | LiveKit URL given to browsers in session `connection`; server-side calls keep using `LIVEKIT_URL`. |
| `PLATFORM_LOG_LEVEL` | `INFO` | Root log level, applied only when logging is not already configured (uvicorn's own config is left alone). |
| `PLATFORM_TRUST_PROXY` | - | Deprecated and ignored (logged at startup). The client address is `request.client.host`; behind a proxy, set uvicorn `--forwarded-allow-ips`. |

Behaviour to know: a session token (`vls_`) can no longer add phone participants; only an
org API key with role member or above (or a superadmin key) can. Rate-limit buckets that
have refilled are dropped once the table passes 10,000 keys.

Tests use isolated SQLite for fast feedback. Local application data is intended to use
PostgreSQL via `PLATFORM_DATABASE_URL` and Alembic migrations.

An opt-in PostgreSQL concurrency test verifies that simultaneous deliveries of the same
transcript event persist one event and one message. Run it only against a disposable
database that has been migrated to the current Alembic head:

```sh
export PLATFORM_TEST_POSTGRES_URL='postgresql+psycopg://user:password@localhost:5432/voice_labs_test'
export PLATFORM_DATABASE_URL="$PLATFORM_TEST_POSTGRES_URL"
uv run --locked --extra test python -m alembic upgrade head
uv run --locked --extra test pytest -q tests/test_postgres_concurrency.py
```

## Voices

Voices are created as `library` (existing provider voice id), `design`, or `clone`. See
`docs/contracts.md` (Voices) for the full contract. Platform never stores audio.

Cloning supports `cartesia` and `elevenlabs`:

- `POST /voices/clone` + `POST /voices/{id}/provision`: clone from a consented HTTPS
  `sample_reference`; Runtime downloads it. The provider is the voice's `tts_provider_id`
  record's provider (default `cartesia`).
- `POST /voices/clone/upload` (admin, `multipart/form-data`): clone from a recorded
  sample. Fields: `name`, `language`, `consent` (a JSON string of the `VoiceConsent`
  object: `confirmed`, `subject_name`, `statement`, `method`), `file` (`audio/*`,
  non-empty, at most 10 MiB, counted while reading), optional `description` and `provider`
  (`cartesia` | `elevenlabs`). When `provider` is omitted it is the provider of the
  organisation's oldest enabled TTS provider config that can clone, else `cartesia`.
  The bytes are streamed to Runtime `POST /voices/clone-upload` (90 s timeout) and never
  stored or logged; the voice ends `ready` (201) or `failed` with a `failure_reason` such
  as `clone_plan_upgrade_required`, `invalid_sample` or `runtime_unreachable` (502
  `voice_provisioning_failed`). Errors before any voice is created: `422
  voice_consent_required`, `422 invalid_sample`, `422 invalid_voice_language`, `422
  unsupported_tts_provider`. This route needs `python-multipart` (in `pyproject.toml`).
- `DELETE /voices/{id}` removes the provider voice through Runtime for the voice's own
  provider (including `elevenlabs`).

Session contract: an agent whose `voice_profile_id` voice was deleted or failed after it was
attached still starts sessions, with `voice_id: null` (Runtime's default voice) and a logged
warning. Explicit choices (a participant's `voice_profile_id`) and not-yet-ready, disabled or
foreign voices still return `409 selected_voice_unavailable`.

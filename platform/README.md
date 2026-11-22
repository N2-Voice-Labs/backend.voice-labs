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

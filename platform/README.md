# platform.voice-labs

The Voice Labs control plane. It owns durable agent, provider, conversation, message,
and voice-session state. Realtime audio stays in `runtime.voice-labs`.

Run locally with `python -m alembic upgrade head` followed by
`uvicorn src.platform.main:app --reload`.

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

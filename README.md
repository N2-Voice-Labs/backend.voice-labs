# Voice Labs Backend

Local consolidation of the Voice Labs control plane, realtime runtime, and evaluation
package. The original sibling repositories remain untouched; this directory is a
separate staging repository and is not connected to any remote.

## Projects

- `platform/` — FastAPI control plane, durable conversation/session state, provider
  references, and Runtime event ingestion. It does not process realtime audio.
- `runtime/` — FastAPI, Pipecat, and LiveKit realtime sessions and provider pipeline.
- `evals/` — deterministic speech evaluation metrics and tests; no private datasets or
  recordings are included.

Each member retains its own `AGENTS.md`, README, project manifest, lockfile, tests, and
configuration. Follow those member instructions when working within a project.

## Docs

See [`docs/`](docs/README.md) for contracts, status and plans.

## Checks

From this directory (the `pythonpath` pytest setting makes the existing `src.*`
test imports work when uv runs each workspace member):

```sh
uv run --directory platform --locked --extra test pytest -q
uv run --directory runtime --locked --extra voice --extra test pytest -q
uv run --directory evals --locked --extra test pytest -q
```

For deterministic cross-service acceptance, start the stack's local LiveKit service
and run this from the backend workspace using the local development credentials shown
in `stack.voice-labs/.env.example`:

```sh
LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
  uv run --directory runtime --locked --extra voice --extra test pytest -q \
  ../../stack.voice-labs/tests
```

This exercises synthetic media and stubbed speech providers; it does not verify
provider-backed STT/LLM/TTS. The real-provider acceptance test is opt-in and requires
the documented credentials and a consented audio fixture.

Run Platform migrations only with a disposable/local database configured through the
Platform settings:

```sh
uv run --directory platform --locked --extra test python -m alembic upgrade head
```

Runtime voice/provider checks require the documented local LiveKit and provider
configuration. Passing stubbed or synthetic-media tests does not establish real
STT/LLM/TTS behavior.

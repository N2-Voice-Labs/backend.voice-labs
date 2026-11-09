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

## Run with Docker

One command starts Postgres, LiveKit, Platform (with migrations) and Runtime. Requires
Docker with Compose v2 (Docker Desktop, or colima on macOS).

```sh
cp .env.example .env      # then fill the provider keys (see below)
make up                   # docker compose up -d --build
make ps                   # all services should become healthy; `migrate` exits 0
```

Already have keys in the sibling stack? Copy them yourself rather than retyping:
`cp ../stack.voice-labs/.env .env` and add the new variables from `.env.example`.

| Service  | URL / port                     | Notes |
|----------|--------------------------------|-------|
| Platform | http://localhost:8300 (`/docs`, `/health`, `/health/providers`) | auth is disabled while `PLATFORM_API_KEYS` is empty |
| Runtime  | http://localhost:8301 (`/health`) | single replica only |
| LiveKit  | ws://localhost:7880 (+ 7881/tcp, 7882/udp) | profile `local-livekit` (default in `.env.example`); pinned `livekit-server` v1.9.0, config in `docker/livekit.yaml` |
| Postgres | localhost:5432 (`voice_labs` / `voice_labs` / `local-only`) | data in the `postgres-data` volume |

Published ports are bound to `127.0.0.1` only; nothing is reachable from the network
unless you opt into the Cloudflare tunnel below.

Without provider keys everything starts, but `/health/providers` reports the missing
required providers (`deepgram`, `openai`, `cartesia`) as `down` ("API key not
configured"), overall health is `down`, and `POST /voice/sessions` returns
`503 provider_unavailable` (the Runtime worker needs `DEEPGRAM_API_KEY`,
`OPENAI_API_KEY` and `CARTESIA_API_KEY`; `ELEVENLABS_API_KEY` is optional). Set
`RUNTIME_TOKEN` to your own value outside local use; it is shared by both services.

Create an agent and a voice session:

```sh
AGENT=$(curl -s -XPOST localhost:8300/agents -H 'content-type: application/json' \
  -d '{"name":"Demo","system_prompt":"You are a helpful voice assistant.","enabled_languages":["en"]}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
curl -s -XPOST localhost:8300/voice/sessions -H 'content-type: application/json' \
  -d "{\"agent_id\":\"$AGENT\"}"
# -> {"session_id": ..., "connection": {"url": "ws://localhost:7880", "token": "..."}, ...}
```

(If you set `PLATFORM_API_KEYS`, add `-H 'authorization: Bearer <key>'`.)

LiveKit URLs: containers reach LiveKit at `ws://livekit:7880`, but the URL returned to
browsers is `LIVEKIT_PUBLIC_URL` (default `ws://localhost:7880`; Platform falls back to
`LIVEKIT_URL` when it is unset). LiveKit advertises `LIVEKIT_NODE_IP` (default
`127.0.0.1`) for WebRTC media, which works for a browser on the same machine, including
macOS/colima. To connect from another device, set both `LIVEKIT_NODE_IP` and
`LIVEKIT_PUBLIC_URL` to this host's LAN IP.

Frontend: point it at `http://localhost:8300` as the Platform API base URL, and serve it
from the origin in `PLATFORM_WEB_ORIGIN` (default `http://localhost:3000`) so CORS
allows it. The frontend joins the call with the `connection.url`/`token` returned by
`POST /voice/sessions`.

Other targets: `make logs [S=platform]`, `make down`, `make migrate`, `make reset`
(deletes the database volume, asks first), `make test` (the three pytest suites on the
host via uv). The stack does not use Redis/Valkey.

### Use the deployed frontend with this Mac (public demo)

Lets https://voice-lab.nomadic.fyi talk to this backend running on your Mac: the API
goes through a Cloudflare Tunnel, call media goes through LiveKit Cloud (no local
LiveKit, no open inbound ports).

1. **LiveKit Cloud.** Create a project at https://cloud.livekit.io and copy its
   WebSocket URL (`wss://<project>.livekit.cloud`), API key and API secret.
2. **Cloudflare tunnel.** Cloudflare Zero Trust -> Networks -> Tunnels -> Create a
   tunnel -> Cloudflared. Copy the tunnel token. Under *Public Hostname* add
   `voice-lab-api.nomadic.fyi` (or another hostname on your zone), Service type
   **HTTP**, URL `platform:8300`. (The `tunnel` container runs on the compose network,
   so use the service name `platform`, not `localhost`.)
3. **Configure.** `cp .env.public.example .env` and fill: `CLOUDFLARE_TUNNEL_TOKEN`,
   `LIVEKIT_URL` and `LIVEKIT_PUBLIC_URL` (the same `wss://` URL), `LIVEKIT_API_KEY`,
   `LIVEKIT_API_SECRET`, the provider keys (`DEEPGRAM_API_KEY`, `OPENAI_API_KEY`,
   `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`) and `RUNTIME_TOKEN` (`openssl rand -hex 32`).
   `PLATFORM_WEB_ORIGIN` (CORS) is preset to `https://voice-lab.nomadic.fyi`.
4. **Start.** `make public` (checks the token and `wss://` URLs, then starts the stack with
   `COMPOSE_PROFILES=tunnel`). Stop with `make public-down`.
5. **Frontend.** Redeploy it with `NEXT_PUBLIC_PLATFORM_URL=https://voice-lab-api.nomadic.fyi`.
6. **Verify.**

   ```sh
   curl https://voice-lab-api.nomadic.fyi/health
   curl https://voice-lab-api.nomadic.fyi/health/providers   # overall "ok", providers ok
   ```

   Missing or rejected provider keys show up here as `down`, and session creation
   fails with `503 provider_unavailable` instead of producing a silent call.

Voice cloning (needs a paid Cartesia plan; on the free tier Cartesia answers 402 and the
voice shows `failed` with `clone_plan_upgrade_required`):

- The sample must be an `https://` link the Runtime can download without logging in:
  audio content type (or `application/octet-stream`), at most 10 MB, on a public host.
  A presigned S3/R2 link works. Share pages (Google Drive, Dropbox previews) return HTML
  and fail with `sample_unavailable`.
- With `PLATFORM_AUTO_PROVISION_CLONES=1` (the compose default) the voice goes
  `pending` -> `processing` -> `ready` by itself after `POST /voices/clone`; Retry on a
  failed voice runs it again.
- Designed voices (`POST /voices/design`) are stored only: nothing creates them at a
  provider, so they stay `pending`.

Security notes:

- The API is **open**: the frontend sends no credentials, so anyone who learns the
  hostname can start calls billed to your provider keys. CORS only restricts browsers,
  not `curl`. Spend is capped by `PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG=3`,
  `PLATFORM_SESSION_CREATE_RATE=10` (per minute), `RUNTIME_MAX_SESSIONS=4`,
  `RUNTIME_MAX_SESSION_SECS=600` and `RUNTIME_IDLE_END_SECS=60`; tune them in `.env`.
  Use provider-side spending limits too, and keep the hostname private.
- Stop with `make public-down` when you are not demoing.
- The Mac must stay awake and online while demoing: run `caffeinate -dims` in a terminal.
- Only Platform is exposed (via the tunnel). Postgres and Runtime stay on
  `127.0.0.1`; never put `.env` in git.

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

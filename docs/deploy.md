# UNAI MVP demo deployment

This kit runs Platform, Runtime and Postgres on one Docker host, using LiveKit
Cloud for media. The two frontends deploy separately to Cloudflare Workers.
Run commands from the backend repository root unless stated otherwise. Do not
merge the old deployment branch. Images use the root workspace lock with
`uv sync --locked`; Platform runs `alembic upgrade head` before serving, including
`0015_handoffs`. Keep one Platform instance and one Runtime worker for this demo;
session ownership and throttling include process-local state.

1. Prepare a server or laptop with Docker Engine/Desktop and Docker Compose.
   For seeding, also install Python 3.11+ with `httpx` (the Platform environment
   already includes it). Keep the existing `agents/inha-admissions` seed pack
   and `scripts/seed_demo_agent.py` in the host checkout. They are not image assets.
   Back up any existing database before migrating; initial demo setup should use
   the new Compose-managed Postgres volume.

2. Copy `.env.example` to `.env` yourself and fill the blank required values.
   Generate independent random Postgres, admin and staff secrets, plus one shared
   Runtime token copied into both `PLATFORM_RUNTIME_TOKEN` and
   `RUNTIME_PLATFORM_TOKEN`. Set `PLATFORM_DATABASE_URL` to
   `postgresql+psycopg://voice_labs:<URL-encoded-password>@postgres:5432/voice_labs`
   (adjust user/database if changed). Set the LiveKit Cloud `wss://` URL and its
   API key/secret. Fill ElevenLabs key/voice ID, Gemini key and Google Translate
   key with access to the corresponding APIs. Confirm the chat model and the
   seed agent's configured model are available to your account. Chat can use the
   optional `PLATFORM_CHAT_LLM_API_KEY`/`PLATFORM_CHAT_LLM_BASE_URL` overrides.
   Leave `PLATFORM_DEMO_AGENTS` blank until seeded. No `LIVEKIT_PUBLIC_URL` is used:
   current Platform returns `LIVEKIT_URL` directly to browsers.

3. Start the backend:

   ```sh
   docker compose up -d
   docker compose ps
   docker compose logs --tail=100 platform runtime
   ```

   Compose builds missing images; use `docker compose up -d --build` after source
   changes. Platform waits for healthy Postgres and Runtime, then migrates before
   becoming healthy. Image healthchecks call each service's `/health`; these are
   liveness checks, not proof of provider credentials or working audio. Postgres
   persists in `postgres-data`. `make down` keeps that volume.

4. Seed through the running Platform API with a key from `PLATFORM_API_KEYS`:

   ```sh
   python scripts/seed_demo_agent.py --platform-url http://localhost:8000 --api-key '<admin-key>'
   ```

   Run this with the prepared Python environment; alternatively use
   `uv run --locked --package platform-voice-labs python scripts/seed_demo_agent.py ...`.
   `make seed API_KEY='<admin-key>'` is a shortcut. Paste the printed
   `PLATFORM_DEMO_AGENTS=inha-admissions=<UUID>` line into `.env`, replacing the
   blank line. Apply it with:

   ```sh
   docker compose up -d --force-recreate platform
   ```

   A plain `docker compose restart platform` does not reload changed environment
   values. Keep the admin key out of browser build variables.

5. Expose only Platform over HTTPS. For a quick demo, run cloudflared on the host:

   ```sh
   cloudflared tunnel --url http://localhost:8000
   ```

   Keep that process running and record the assigned HTTPS URL. Quick tunnel
   hostnames change when restarted; update frontend build variables and redeploy
   if they change. For a stable URL use a named tunnel or a TLS reverse proxy on
   the same host targeting `127.0.0.1:8000`. Runtime and Postgres have no published
   host ports. LiveKit Cloud already provides `wss://` media signaling. The template
   enables `PLATFORM_TRUST_PROXY=1` and Uvicorn forwarded headers; keep Platform's
   host port loopback-bound and ensure the chosen proxy supplies trusted client IP
   headers. A containerized proxy must share the Compose network and target
   `http://platform:8000` instead.

6. Build/deploy the frontends from their separate checkouts, following each repo's
   existing Cloudflare configuration and authentication:

   - `frontend.voice-labs` (Inha portal), Worker `unai-inha-demo`: configure
     `NEXT_PUBLIC_PLATFORM_URL=<Platform HTTPS URL>` and
     `NEXT_PUBLIC_PORTAL_AGENT_KEY=inha-admissions`, then run `npm run deploy`.
   - `frontend` (marketing), Worker `unai-site`: configure
     `NEXT_PUBLIC_PORTAL_URL=<deployed portal URL>`, then run `npm run deploy`.

   These public variables must be present at build time. Worker names, deploy
   scripts and frontend behavior are requested deployment targets and have not
   been checked against the separate frontend repositories here.

7. Set `PLATFORM_WEB_ORIGIN` to the actual portal and marketing origins, e.g.
   `https://unai-inha-demo.workers.dev,https://unai-site.workers.dev` (use actual
   account subdomains or custom domains). Current Platform splits on commas and
   trims whitespace. Origins include scheme and hostname, without paths or a
   trailing slash. The portal origin is required. Recreate Platform as in step 4
   after changing this setting. CORS does not replace API authentication.

8. Complete the smoke checklist:

   - `curl --fail https://<platform-host>/health` returns 200.
   - `curl --fail https://<platform-host>/demo/agents/inha-admissions` returns the
     seeded public agent. Check portal chat and knowledge-grounded answers.
   - Open `<portal-url>/staff` and enter the staff key. The backing API is
     `GET /staff/handoffs` with `X-Staff-Key`, not a backend `/staff` webpage.
     Verify an absent/incorrect key fails.
   - Start a voice call from the portal, allow microphone access, speak and hear
     an answer; verify transcript and hangup. Confirm provider errors in logs.
   - Request human handoff on device one; on device two open the staff page,
     accept/join, and verify bidirectional translated audio (different languages).
   - Repeat on mobile over HTTPS, including permission prompts, speaker output,
     reconnect/end behavior and session cleanup after the configured demo cap.

`make up`, `make down`, `make logs S=platform`, `make seed` and `make test`
provide shortcuts. `make test` runs each member's pytest suite from its directory;
install each member's test dependencies first. Never use volume deletion as a
routine restart.

For optional local LiveKit, set `COMPOSE_PROFILES=local-livekit`, provide local
LiveKit credentials, set `LIVEKIT_URL=ws://<Docker-host-LAN-IP>:7880`, and set
`rtc.node_ip` in `docker/livekit.yaml` to the same reachable IP. This avoids a
container-only hostname being returned to browsers. Local non-TLS signaling is
for local development; use Cloud for the HTTPS demo. The local LiveKit profile
has a restart policy but no image-bundled healthcheck; app healthchecks do not
validate its media connectivity.

Verification boundary: Compose validation and image-build results are reported
with the delivery. Live migrations against Postgres, seeding, provider access,
tunnel publication, both Worker deployments, browser audio, two-device handoff
and mobile behavior require operator verification with real credentials/devices.

Local delivery checks: both default and `local-livekit` Compose configurations
validated with a dummy env file; both image dependency selections passed
`uv sync --locked --dry-run`. Image builds were attempted but could not connect
to `/var/run/docker.sock` (permission denied), so neither image is build-verified.
The committable template is `.env.example` (the ignore rules allow it); the local
`.env.deploy.example`/`.env.public.example` copies are ignored and not needed.

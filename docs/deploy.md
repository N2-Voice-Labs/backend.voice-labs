# Deploying the backend on one VPS

Platform, Runtime and Postgres run with Docker Compose on a single Ubuntu VPS.
Media goes through LiveKit Cloud; HTTPS comes from a Cloudflare named tunnel (the
VPS opens no inbound ports besides SSH). The two frontends are separate Cloudflare
Workers. Run everything from the repository root.

Keep exactly one Platform and one Runtime container: session ownership and rate
limiting live in process memory.

## 1. Rent a VPS

Ubuntu 22.04 or 24.04, 4 vCPU / 8 GB RAM, 40 GB disk. Pick a region near the
users, e.g. Seoul for the Inha demo. Add your SSH key. No firewall changes are
needed; if you use `ufw`, allow only SSH, because the tunnel connects outbound.

Accounts you need before the next step: LiveKit Cloud (project `wss://` URL, API
key and secret), ElevenLabs (API key and voice ID), Gemini API key, Google Cloud
Translation API key, and a Cloudflare account with a domain on it.

## 2. Clone and deploy

```sh
git clone <repo-url> backend && cd backend
scripts/deploy.sh
```

The script is safe to re-run. It offers to install Docker (official convenience
script, asks first), creates `.env` from `.env.example`, generates the random
secrets (Postgres password, `PLATFORM_API_KEYS`, `PLATFORM_STAFF_KEY`, the shared
Platform/Runtime token) and builds `PLATFORM_DATABASE_URL`.

- The **admin API key is printed once**, at the run that generates it. Store it in
  a password manager; it is also in `.env` (mode 600). Never put it in a frontend
  build variable.
- It then stops and lists the required values still blank: `LIVEKIT_URL`,
  `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `ELEVENLABS_API_KEY`,
  `ELEVENLABS_VOICE_ID`, `GEMINI_API_KEY`, `GOOGLE_TRANSLATE_API_KEY` and
  `PLATFORM_WEB_ORIGIN`. Edit `.env` (`nano .env`) and run `scripts/deploy.sh`
  again. You can set `PLATFORM_WEB_ORIGIN` to a guess now and correct it in step 5.
- Then it runs `docker compose up -d --build`, waits for health and calls
  `http://127.0.0.1:8000/health`. The first build takes several minutes.
- It offers to seed the demo agent (`agents/inha-admissions`) and prints the
  `PLATFORM_DEMO_AGENTS=inha-admissions=<UUID>` line. Paste it into `.env`, then
  apply it (a plain `restart` does not reload `.env`):

  ```sh
  docker compose up -d --force-recreate platform
  ```

Platform runs `alembic upgrade head` on every start, under a Postgres advisory
lock, so restarts are safe. Healthchecks are liveness only; they do not prove that
provider keys work.

## 3. Cloudflare named tunnel

Create the tunnel once in the Cloudflare dashboard, then choose where cloudflared
runs.

1. Cloudflare dashboard > **Zero Trust** > **Networks** > **Tunnels** > **Create a
   tunnel** > **Cloudflared** > name it (e.g. `unai-backend`) > **Save tunnel**.
2. Copy the **token** (the long string after `--token` / `service install` in the
   shown command). You do not need to run the command it shows, except in option B.
3. Open the tunnel's **Public Hostname** tab (newer dashboards call it *Published
   application routes*) > **Add**: subdomain `api`, domain your domain, **Type
   HTTP**, **URL** as below. Save. Cloudflare creates the DNS record for you.

**Option A, compose profile (default, recommended).** The URL is
`http://platform:8000`. Put the token in `.env` and start:

```sh
# .env
TUNNEL_TOKEN=<token>
COMPOSE_PROFILES=tunnel      # scripts/deploy.sh sets this automatically when TUNNEL_TOKEN is set
```

```sh
docker compose up -d         # starts cloudflared after Platform is healthy
docker compose logs --tail=20 cloudflared   # look for "Registered tunnel connection"
```

**Option B, cloudflared on the host.** The URL is `http://localhost:8000`. Platform's
port is published on `127.0.0.1:8000` only. Install cloudflared from Cloudflare's
package repository (see developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads),
then `sudo cloudflared service install <token>`. Leave `TUNNEL_TOKEN` and
`COMPOSE_PROFILES` blank.

Check from anywhere: `curl https://api.example.com/health` returns 200.

**Client IPs.** Per-IP rate limits use the address uvicorn derives from `X-Forwarded-For`. `.env` pins the compose network to `COMPOSE_SUBNET=172.29.0.0/24` and sets
`FORWARDED_ALLOW_IPS` to the same subnet. Both topologies connect from inside it
(option A from the cloudflared container, option B through the published port as the
network gateway `172.29.0.1`), so Platform trusts `X-Forwarded-For` from exactly the
tunnel and nothing else. Do not set `127.0.0.1` (nothing connects from there). `*`
also works only because the host port is loopback-bound, but it trusts a
client-controlled header value, so prefer the subnet. Change both values together if
that subnet clashes with your network.

## 4. Frontends and CORS

Set the frontend's Platform URL to the tunnel hostname and redeploy each Worker
(the variables are baked in at build time):

- Portal (`frontend.voice-labs`): `NEXT_PUBLIC_PLATFORM_URL=https://api.example.com`,
  `NEXT_PUBLIC_PORTAL_AGENT_KEY=inha-admissions`, `npm run deploy`.
- Marketing site (`frontend`): `NEXT_PUBLIC_PORTAL_URL=<portal URL>`, `npm run deploy`.

Then set `PLATFORM_WEB_ORIGIN` in `.env` to the exact frontend origins,
comma-separated, scheme and host only (no path, no trailing slash), e.g.
`https://frontend-voice-labs.<acct>.workers.dev,https://voice-labs.<acct>.workers.dev`,
and apply it with `docker compose up -d --force-recreate platform`. Staff open
`<portal URL>/staff` and enter `PLATFORM_STAFF_KEY`.

## 5. Test from a phone

Switch the phone to mobile data (not the office Wi-Fi), open the portal over HTTPS,
allow the microphone, start a call, speak and listen for the answer, hang up. For
the handoff flow, use two devices in different languages and check both directions.
`docker compose logs -f platform runtime` while testing shows provider errors.

Venue note: the audience shares one public IP, so per-IP limits default high
(`PLATFORM_CALLS_PER_IP_HOUR=60`, `PLATFORM_CHAT_PER_IP_MINUTE=60`). Keep
`PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG` at or below `RUNTIME_MAX_SESSIONS` (4 fits
this VPS) and lower the per-IP limits again after the event.

## 6. Backups

```sh
scripts/backup_db.sh        # ./backups/voice_labs-<UTC time>.sql.gz, keeps the newest 14
```

Cron (daily 03:15 server time, keep 30; adjust the path, `crontab -e`):

```cron
15 3 * * * cd /home/ubuntu/backend && KEEP=30 scripts/backup_db.sh >> backups/backup.log 2>&1
```

Copy `backups/` off the VPS (e.g. `rclone` or `scp`) because the volume and the dumps
live on the same disk. Restore (stop Platform first, start it again afterwards):

```sh
docker compose stop platform
gunzip -c backups/<file>.sql.gz | docker compose exec -T postgres psql -U voice_labs -d voice_labs
docker compose start platform
```

Never delete the `postgres-data` volume as a routine restart.

## 7. Updating

```sh
git pull && docker compose up -d --build
```

Migrations run on start. Take a backup first when a release adds migrations.
`docker compose logs --tail=50 platform` confirms the new version started.
Runtime waits up to 45 s for live calls to finish when it stops, so an update
briefly interrupts calls in progress.

## Troubleshooting

- **Logs and state:** `docker compose ps`; `docker compose logs -f --tail=100 platform runtime cloudflared`
  (`make logs S=platform`). Raise detail with `PLATFORM_LOG_LEVEL=DEBUG`, then recreate.
- **Health:** `curl -s http://127.0.0.1:8000/health` on the VPS; if that works but
  the public hostname does not, the problem is the tunnel (hostname route, token,
  `docker compose logs cloudflared`). Platform has `restart: unless-stopped`, and a
  crash loop on start usually shows a missing/invalid `.env` value or failed migration.
- **Browser CORS error:** the page's origin is not in `PLATFORM_WEB_ORIGIN`. Compare
  it to the address bar exactly (scheme, host, no trailing slash), edit `.env`,
  and `docker compose up -d --force-recreate platform`. A 401 shown as a CORS
  error means the API key/staff key is wrong or missing.
- **LiveKit connection fails in the browser:** `LIVEKIT_URL` must be the `wss://`
  project URL from LiveKit Cloud (not `https://`, not `ws://` on an HTTPS page). If
  browsers need a different URL than the server uses, set `LIVEKIT_PUBLIC_URL`
  (it defaults to `LIVEKIT_URL`). Check the key/secret belong to the same project.
- **Provider quota or key errors (429, 401, "provider unavailable"):** check the
  Runtime logs for the failing provider; confirm the key and quota in the
  ElevenLabs / Gemini / Google Cloud / LiveKit consoles. Providers marked down
  are reported by Platform; set `PLATFORM_ALERT_WEBHOOK_URL` for notifications.
- **Platform returns 401 everywhere:** `PLATFORM_API_KEYS` is blank or the client
  sends another key. `PLATFORM_AUTH_DISABLED=1` bypasses auth: never use it in prod.
- **Chat with Claude:** set `PLATFORM_CHAT_LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`
  and `PLATFORM_CHAT_LLM_MODEL=claude-haiku-5-5`; the image already contains the SDK.
- **Disk filling up:** `docker system df`, `docker image prune -f`; prune old backups.

Local LiveKit (development only): `COMPOSE_PROFILES=local-livekit`,
`LIVEKIT_URL=ws://<LAN-IP>:7880` and `rtc.node_ip` in `docker/livekit.yaml` set to the
same IP. Use LiveKit Cloud for the HTTPS/mobile demo.

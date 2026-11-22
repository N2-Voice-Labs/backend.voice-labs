#!/usr/bin/env bash
# One-shot, idempotent deploy of the backend on a single Ubuntu 22.04/24.04 VPS.
#
#   scripts/deploy.sh [--yes] [--seed | --no-seed]
#
# 1. checks Docker (offers to install it via the official convenience script),
# 2. creates .env from .env.example and generates the random secrets (only blank ones),
# 3. stops with the list of required values you still have to fill in,
# 4. docker compose up -d --build, waits for /health,
# 5. optionally seeds the demo agent and prints the PLATFORM_DEMO_AGENTS line to paste.
#
# Re-running is safe: existing .env values are never overwritten, and secrets are never
# printed, except the admin API key once, in the run that generates it.
# Optional: PUBLIC_URL=https://api.example.com also checks the tunnel end to end.
# See docs/deploy.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ASSUME_YES=0
SEED=ask
HEALTH_URL="http://127.0.0.1:8000/health"
WAIT_SECS="${WAIT_SECS:-240}"

info() { printf '==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    -y | --yes) ASSUME_YES=1 ;;
    --seed) SEED=yes ;;
    --no-seed) SEED=no ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# confirm "question": returns 0 on yes. Without a terminal it only says yes under --yes.
confirm() {
  if [ "$ASSUME_YES" = 1 ]; then return 0; fi
  if [ ! -t 0 ]; then return 1; fi
  local reply
  read -r -p "$1 [y/N] " reply
  case "$reply" in [yY] | [yY][eE][sS]) return 0 ;; *) return 1 ;; esac
}

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

# ---- 0. host sanity -----------------------------------------------------------------
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:22.04 | ubuntu:24.04) ;;
    *) warn "tested on Ubuntu 22.04/24.04; this is ${PRETTY_NAME:-unknown}. Continuing." ;;
  esac
fi
command -v curl >/dev/null || die "curl is required: sudo apt-get install -y curl ca-certificates"
command -v openssl >/dev/null || die "openssl is required: sudo apt-get install -y openssl"
[ -f compose.yml ] && [ -f .env.example ] || die "run this from a checkout of the backend repository"

# ---- 1. Docker ----------------------------------------------------------------------
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  info "Docker Engine with the Compose plugin was not found."
  confirm "Install it now with the official script from https://get.docker.com?" \
    || die "Docker is required. Install it (https://docs.docker.com/engine/install/ubuntu/) and re-run."
  tmp_installer="$(mktemp)"
  trap 'rm -f "$tmp_installer"' EXIT
  curl -fsSL https://get.docker.com -o "$tmp_installer"
  $SUDO sh "$tmp_installer"
  $SUDO systemctl enable --now docker >/dev/null 2>&1 || true
  if [ -n "$SUDO" ]; then
    $SUDO usermod -aG docker "$USER" || true
    info "Added $USER to the docker group; log out and in again to use docker without sudo."
  fi
fi

# Use sudo for docker when the current user is not in the docker group (yet).
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if [ -n "$SUDO" ] && $SUDO docker info >/dev/null 2>&1; then
    DOCKER=("$SUDO" docker)
  else
    die "cannot talk to the Docker daemon (is it running? are you in the docker group?)"
  fi
fi
dc() { "${DOCKER[@]}" compose "$@"; }

# ---- 2. .env ------------------------------------------------------------------------
# get_env KEY: value of the last KEY= line, without surrounding quotes/CR.
get_env() {
  local v
  v="$(sed -n "s/^$1=//p" .env | tail -n1 | tr -d '\r')"
  v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
  printf '%s' "$v"
}

# set_env KEY VALUE: replace the KEY= line (or append it). The value goes through the
# environment so no quoting or sed escaping is needed.
set_env() {
  local tmp
  tmp="$(mktemp .env.XXXXXX)"
  KEY="$1" VAL="$2" awk '
    BEGIN { k = ENVIRON["KEY"]; v = ENVIRON["VAL"]; done = 0 }
    index($0, k "=") == 1 { if (!done) { print k "=" v; done = 1 } ; next }
    { print }
    END { if (!done) print k "=" v }
  ' .env >"$tmp"
  chmod 600 "$tmp"
  mv "$tmp" .env
}

gen() { openssl rand -hex 32; }

if [ ! -f .env ]; then
  info "Creating .env from .env.example"
  (umask 077 && cp .env.example .env)
fi
chmod 600 .env

# Generate only what is blank, so re-runs keep your values (changing POSTGRES_PASSWORD
# after the database volume exists would lock the app out of Postgres).
admin_key_generated=""
if [ -z "$(get_env POSTGRES_PASSWORD)" ]; then
  set_env POSTGRES_PASSWORD "$(gen)"
  info "Generated POSTGRES_PASSWORD"
fi
if [ -z "$(get_env PLATFORM_DATABASE_URL)" ]; then
  db_user="$(get_env POSTGRES_USER)"; db_user="${db_user:-voice_labs}"
  db_name="$(get_env POSTGRES_DB)"; db_name="${db_name:-voice_labs}"
  # The generated password is hex, so it needs no URL-encoding. A hand-set password with
  # special characters must be URL-encoded by you.
  set_env PLATFORM_DATABASE_URL "postgresql+psycopg://${db_user}:$(get_env POSTGRES_PASSWORD)@postgres:5432/${db_name}"
  info "Built PLATFORM_DATABASE_URL"
fi
if [ -z "$(get_env PLATFORM_API_KEYS)" ]; then
  admin_key_generated="$(gen)"
  set_env PLATFORM_API_KEYS "$admin_key_generated"
  info "Generated PLATFORM_API_KEYS"
fi
if [ -z "$(get_env PLATFORM_STAFF_KEY)" ]; then
  set_env PLATFORM_STAFF_KEY "$(gen)"
  info "Generated PLATFORM_STAFF_KEY"
fi
# One shared token in both variables: reuse whichever is already set.
shared="$(get_env PLATFORM_RUNTIME_TOKEN)"
[ -n "$shared" ] || shared="$(get_env RUNTIME_PLATFORM_TOKEN)"
if [ -z "$shared" ]; then
  shared="$(gen)"
  info "Generated the shared Platform/Runtime token"
fi
[ "$(get_env PLATFORM_RUNTIME_TOKEN)" = "$shared" ] || set_env PLATFORM_RUNTIME_TOKEN "$shared"
[ "$(get_env RUNTIME_PLATFORM_TOKEN)" = "$shared" ] || set_env RUNTIME_PLATFORM_TOKEN "$shared"
unset shared

if [ -n "$admin_key_generated" ]; then
  cat <<EOF

  ADMIN API KEY (shown only this once; it is also stored in .env, mode 600):

      $admin_key_generated

  Store it in your password manager now. It is the Authorization: Bearer key for
  the Platform API and for seeding. Never put it in a frontend build variable.

EOF
fi

# ---- 3. required values ---------------------------------------------------------------
missing=()
for key in LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET ELEVENLABS_API_KEY \
  ELEVENLABS_VOICE_ID GEMINI_API_KEY GOOGLE_TRANSLATE_API_KEY PLATFORM_WEB_ORIGIN; do
  [ -n "$(get_env "$key")" ] || missing+=("$key")
done
problems=()
lk="$(get_env LIVEKIT_URL)"
if [ -n "$lk" ] && [[ "$lk" != wss://* && "$lk" != ws://* ]]; then
  problems+=("LIVEKIT_URL must start with wss:// (the LiveKit Cloud websocket URL)")
fi
origins="$(get_env PLATFORM_WEB_ORIGIN)"
if [[ "$origins" == *"<"* || "$origins" == */ || "$origins" == */,* ]]; then
  problems+=("PLATFORM_WEB_ORIGIN must hold real origins without paths or trailing slashes")
fi
if [ "$(get_env PLATFORM_AUTH_DISABLED)" = "1" ]; then
  problems+=("PLATFORM_AUTH_DISABLED=1 turns API authentication off; remove it for a public deployment")
fi
if [ ${#missing[@]} -gt 0 ] || [ ${#problems[@]} -gt 0 ]; then
  echo
  if [ ${#missing[@]} -gt 0 ]; then
    echo "These required values in $ROOT/.env are still blank:"
    printf '  - %s\n' "${missing[@]}"
  fi
  if [ ${#problems[@]} -gt 0 ]; then
    echo "These values need fixing in $ROOT/.env:"
    printf '  - %s\n' "${problems[@]}"
  fi
  echo
  echo "Edit .env (docs/deploy.md explains where each value comes from), then re-run scripts/deploy.sh."
  exit 1
fi

# With a tunnel token, run cloudflared from compose (profile "tunnel").
if [ -n "$(get_env TUNNEL_TOKEN)" ]; then
  profiles="$(get_env COMPOSE_PROFILES)"
  case ",$profiles," in
    *,tunnel,*) ;;
    *) set_env COMPOSE_PROFILES "${profiles:+$profiles,}tunnel"; info "TUNNEL_TOKEN is set: enabled the 'tunnel' compose profile" ;;
  esac
fi

# ---- 4. start -------------------------------------------------------------------------
info "Validating compose configuration"
dc config -q
info "Building and starting (first build takes several minutes)"
dc up -d --build

info "Waiting for the backend to become healthy (up to ${WAIT_SECS}s)"
deadline=$((SECONDS + WAIT_SECS))
healthy=0
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -fsS -m 4 -o /dev/null "$HEALTH_URL" 2>/dev/null; then healthy=1; break; fi
  sleep 3
done
if [ "$healthy" != 1 ]; then
  dc ps || true
  dc logs --tail=40 platform runtime || true
  if [ "$(dc ps platform --format '{{.Health}}' 2>/dev/null || true)" = "healthy" ]; then
    die "the platform container is healthy but $HEALTH_URL is unreachable from this host (port 8000 in use or blocked?)"
  fi
  die "platform did not become healthy; see the logs above (docker compose logs -f)"
fi
info "GET $HEALTH_URL -> $(curl -fsS -m 4 "$HEALTH_URL")"
dc ps

if [ -n "${PUBLIC_URL:-}" ]; then
  if curl -fsS -m 10 -o /dev/null "${PUBLIC_URL%/}/health"; then
    info "Public URL ${PUBLIC_URL%/}/health is reachable through the tunnel"
  else
    warn "Public URL ${PUBLIC_URL%/}/health is NOT reachable; check the tunnel (docs/deploy.md, troubleshooting)"
  fi
fi

# ---- 5. optional demo agent -------------------------------------------------------------
seed_demo() {
  local key out
  key="$(get_env PLATFORM_API_KEYS)"; key="${key%%,*}"
  if command -v python3 >/dev/null && python3 -c 'import httpx' >/dev/null 2>&1; then
    out="$(python3 scripts/seed_demo_agent.py --platform-url http://127.0.0.1:8000 --api-key "$key")" || return 1
  else
    info "No host Python with httpx; seeding from a throwaway python:3.11-slim container"
    # shellcheck disable=SC2016  # $SEED_KEY must expand inside the container, not here
    out="$(SEED_KEY="$key" "${DOCKER[@]}" run --rm --network host -e SEED_KEY -v "$ROOT:/work:ro" -w /work python:3.11-slim \
      sh -c 'pip install -q --disable-pip-version-check httpx >/dev/null 2>&1 && python scripts/seed_demo_agent.py --platform-url http://127.0.0.1:8000 --api-key "$SEED_KEY"')" || return 1
  fi
  printf '%s\n' "$out" | grep '^Agent ID' || true
  demo_line="$(printf '%s\n' "$out" | grep '^PLATFORM_DEMO_AGENTS=' | tail -n1)"
}

if [ -z "$(get_env PLATFORM_DEMO_AGENTS)" ] && [ -d agents/inha-admissions ]; then
  do_seed=0
  case "$SEED" in
    yes) do_seed=1 ;;
    ask) confirm "Seed the demo agent (agents/inha-admissions) now?" && do_seed=1 ;;
  esac
  if [ "$do_seed" = 1 ]; then
    demo_line=""
    if seed_demo && [ -n "$demo_line" ]; then
      cat <<EOF

Paste this line into .env (replacing the blank PLATFORM_DEMO_AGENTS=), then apply it:

    $demo_line

    docker compose up -d --force-recreate platform

EOF
    else
      warn "seeding failed; re-run later: make seed API_KEY='<admin key>' (see docs/deploy.md)"
    fi
  fi
fi

cat <<'EOF'

Done. Next (docs/deploy.md):
  - Cloudflare tunnel: public hostname -> http://platform:8000 (or http://localhost:8000 for cloudflared on the host)
  - set the frontend API URL to that hostname and make sure PLATFORM_WEB_ORIGIN lists the frontends
  - schedule scripts/backup_db.sh
EOF

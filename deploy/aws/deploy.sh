#!/usr/bin/env bash
# Pull main and rebuild one UNAI app on the AWS server (EC2, /opt/unai).
#
#   deploy.sh backend   backend.voice-labs main -> Platform, Runtime (Postgres is left running)
#   deploy.sh site      frontend main           -> marketing site + /inha portal + workspace
#   deploy.sh all       both
#
# GitHub Actions runs this through the SSM document UNAI-Deploy after a push to main
# (see README.md here). It is also safe to run by hand on the server as root.
# A failed build leaves the running containers untouched.
set -euo pipefail

# The backend pull rewrites this file; run from a copy so bash never reads a changed script.
if [ -z "${UNAI_DEPLOY_COPY:-}" ]; then
  copy="$(mktemp /tmp/unai-deploy.XXXXXX)"
  cp "$0" "$copy"
  UNAI_DEPLOY_COPY="$copy" exec bash "$copy" "$@"
fi
trap 'rm -f "$UNAI_DEPLOY_COPY"' EXIT

export HOME="${HOME:-/root}"
ROOT=/opt/unai
HERE="$ROOT/backend/deploy/aws"
FRONTENDS=(docker compose -p unai-frontends -f "$HERE/compose.frontends.yml")

log() { printf '==> %s\n' "$*"; }

# One deploy at a time (backend and site pushes can arrive together).
exec 9>/var/lock/unai-deploy.lock
flock 9

pull() {
  git -C "$1" fetch -q origin main
  git -C "$1" reset -q --hard origin/main
  log "$(basename "$1") now at $(git -C "$1" log --oneline -1)"
}

wait_ok() {
  local url=$1
  for _ in $(seq 1 80); do
    if curl -fsS -m 4 -o /dev/null "$url"; then log "OK $url"; return 0; fi
    sleep 3
  done
  printf 'ERROR: %s did not answer in time\n' "$url" >&2
  return 1
}

deploy_backend() {
  pull "$ROOT/backend"
  cd "$ROOT/backend"
  docker compose up -d --build
  wait_ok http://127.0.0.1:8000/health
  docker compose ps --format '{{.Name}} {{.Status}}'
}

deploy_site() {
  pull "$ROOT/frontend"
  "${FRONTENDS[@]}" up -d --build site
  wait_ok http://127.0.0.1:3001/
}

case "${1:-}" in
  backend) deploy_backend ;;
  site) deploy_site ;;
  all) deploy_backend; deploy_site ;;
  *) echo "usage: $0 backend|site|all" >&2; exit 2 ;;
esac

# Old images pile up with every build; the disk is 40 GB.
docker image prune -f >/dev/null
log "deploy of $1 finished"

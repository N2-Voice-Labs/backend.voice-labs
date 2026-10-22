#!/bin/zsh
# Start / stop a private Platform (:8300) and Runtime (:8301) for the live latency report.
#
#   scripts/live_servers.sh start     # needs LiveKit up: (cd ../../stack.voice-labs && docker compose up -d livekit)
#   scripts/live_servers.sh stop
#
# Uses a throw-away sqlite database and the keys in stack.voice-labs/.env (never printed).
# Ports 8000/8001 are left alone. Override with PLATFORM_PORT / RUNTIME_PORT / ENV_FILE / PY.
HERE=${0:A:h}
BACKEND=${HERE:h:h}
ENV_FILE=${ENV_FILE:-$BACKEND/../stack.voice-labs/.env}
PY=${PY:-$HOME/.venvs/voice-labs/bin/python}
PP=${PLATFORM_PORT:-8300}
RP=${RUNTIME_PORT:-8301}
D=${LIVE_DIR:-$HOME/.cache/voice-labs-evals/live-servers}
mkdir -p "${D:?}"
case $1 in
start)
  [ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE"; exit 1; }
  for port in $PP $RP; do
    lsof -iTCP:$port -sTCP:LISTEN -nP >/dev/null 2>&1 && { echo "port $port is already in use"; exit 1; }
  done
  set -a; source "$ENV_FILE"; set +a
  export PYTHONDONTWRITEBYTECODE=1
  rm -f "${D:?}/platform.db"
  export PLATFORM_DATABASE_URL=sqlite:///$D/platform.db PLATFORM_AUTO_CREATE_SCHEMA=1
  export RUNTIME_API_URL=http://localhost:$RP RUNTIME_PLATFORM_URL=http://localhost:$PP PLATFORM_API_URL=http://localhost:$PP
  unset PLATFORM_API_KEYS
  (cd "$BACKEND/platform" && nohup $PY -m uvicorn src.platform.main:app --port $PP --log-level warning > "$D/platform.log" 2>&1 & echo $! > "$D/platform.pid")
  (cd "$BACKEND/runtime" && nohup $PY -m uvicorn src.runtime.api:app --port $RP --log-level info > "$D/runtime.log" 2>&1 & echo $! > "$D/runtime.pid")
  for i in {1..40}; do
    curl -sf localhost:$PP/health >/dev/null && curl -sf localhost:$RP/health >/dev/null && { echo "up: platform :$PP runtime :$RP (logs in $D)"; exit 0; }
    sleep 1
  done
  echo "failed to start; see $D/platform.log and $D/runtime.log"; exit 1;;
stop)
  for name in platform runtime; do
    [ -f "$D/$name.pid" ] && kill "$(cat "$D/$name.pid")" 2>/dev/null
    rm -f "${D:?}/${name:?}.pid"
  done
  echo stopped;;
*) echo "usage: $0 start|stop"; exit 2;;
esac

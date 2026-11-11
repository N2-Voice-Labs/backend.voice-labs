#!/bin/bash
# Local stack for a real two-person call, with every log kept (they used to be lost with a temp directory).
#
#   scripts/live_call_stack.sh start    # Platform, Runtime (timeline diagnostics on), and the frontend if FRONTEND_DIR is set
#   scripts/live_call_stack.sh stop
#   scripts/live_call_stack.sh summary  # per-turn timelines of the latest run (speaker, listener, destination, marks)
#
# Needs the python environments in PLATFORM_PY / RUNTIME_PY (default: the repo's .venv under each member), the
# provider keys and LiveKit settings in backend.voice-labs/.env (never printed), and for the browser side
# FRONTEND_DIR (a built frontend; it is served on :3000). Logs: runtime/bench/live/<timestamp>/{platform,runtime,web}.log
# (git-ignored; they contain what was said when diagnostics are on: keep them private).
B="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$B/runtime/bench/live"
PIDS="$RUN/.pids"
case "$1" in
  start)
    D="$RUN/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$D"; echo "$D" > "$RUN/latest"; : > "$PIDS"
    set -a; . "$B/.env"; set +a
    export PLATFORM_RUNTIME_TOKEN="${PLATFORM_RUNTIME_TOKEN:-local-runtime-token}" RUNTIME_PLATFORM_TOKEN="${PLATFORM_RUNTIME_TOKEN:-local-runtime-token}"
    export PLATFORM_API_KEYS="${PLATFORM_API_KEYS:-$(openssl rand -hex 16)}"
    export PLATFORM_DATABASE_URL="sqlite:///$D/platform.db" PLATFORM_AUTO_CREATE_SCHEMA=1 PLATFORM_PUBLIC_CALLS=1
    export PLATFORM_CALLS_PROVIDERS="${PLATFORM_CALLS_PROVIDERS:-stt=elevenlabs,llm=google-translate,tts=elevenlabs}"
    export PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG="${PLATFORM_MAX_ACTIVE_SESSIONS_PER_ORG:-1}"
    export RUNTIME_API_URL=http://127.0.0.1:18001 RUNTIME_PLATFORM_URL=http://127.0.0.1:18000
    (cd "$B/runtime" && RUNTIME_TIMELINE_DIAG=1 PYTHONPATH=src setsid "${RUNTIME_PY:-$B/runtime/.venv/bin/python}" -m uvicorn runtime.api:app --host 127.0.0.1 --port 18001 > "$D/runtime.log" 2>&1 & echo $! >> "$PIDS")
    (cd "$B/platform" && setsid "${PLATFORM_PY:-$B/platform/.venv/bin/python}" -m uvicorn src.platform.main:app --host 127.0.0.1 --port 18000 > "$D/platform.log" 2>&1 & echo $! >> "$PIDS")
    if [ -n "$FRONTEND_DIR" ]; then
      (cd "$FRONTEND_DIR" && PLATFORM_PROXY_TARGET=http://127.0.0.1:18000 setsid npx next start --hostname 127.0.0.1 --port "${WEB_PORT:-3000}" > "$D/web.log" 2>&1 & echo $! >> "$PIDS")
    fi
    echo "logs: $D"; echo "admin key (for reading status): $PLATFORM_API_KEYS" | sed 's/: .*/: (set in this shell as PLATFORM_API_KEYS)/' ;;
  stop)
    [ -f "$PIDS" ] && while read -r p; do kill "$p" 2>/dev/null; done < "$PIDS"; : > "$PIDS"; echo stopped ;;
  summary)
    D="$(cat "$RUN/latest")"
    sed 's/\x1b\[[0-9;]*m//g' "$D/runtime.log" | grep "utterance_timeline" | python3 -c '
import sys, re, json
for line in sys.stdin:
    m = re.search(r"turn=(\S+) engine=(\S+) speaker=(\S+) listener=(\S+) destination=(\S+) t0_monotonic_s=(\S+) marks_ms=(\{.*\})", line)
    if not m:
        print("(old format)", line[:160].strip()); continue
    turn, engine, speaker, listener, dest, t0, marks = m.groups(); d = json.loads(marks); se = d.get("speech_end")
    rel = {k: round(v - se) for k, v in d.items() if se is not None and k in ("final_transcript","turn_end","translation_first_token","tts_first_frame","livekit_submit","livekit_submitted","first_audio","stable")}
    print(f"turn {turn} {speaker} -> {listener} [{dest}]  (ms after speech end)", rel)
' ;;
  *) echo "usage: $0 start|stop|summary" ;;
esac

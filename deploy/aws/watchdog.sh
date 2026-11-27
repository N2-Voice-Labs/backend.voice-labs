#!/usr/bin/env bash
# Restart Platform when Docker marks it unhealthy (it hung on 2026-11-27 with /health
# timing out for hours). Runs every minute from /etc/cron.d/unai-watchdog.
# Logs before each restart are kept in /opt/unai/logs for debugging.
set -uo pipefail

state="$(docker inspect -f '{{.State.Health.Status}}' unai-mvp-platform-1 2>/dev/null || true)"
[ "$state" = unhealthy ] || exit 0

mkdir -p /opt/unai/logs
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
docker logs --since 2h unai-mvp-platform-1 >"/opt/unai/logs/platform-unhealthy-$stamp.log" 2>&1
docker restart unai-mvp-platform-1 >/dev/null
echo "$stamp restarted unai-mvp-platform-1 (unhealthy)" >>/opt/unai/logs/watchdog.log
# Keep the newest 20 hang logs.
ls -1t /opt/unai/logs/platform-unhealthy-*.log 2>/dev/null | tail -n +21 | xargs -r rm -f

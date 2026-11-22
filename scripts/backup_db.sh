#!/usr/bin/env bash
# Dump the Compose Postgres into ./backups/<db>-<UTC timestamp>.sql.gz and keep the newest N.
#   scripts/backup_db.sh            # keep 14 (default)
#   KEEP=30 BACKUP_DIR=/mnt/b scripts/backup_db.sh
# Safe for cron; see docs/deploy.md. Restore:
#   gunzip -c backups/<file>.sql.gz | docker compose exec -T postgres psql -U voice_labs -d voice_labs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KEEP="${KEEP:-14}"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"
case "$KEEP" in '' | *[!0-9]*) echo "KEEP must be a positive integer" >&2; exit 2 ;; esac
[ "$KEEP" -ge 1 ] || { echo "KEEP must be >= 1" >&2; exit 2; }

# Database/user come from .env when present (same defaults as compose.yml).
if [ -f .env ]; then
  POSTGRES_DB="${POSTGRES_DB:-$(sed -n 's/^POSTGRES_DB=//p' .env | tail -n1)}"
  POSTGRES_USER="${POSTGRES_USER:-$(sed -n 's/^POSTGRES_USER=//p' .env | tail -n1)}"
fi
DB="${POSTGRES_DB:-voice_labs}"
DB_USER="${POSTGRES_USER:-voice_labs}"

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
# Dumps hold user data: keep them out of git even if the directory is inside the checkout.
[ -e "$BACKUP_DIR/.gitignore" ] || printf '*\n' >"$BACKUP_DIR/.gitignore"
umask 077

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$BACKUP_DIR/${DB}-${stamp}.sql.gz"
tmp="$out.partial"
trap 'rm -f "$tmp"' EXIT

# The database user/name are expanded inside the container; pg_dump runs as that user.
docker compose exec -T postgres pg_dump -U "$DB_USER" -d "$DB" --no-owner --clean --if-exists | gzip -9 >"$tmp"
# pipefail covers pg_dump failure; also refuse an empty/trivial dump.
[ "$(wc -c <"$tmp")" -gt 200 ] || { echo "backup looks empty, aborting" >&2; exit 1; }
mv "$tmp" "$out"
echo "backup written: $out ($(du -h "$out" | cut -f1))"

# Keep the newest $KEEP backups of this database.
# shellcheck disable=SC2012
ls -1t "$BACKUP_DIR/${DB}-"*.sql.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | while IFS= read -r old; do
  rm -f -- "$old"
  echo "pruned: $old"
done

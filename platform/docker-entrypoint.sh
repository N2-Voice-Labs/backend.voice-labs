#!/bin/sh
# Commands:
#   serve   (default) start uvicorn; runs migrations first if PLATFORM_MIGRATE_ON_START=1
#   migrate run `alembic upgrade head` and exit
#   <other> exec as-is (e.g. `python -m alembic current`)
set -eu

# `alembic upgrade head` under a Postgres advisory lock, so two containers starting at the
# same time (e.g. a rolling restart) never migrate concurrently: the second one waits, then
# finds nothing to do. The lock is session-scoped and is released when the connection
# closes, including if this process dies. SQLite (or no URL) skips the lock.
migrate() {
  python - <<'PY'
import os
import subprocess
import sys

url = os.environ.get("PLATFORM_DATABASE_URL", "")
cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
if not url.startswith("postgres"):
    sys.exit(subprocess.call(cmd))

from sqlalchemy import create_engine, text

LOCK_ID = 7243001  # arbitrary constant, shared by every Platform instance
engine = create_engine(url, isolation_level="AUTOCOMMIT")
with engine.connect() as conn:
    print("migrate: waiting for advisory lock", flush=True)
    conn.execute(text("SELECT pg_advisory_lock(:id)"), {"id": LOCK_ID})
    try:
        code = subprocess.call(cmd)
    finally:
        conn.execute(text("SELECT pg_advisory_unlock(:id)"), {"id": LOCK_ID})
sys.exit(code)
PY
}

case "${1:-serve}" in
  migrate)
    migrate
    exit $?
    ;;
  serve)
    if [ "${PLATFORM_MIGRATE_ON_START:-1}" = "1" ]; then
      migrate
    fi
    exec uvicorn src.platform.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --proxy-headers --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-127.0.0.1}"
    ;;
  *)
    exec "$@"
    ;;
esac

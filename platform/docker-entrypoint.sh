#!/bin/sh
# Commands:
#   serve   (default) start uvicorn; runs migrations first if PLATFORM_MIGRATE_ON_START=1
#   migrate run `alembic upgrade head` and exit
#   <other> exec as-is (e.g. `python -m alembic current`)
set -eu

case "${1:-serve}" in
  migrate)
    exec python -m alembic upgrade head
    ;;
  serve)
    if [ "${PLATFORM_MIGRATE_ON_START:-1}" = "1" ]; then
      python -m alembic upgrade head
    fi
    exec uvicorn src.platform.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
      --proxy-headers --forwarded-allow-ips="${FORWARDED_ALLOW_IPS:-127.0.0.1}"
    ;;
  *)
    exec "$@"
    ;;
esac

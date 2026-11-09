# syntax=docker/dockerfile:1.7
# Voice Labs Platform (control plane) image.
# Build from the repository ROOT so the root uv.lock is used:
#   docker build -f platform/Dockerfile -t voice-labs-platform .

# ---- builder: resolve the locked environment for the platform member only ----
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.11-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /workspace

# Workspace manifests only (all members must be present for uv to load the workspace).
COPY pyproject.toml uv.lock ./
COPY platform/pyproject.toml platform/pyproject.toml
COPY runtime/pyproject.toml runtime/pyproject.toml
COPY evals/pyproject.toml evals/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package platform-voice-labs

# ---- final: venv + platform source only ----
FROM python:3.11-slim-bookworm AS final

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8300

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY platform/src ./src
COPY platform/migrations ./migrations
COPY platform/alembic.ini ./alembic.ini
COPY --chmod=0755 platform/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

USER app
EXPOSE 8300

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8300'), timeout=4).status == 200 else 1)"]

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["serve"]

# syntax=docker/dockerfile:1.7
#
# Runtime service: FastAPI control API (uvicorn) that spawns Pipecat/LiveKit voice sessions
# in-process. Build context is the REPO ROOT:
#
#   docker build -f runtime/Dockerfile -t voice-labs-runtime .
#
# Runtime keeps session ownership in process memory: run exactly ONE replica / ONE worker.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.13.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------------------
# builder: resolve the workspace lockfile into /opt/venv (runtime package + voice extra,
# no dev/test dependencies).
# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /workspace

# Only manifests: the workspace needs every member's pyproject.toml to validate the
# frozen lockfile, but no member source. This layer is cached until a manifest changes.
COPY pyproject.toml uv.lock ./
COPY platform/pyproject.toml platform/pyproject.toml
COPY runtime/pyproject.toml runtime/pyproject.toml
COPY evals/pyproject.toml evals/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-workspace --package runtime-voice-labs --extra voice

# ---------------------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS final

# libgomp1: OpenMP runtime required by onnxruntime (Silero VAD + Smart Turn v3).
# tini: PID 1 that forwards SIGTERM and reaps children.
# ca-certificates: already in the base image; listed so TLS to providers never regresses.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    PYTHONPATH=/app/runtime/src \
    PORT=8001

COPY --from=builder /opt/venv /opt/venv

# Import runtime.api through PYTHONPATH=src from the runtime directory.
WORKDIR /app/runtime
COPY runtime/src ./src
# Includes src/runtime/assets/apology/*.wav (loaded relative to the package at runtime).
# Drop stray local build metadata that may ride along in the context.
RUN rm -rf src/*.egg-info \
    && find src -name __pycache__ -type d -prune -exec rm -rf {} + \
    && python -m compileall -q src \
    && test "$(ls src/runtime/assets/apology/*.wav | wc -l)" -ge 1

# Build-time smoke test as the runtime user. Pipecat ships the Silero VAD and Smart Turn v3
# ONNX weights inside its wheel, so nothing is downloaded at call time; this proves the
# native libs (onnxruntime/libgomp, livekit, soundfile) load and the models instantiate.
USER app
RUN python - <<'PY'
import livekit.rtc, onnxruntime, pipecat  # noqa: F401
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
SileroVADAnalyzer()
LocalSmartTurnAnalyzerV3()
import runtime.api  # noqa: F401
print("runtime image smoke test ok")
PY

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8001'), timeout=4).status == 200 else 1)"]

ENTRYPOINT ["tini", "--"]
# Single uvicorn worker on purpose (process-local session state). PORT is overridable.
CMD ["sh", "-c", "exec uvicorn runtime.api:app --host 0.0.0.0 --port ${PORT:-8001} --timeout-graceful-shutdown 30"]

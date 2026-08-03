# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: install the package into an isolated virtualenv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY agent_reflex ./agent_reflex

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Runtime: non-root, no build tooling, python-only healthcheck (no curl).
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN useradd --uid 10001 --create-home appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY deploy/logging-config.json /app/logging-config.json

USER appuser

EXPOSE 8000

# Liveness: /health. Orchestrators should use GET /ready for readiness.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"]

CMD ["uvicorn", "agent_reflex.dashboard.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "/app/logging-config.json"]

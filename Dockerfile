# ─────────────────────────────────────────────────────────
#  Stage 1 — builder
# ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv venv .venv --python 3.12 && \
    uv sync --frozen --no-dev

# ─────────────────────────────────────────────────────────
#  Stage 2 — runtime
# ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv .venv

COPY src/       ./src/
COPY models/    ./models/
COPY frontend/  ./frontend/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
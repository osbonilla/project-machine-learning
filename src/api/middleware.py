"""
middleware.py — CORS y logging de requests
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

logger = logging.getLogger(__name__)


def add_middleware(app: FastAPI) -> None:
    """Registra todos los middlewares en la app FastAPI."""

    # ── CORS ─────────────────────────────────────────────
    # Necesario para que el frontend (mismo origen o diferente)
    # pueda llamar a la API y para el WebSocket de speech
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # en producción: especificar dominio
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Logging de requests ───────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({duration:.1f}ms)"
        )
        return response
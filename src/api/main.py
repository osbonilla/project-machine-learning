"""
main.py — Aplicación FastAPI principal
Monta los routers, middlewares y sirve el frontend estático.

Uso:
    uv run uvicorn src.api.main:app --reload --port 8000
    uv run serve-api
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes import router
from src.api.middleware import add_middleware

# ─────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"


# ─────────────────────────────────────────────────────────
#  Lifespan — carga modelos al arrancar
# ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga los modelos en memoria al iniciar la API.
    Así la primera request no tiene latencia de carga.
    """
    logger.info("Iniciando API — cargando modelos...")

    from src.config import MODEL01_PATH, MODEL02_PATH

    if MODEL01_PATH.exists():
        import joblib
        app.state.model01 = joblib.load(MODEL01_PATH)
        logger.info("✓ Model01 cargado en memoria")
    else:
        app.state.model01 = None
        logger.warning("⚠ Model01 no encontrado — entrena primero con: uv run train-model01")

    if MODEL02_PATH.exists():
        import joblib
        app.state.model02 = joblib.load(MODEL02_PATH)
        logger.info("✓ Model02 cargado en memoria")
    else:
        app.state.model02 = None
        logger.warning("⚠ Model02 no encontrado — entrena primero con: uv run train-model02")

    logger.info("✓ API lista en http://localhost:8000")
    logger.info("  Docs: http://localhost:8000/docs")

    yield  # app corriendo

    logger.info("Cerrando API...")


# ─────────────────────────────────────────────────────────
#  App FastAPI
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Geo-Intent Classifier API",
    description="Clasificador de intención NLP para agente geoespacial multiagente",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middlewares ───────────────────────────────────────────
add_middleware(app)

# ── Routers ──────────────────────────────────────────────
app.include_router(router)

# ── Frontend estático (plus) ──────────────────────────────
# Sirve frontend/index.html en la raíz
# FastAPI detecta index.html automáticamente con html=True
if FRONTEND_DIR.exists():
    app.mount(
        "/app",
        StaticFiles(directory=str(FRONTEND_DIR), html=True),
        name="frontend",
    )
    logger.info(f"✓ Frontend montado en /app desde {FRONTEND_DIR}")

    @app.get("/", include_in_schema=False)
    async def root():
        """Redirige la raíz al frontend."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Geo-Intent Classifier API", "docs": "/docs"}


# ─────────────────────────────────────────────────────────
#  Entrypoint para uv run serve-api
# ─────────────────────────────────────────────────────────
def serve():
    import uvicorn
    from src.config import API_HOST, API_PORT, API_RELOAD
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )


if __name__ == "__main__":
    serve()
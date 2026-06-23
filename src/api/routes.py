"""
routes.py — Endpoints de la API
POST /predict  → clasificación de texto
GET  /health   → estado del servicio
WS   /speech   → flujo de voz en tiempo real (plus)
"""

import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from src.api.schemas import IntentRequest, IntentResponse, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────
#  POST /predict
# ─────────────────────────────────────────────────────────
@router.post(
    "/predict",
    response_model=IntentResponse,
    summary="Clasificar intención de un texto",
    tags=["Clasificación"],
)
async def predict_intent(request: IntentRequest):
    """
    Recibe un texto y devuelve el intent geoespacial detectado.
    """
    try:
        model_key = request.model if request.model in ("model01", "model02") else "model02"

        if model_key == "model01":
            from src.models.model01.predict import predict
        else:
            from src.models.model02.predict import predict

        result = predict(request.text)

        return IntentResponse(
            intent=result["intent"],
            confidence=result["confidence"],
            agent=result["agent"],
            model=result["model"],
            text=request.text,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────
#  GET /health
# ─────────────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    tags=["Sistema"],
)
async def health_check():
    """Verifica que la API y los modelos estén disponibles."""
    from src.config import MODEL01_PATH, MODEL02_PATH

    return HealthResponse(
        status="ok",
        model01=MODEL01_PATH.exists(),
        model02=MODEL02_PATH.exists(),
    )


# ─────────────────────────────────────────────────────────
#  WebSocket /speech
# ─────────────────────────────────────────────────────────
@router.websocket("/speech")
async def speech_endpoint(websocket: WebSocket):
    """WebSocket para clasificación por voz."""
    await websocket.accept()
    logger.info("WebSocket /speech conectado")

    try:
        while True:
            text = await websocket.receive_text()

            if not text.strip():
                await websocket.send_json({"error": "Texto vacío"})
                continue

            try:
                from src.models.model01.predict import predict
                result = predict(text)
                await websocket.send_json({
                    "text":       text,
                    "intent":     result["intent"],
                    "confidence": result["confidence"],
                    "agent":      result["agent"],
                })
            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket /speech desconectado")
"""
schemas.py — Modelos Pydantic para la API
Define la estructura de requests y responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class IntentRequest(BaseModel):
    """Request para predecir el intent de un texto."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Texto o instrucción del usuario",
        examples=["muéstrame las capas disponibles"],
    )
    model: Optional[str] = Field(
        default="model01",
        description="Modelo a usar: 'model01' o 'model02'",
    )


class IntentResponse(BaseModel):
    """Response con el intent clasificado."""
    intent:     str   = Field(..., description="Intent detectado")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")
    agent:      str   = Field(..., description="Agente destino")
    model:      str   = Field(..., description="Modelo usado")
    text:       str   = Field(..., description="Texto recibido")


class HealthResponse(BaseModel):
    """Response del endpoint de salud."""
    status:  str  = Field(..., description="'ok' si el servicio está activo")
    model01: bool = Field(..., description="Model01 cargado")
    model02: bool = Field(..., description="Model02 cargado")
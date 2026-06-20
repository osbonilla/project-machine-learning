"""
predict.py — Inferencia con model01
Carga el pipeline serializado y predice intents.

Uso:
    from src.models.model01.predict import predict, predict_batch
"""

import logging
import joblib
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.config import MODEL01_PATH, LABEL_ENCODER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    """Carga el pipeline y el label encoder desde disco."""
    if not MODEL01_PATH.exists():
        raise FileNotFoundError(
            f"Model01 no encontrado: {MODEL01_PATH}\n"
            "Ejecuta primero: uv run train-model01"
        )
    pipeline = joblib.load(MODEL01_PATH)
    le       = joblib.load(LABEL_ENCODER_PATH)
    return pipeline, le


def predict(text: str, pipeline=None, le=None) -> dict:
    """
    Predice el intent de un texto.

    Args:
        text    : texto de entrada
        pipeline: pipeline cargado (si None, lo carga)
        le      : LabelEncoder cargado (si None, lo carga)

    Returns:
        Dict con intent, confianza y agente destino
        {
            "intent":     "query_layer",
            "confidence": 0.94,
            "agent":      "geo_agent",
        }
    """
    if pipeline is None or le is None:
        pipeline, le = load_model()

    # Predecir clase
    y_pred = pipeline.predict([text])[0]
    intent = le.inverse_transform([y_pred])[0]

    # Probabilidad de confianza
    proba = pipeline.predict_proba([text])[0]
    confidence = float(np.max(proba))

    return {
        "intent":     intent,
        "confidence": round(confidence, 4),
        "agent":      "geo_agent",    # todos los intents van al agente geoespacial
        "model":      "model01",
    }


def predict_batch(texts: list[str], pipeline=None, le=None) -> list[dict]:
    """
    Predice el intent de una lista de textos.

    Args:
        texts   : lista de textos
        pipeline: pipeline cargado (si None, lo carga)
        le      : LabelEncoder cargado (si None, lo carga)

    Returns:
        Lista de dicts con predicciones
    """
    if pipeline is None or le is None:
        pipeline, le = load_model()

    y_preds    = pipeline.predict(texts)
    intents    = le.inverse_transform(y_preds)
    probas     = pipeline.predict_proba(texts)
    confidences = np.max(probas, axis=1)

    return [
        {
            "text":       text,
            "intent":     intent,
            "confidence": round(float(conf), 4),
            "agent":      "geo_agent",
            "model":      "model01",
        }
        for text, intent, conf in zip(texts, intents, confidences)
    ]
"""
predict.py — Inferencia con model02
Misma interfaz que model01/predict.py para intercambiabilidad.
"""

import logging
import joblib
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.config import MODEL02_PATH, LABEL_ENCODER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    if not MODEL02_PATH.exists():
        raise FileNotFoundError(
            f"Model02 no encontrado: {MODEL02_PATH}\n"
            "Ejecuta primero: uv run train-model02"
        )
    pipeline = joblib.load(MODEL02_PATH)
    le       = joblib.load(LABEL_ENCODER_PATH)
    return pipeline, le


def predict(text: str, pipeline=None, le=None) -> dict:
    """
    Predice el intent de un texto con model02.

    Returns:
        {"intent": ..., "confidence": ..., "agent": ..., "model": "model02"}
    """
    if pipeline is None or le is None:
        pipeline, le = load_model()

    y_pred     = pipeline.predict([text])[0]
    intent     = le.inverse_transform([y_pred])[0]
    proba      = pipeline.predict_proba([text])[0]
    confidence = float(np.max(proba))

    return {
        "intent":     intent,
        "confidence": round(confidence, 4),
        "agent":      "geo_agent",
        "model":      "model02",
    }


def predict_batch(texts: list[str], pipeline=None, le=None) -> list[dict]:
    """Predice el intent de una lista de textos con model02."""
    if pipeline is None or le is None:
        pipeline, le = load_model()

    y_preds     = pipeline.predict(texts)
    intents     = le.inverse_transform(y_preds)
    probas      = pipeline.predict_proba(texts)
    confidences = np.max(probas, axis=1)

    return [
        {
            "text":       text,
            "intent":     intent,
            "confidence": round(float(conf), 4),
            "agent":      "geo_agent",
            "model":      "model02",
        }
        for text, intent, conf in zip(texts, intents, confidences)
    ]
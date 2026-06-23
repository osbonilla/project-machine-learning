"""
test_api.py — Tests para los endpoints de la API FastAPI
"""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
    """Cliente de test para FastAPI."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests para GET /health."""

    def test_health_returns_200(self, client):
        """El endpoint /health debe retornar 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """La response de /health debe tener los campos correctos."""
        response = client.get("/health")
        data = response.json()
        assert "status"  in data
        assert "model01" in data
        assert "model02" in data
        assert data["status"] == "ok"


class TestPredictEndpoint:
    """Tests para POST /predict."""

    def test_predict_returns_200(self, client):
        """POST /predict con texto válido debe retornar 200."""
        from src.config import MODEL01_PATH
        if not MODEL01_PATH.exists():
            pytest.skip("model01.pkl no encontrado")

        response = client.post(
            "/predict",
            json={"text": "muéstrame las capas disponibles", "model": "model01"}
        )
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        """La response de /predict debe tener todos los campos."""
        from src.config import MODEL01_PATH
        if not MODEL01_PATH.exists():
            pytest.skip("model01.pkl no encontrado")

        response = client.post(
            "/predict",
            json={"text": "calcula el área del polígono"}
        )
        data = response.json()
        assert "intent"     in data
        assert "confidence" in data
        assert "agent"      in data
        assert "model"      in data
        assert "text"       in data

    def test_predict_confidence_range(self, client):
        """La confianza debe estar entre 0 y 1."""
        from src.config import MODEL01_PATH
        if not MODEL01_PATH.exists():
            pytest.skip("model01.pkl no encontrado")

        response = client.post(
            "/predict",
            json={"text": "exporta los datos a GeoJSON"}
        )
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_valid_intent(self, client):
        """El intent retornado debe ser uno de los definidos."""
        from src.config import MODEL01_PATH, INTENT_LABELS
        if not MODEL01_PATH.exists():
            pytest.skip("model01.pkl no encontrado")

        response = client.post(
            "/predict",
            json={"text": "filtra los polígonos por área"}
        )
        data = response.json()
        assert data["intent"] in INTENT_LABELS

    def test_predict_empty_text_fails(self, client):
        """Texto vacío debe retornar error de validación."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422  # Unprocessable Entity (Pydantic)

    def test_predict_model02(self, client):
        """POST /predict con model02 debe funcionar."""
        from src.config import MODEL02_PATH
        if not MODEL02_PATH.exists():
            pytest.skip("model02.pkl no encontrado")

        response = client.post(
            "/predict",
            json={"text": "crea un buffer de 1km", "model": "model02"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "model02"
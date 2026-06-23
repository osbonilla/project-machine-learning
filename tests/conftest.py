"""
conftest.py — Fixtures compartidos para todos los tests
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_texts():
    """Textos de ejemplo para tests."""
    return [
        "muéstrame las capas disponibles",
        "filtra los polígonos dentro del área",
        "calcula el área total de los parques",
        "qué atributos tiene la capa de ríos",
        "exporta los resultados a shapefile",
        "muestra el mapa con simbología",
        "une la capa de puntos con polígonos",
        "crea un buffer de 500 metros",
    ]


@pytest.fixture
def sample_intents():
    """Labels correspondientes a sample_texts."""
    return [
        "query_layer",
        "spatial_filter",
        "calculate_area",
        "get_attributes",
        "export_data",
        "visualize_map",
        "spatial_join",
        "buffer_analysis",
    ]


@pytest.fixture
def sample_corpus(sample_texts, sample_intents):
    """Corpus mínimo como lista de dicts."""
    return [
        {"text": t, "intent": i}
        for t, i in zip(sample_texts, sample_intents)
    ]


@pytest.fixture
def sample_dataframe(sample_corpus):
    """Corpus mínimo como DataFrame."""
    import pandas as pd
    return pd.DataFrame(sample_corpus)


@pytest.fixture
def label_encoder():
    """LabelEncoder ajustado con todos los intents."""
    from sklearn.preprocessing import LabelEncoder
    from src.config import INTENT_LABELS
    le = LabelEncoder()
    le.fit(INTENT_LABELS)
    return le
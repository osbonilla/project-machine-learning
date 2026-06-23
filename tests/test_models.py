"""
test_models.py — Tests para los pipelines de ML
"""

import pytest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL01_PATH, MODEL02_PATH, LABEL_ENCODER_PATH

# Corpus con suficientes ejemplos por clase (mínimo 3 para CalibratedCV)
TRAIN_TEXTS = [
    # query_layer
    "muéstrame las capas disponibles",
    "qué capas hay en el mapa",
    "lista las capas geoespaciales",
    "ver capas del sistema",
    # spatial_filter
    "filtra los polígonos dentro del área",
    "selecciona features por ubicación",
    "recorta la capa por el límite",
    "filtrar por geometría espacial",
    # calculate_area
    "calcula el área total de los parques",
    "cuántos metros tiene este polígono",
    "dame el perímetro de la zona",
    "calcular superficie del terreno",
    # get_attributes
    "qué atributos tiene la capa de ríos",
    "muéstrame los campos de la tabla",
    "cuáles son las propiedades del polígono",
    "ver atributos de la capa",
    # export_data
    "exporta los resultados a shapefile",
    "descarga la capa en GeoJSON",
    "guarda el mapa como KML",
    "exportar datos espaciales",
    # visualize_map
    "muestra el mapa con simbología",
    "cambia el color de la capa",
    "visualiza las zonas de riesgo",
    "renderizar mapa con estilos",
    # spatial_join
    "une la capa de puntos con polígonos",
    "cruza los datos de población",
    "combina capas por intersección",
    "join espacial de capas",
    # buffer_analysis
    "crea un buffer de 500 metros",
    "genera zona de influencia de 1km",
    "aplica buffer a los puntos",
    "zona de influencia alrededor de ríos",
]

TRAIN_INTENTS = (
    ["query_layer"]    * 4 +
    ["spatial_filter"] * 4 +
    ["calculate_area"] * 4 +
    ["get_attributes"] * 4 +
    ["export_data"]    * 4 +
    ["visualize_map"]  * 4 +
    ["spatial_join"]   * 4 +
    ["buffer_analysis"]* 4
)


class TestModel01Pipeline:

    def test_svm_pipeline_builds(self):
        from sklearn.pipeline import Pipeline
        from src.models.model01.model01 import build_svm_pipeline
        pipeline = build_svm_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert "tfidf" in pipeline.named_steps
        assert "chi2"  in pipeline.named_steps
        assert "clf"   in pipeline.named_steps

    def test_nb_pipeline_builds(self):
        from sklearn.pipeline import Pipeline
        from src.models.model01.model01 import build_nb_pipeline
        pipeline = build_nb_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_svm_pipeline_fit_predict(self):
        """SVM pipeline con cv=2 para tests con pocos datos."""
        from sklearn.preprocessing import LabelEncoder
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import SelectKBest, chi2

        le = LabelEncoder()
        y  = le.fit_transform(TRAIN_INTENTS)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 1))),
            ("chi2",  SelectKBest(chi2, k=10)),
            ("clf",   CalibratedClassifierCV(
                LinearSVC(max_iter=500, random_state=42), cv=2
            )),
        ])
        pipeline.fit(TRAIN_TEXTS, y)
        y_pred = pipeline.predict(TRAIN_TEXTS)
        assert len(y_pred) == len(TRAIN_TEXTS)

    def test_nb_pipeline_fit_predict(self):
        from sklearn.preprocessing import LabelEncoder
        from src.models.model01.model01 import build_nb_pipeline

        le = LabelEncoder()
        y  = le.fit_transform(TRAIN_INTENTS)

        pipeline = build_nb_pipeline(max_features=100, k_features=10)
        pipeline.fit(TRAIN_TEXTS, y)
        y_pred = pipeline.predict(TRAIN_TEXTS)
        assert len(y_pred) == len(TRAIN_TEXTS)

    def test_model01_pkl_exists(self):
        assert MODEL01_PATH.exists(), "Ejecuta: uv run python -m src.models.model01.train"

    def test_model01_predict_returns_valid_intent(self):
        if not MODEL01_PATH.exists():
            pytest.skip("model01.pkl no encontrado")
        from src.models.model01.predict import predict
        from src.config import INTENT_LABELS
        result = predict("muéstrame las capas disponibles")
        assert result["intent"] in INTENT_LABELS
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["agent"] == "geo_agent"

    def test_model01_predict_batch(self):
        if not MODEL01_PATH.exists():
            pytest.skip("model01.pkl no encontrado")
        from src.models.model01.predict import predict_batch
        results = predict_batch(TRAIN_TEXTS[:4])
        assert len(results) == 4
        for r in results:
            assert "intent" in r
            assert "confidence" in r

    def test_model01_accuracy_threshold(self):
        if not MODEL01_PATH.exists() or not LABEL_ENCODER_PATH.exists():
            pytest.skip("Modelos no encontrados")
        import joblib
        from src.data.dataset import load_processed
        try:
            X_train, X_test, y_train, y_test, le = load_processed()
        except FileNotFoundError:
            pytest.skip("Datos procesados no encontrados")
        pipeline = joblib.load(MODEL01_PATH)
        y_pred   = pipeline.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        assert accuracy >= 0.85, f"Accuracy {accuracy:.2f} por debajo del umbral 0.85"


class TestModel02Pipeline:

    def test_mlp_pipeline_builds(self):
        from sklearn.pipeline import Pipeline
        from src.models.model02.model02 import build_mlp_pipeline
        pipeline = build_mlp_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert "features" in pipeline.named_steps
        assert "clf"      in pipeline.named_steps

    def test_mlp_pipeline_fit_predict(self):
        """MLP sin early_stopping para tests con pocos datos."""
        from sklearn.preprocessing import LabelEncoder
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.preprocessing import MaxAbsScaler

        le = LabelEncoder()
        y  = le.fit_transform(TRAIN_INTENTS)

        pipeline = Pipeline([
            ("tfidf",  TfidfVectorizer(max_features=100, ngram_range=(1, 1))),
            ("chi2",   SelectKBest(chi2, k=10)),
            ("scaler", MaxAbsScaler()),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(32,),
                max_iter=200,
                early_stopping=False,
                random_state=42,
            )),
        ])
        pipeline.fit(TRAIN_TEXTS, y)
        y_pred = pipeline.predict(TRAIN_TEXTS)
        assert len(y_pred) == len(TRAIN_TEXTS)

    def test_model02_pkl_exists(self):
        assert MODEL02_PATH.exists(), "Ejecuta: uv run python -m src.models.model02.train"

    def test_model02_predict_returns_valid_intent(self):
        if not MODEL02_PATH.exists():
            pytest.skip("model02.pkl no encontrado")
        from src.models.model02.predict import predict
        from src.config import INTENT_LABELS
        result = predict("exporta los datos a shapefile")
        assert result["intent"] in INTENT_LABELS
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["model"] == "model02"
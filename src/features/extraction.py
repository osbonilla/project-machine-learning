"""
extraction.py — Feature extraction para clasificación de texto
Implementa múltiples estrategias de representación vectorial.

Estrategias disponibles:
    - BoW       : Bag of Words (TfidfVectorizer con binary=True)
    - TF-IDF    : Term Frequency - Inverse Document Frequency
    - TF-IDF + n-grams : unigramas + bigramas

Uso:
    from src.features.extraction import build_tfidf_pipeline, extract_features
"""

import logging
import joblib
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    PROCESSED_FILES,
    PROCESSED_DIR,
    MODELS_DIR,
)

# ─────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  Bag of Words
# ─────────────────────────────────────────────────────────
def build_bow_vectorizer(max_features: int = 3000) -> CountVectorizer:
    """
    Bag of Words: cada feature es la presencia/frecuencia de una palabra.
    Es la representación más simple — baseline de baselines.

    Args:
        max_features: máximo número de términos a considerar

    Returns:
        CountVectorizer configurado
    """
    return CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),     # solo unigramas
        min_df=2,               # ignorar términos que aparecen en < 2 docs
        max_df=0.95,            # ignorar términos que aparecen en > 95% docs
        strip_accents="unicode",
        lowercase=True,
    )


# ─────────────────────────────────────────────────────────
#  TF-IDF
# ─────────────────────────────────────────────────────────
def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple = (1, 1),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    TF-IDF: pondera cada término por qué tan informativo es.
    Términos frecuentes en pocos documentos reciben mayor peso.

    Args:
        max_features : máximo número de features
        ngram_range  : (1,1) unigramas, (1,2) uni+bigramas
        sublinear_tf : aplica log(tf) en lugar de tf crudo (recomendado)

    Returns:
        TfidfVectorizer configurado
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
    )


def build_tfidf_ngram_vectorizer(max_features: int = 8000) -> TfidfVectorizer:
    """
    TF-IDF con unigramas Y bigramas.
    Captura contexto de pares de palabras: "calcular área", "exportar shapefile".
    Mejor para intents con frases clave específicas.

    Args:
        max_features: máximo número de features

    Returns:
        TfidfVectorizer con ngram_range=(1,2)
    """
    return build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


# ─────────────────────────────────────────────────────────
#  Extracción y guardado
# ─────────────────────────────────────────────────────────
def extract_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    vectorizer_type: str = "tfidf_ngram",
    max_features: int = 5000,
) -> tuple:
    """
    Ajusta el vectorizador en train y transforma train y test.

    IMPORTANTE: el vectorizador se ajusta SOLO en X_train
    para evitar data leakage — X_test nunca influye en el vocabulario.

    Args:
        X_train        : array de textos de entrenamiento
        X_test         : array de textos de prueba
        vectorizer_type: "bow" | "tfidf" | "tfidf_ngram"
        max_features   : número máximo de features

    Returns:
        Tupla (X_train_vec, X_test_vec, vectorizer)
    """
    # Seleccionar vectorizador
    vectorizers = {
        "bow":         build_bow_vectorizer(max_features),
        "tfidf":       build_tfidf_vectorizer(max_features),
        "tfidf_ngram": build_tfidf_ngram_vectorizer(max_features),
    }

    if vectorizer_type not in vectorizers:
        raise ValueError(
            f"vectorizer_type debe ser uno de: {list(vectorizers.keys())}"
        )

    vectorizer = vectorizers[vectorizer_type]
    logger.info(f"Vectorizador seleccionado: {vectorizer_type} (max_features={max_features})")

    # Fit SOLO en train — transform en ambos
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    logger.info(f"✓ Features extraídas:")
    logger.info(f"  X_train : {X_train_vec.shape}  (ejemplos × features)")
    logger.info(f"  X_test  : {X_test_vec.shape}")
    logger.info(f"  Vocabulario: {len(vectorizer.vocabulary_)} términos")

    return X_train_vec, X_test_vec, vectorizer


def save_vectorizer(vectorizer, path: Path) -> None:
    """Serializa el vectorizador para uso en producción."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    logger.info(f"✓ Vectorizador guardado en: {path}")


def load_vectorizer(path: Path):
    """Carga un vectorizador serializado."""
    if not path.exists():
        raise FileNotFoundError(f"Vectorizador no encontrado: {path}")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────
#  Comparación de representaciones (para notebooks)
# ─────────────────────────────────────────────────────────
def compare_representations(X_train: np.ndarray) -> None:
    """
    Ajusta los 3 vectorizadores y compara sus características.
    Útil para el notebook 04_feature_extraction.ipynb.

    Args:
        X_train: array de textos de entrenamiento
    """
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE REPRESENTACIONES VECTORIALES")
    print("=" * 60)

    configs = {
        "Bag of Words":     build_bow_vectorizer(3000),
        "TF-IDF":           build_tfidf_vectorizer(5000),
        "TF-IDF + n-grams": build_tfidf_ngram_vectorizer(8000),
    }

    for name, vec in configs.items():
        X_vec = vec.fit_transform(X_train)
        # Densidad de la matriz (fracción de valores no cero)
        density = X_vec.nnz / (X_vec.shape[0] * X_vec.shape[1]) * 100

        print(f"\n{name}")
        print(f"  Shape     : {X_vec.shape}")
        print(f"  Vocab     : {len(vec.vocabulary_)} términos")
        print(f"  Densidad  : {density:.2f}% (valores no cero)")
        print(f"  Memoria   : {X_vec.data.nbytes / 1024:.1f} KB")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────
#  Top términos por intent (para EDA / notebook)
# ─────────────────────────────────────────────────────────
def top_terms_per_intent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label_encoder,
    n_terms: int = 10,
) -> dict:
    """
    Calcula los términos más frecuentes por cada intent.
    Útil para visualizar qué palabras definen cada clase.

    Args:
        X_train      : textos de entrenamiento
        y_train      : labels codificados
        label_encoder: LabelEncoder para decodificar labels
        n_terms      : número de términos top a mostrar

    Returns:
        Dict {intent: [top_terms]}
    """
    vectorizer = build_tfidf_vectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X_train)
    feature_names = vectorizer.get_feature_names_out()

    result = {}
    classes = label_encoder.classes_

    print("\n" + "=" * 60)
    print(f"TOP {n_terms} TÉRMINOS POR INTENT")
    print("=" * 60)

    for class_idx, class_name in enumerate(classes):
        # Seleccionar solo los docs de esta clase
        mask = y_train == class_idx
        X_class = X_vec[mask]

        # Sumar pesos TF-IDF por término
        mean_tfidf = np.asarray(X_class.mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[i] for i in top_indices]

        result[class_name] = top_terms
        print(f"\n{class_name}:")
        print(f"  {', '.join(top_terms)}")

    print("=" * 60 + "\n")
    return result
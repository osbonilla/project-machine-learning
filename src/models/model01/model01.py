"""
model01.py — Definición del pipeline baseline
TF-IDF + SelectKBest(chi2) + SVM / Naive Bayes

Este módulo define la arquitectura del modelo.
El entrenamiento y evaluación están en train.py.
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV


def build_svm_pipeline(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    k_features: int = 2000,
    C: float = 1.0,
) -> Pipeline:
    """
    Pipeline baseline con TF-IDF + chi2 + LinearSVC.

    LinearSVC es la opción preferida para clasificación de texto:
    - Muy rápido en datasets grandes
    - Funciona bien con alta dimensionalidad (TF-IDF)
    - Excelente en clasificación multiclase con pocos datos

    CalibratedClassifierCV envuelve LinearSVC para obtener
    probabilidades (predict_proba) necesarias para el score de confianza.

    Args:
        max_features: máximo de términos en TF-IDF
        ngram_range : rango de n-gramas (1,1) o (1,2)
        k_features  : features a conservar tras chi2
        C           : parámetro de regularización de SVM

    Returns:
        Pipeline sklearn listo para fit/predict
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("chi2", SelectKBest(chi2, k=k_features)),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=2000, random_state=42),
            cv=3,
        )),
    ])


def build_nb_pipeline(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    k_features: int = 2000,
    alpha: float = 1.0,
) -> Pipeline:
    """
    Pipeline alternativo con TF-IDF + chi2 + Multinomial Naive Bayes.

    MultinomialNB es el baseline clásico para clasificación de texto:
    - Extremadamente rápido
    - Funciona bien con vocabularios grandes
    - Interpretable (puedes ver qué palabras definen cada clase)

    Args:
        max_features: máximo de términos en TF-IDF
        ngram_range : rango de n-gramas
        k_features  : features a conservar tras chi2
        alpha       : suavizado de Laplace

    Returns:
        Pipeline sklearn listo para fit/predict
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=False,    # NB funciona mejor sin log-scaling
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("chi2", SelectKBest(chi2, k=k_features)),
        ("clf", MultinomialNB(alpha=alpha)),
    ])


# Pipelines disponibles para GridSearchCV
AVAILABLE_PIPELINES = {
    "svm": build_svm_pipeline,
    "nb":  build_nb_pipeline,
}
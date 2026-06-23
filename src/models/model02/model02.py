"""
model02.py — Definición del pipeline avanzado
TF-IDF con características enriquecidas + MLPClassifier

Diferencia respecto a model01:
    - Usa TF-IDF con configuración más rica (char n-grams + word n-grams)
    - Clasificador MLP (red neuronal densa) en lugar de SVM/NB
    - Captura patrones más complejos en el texto

Justificación de MLP sobre transformers:
    - Dataset pequeño (~640 ejemplos): transformers tienden a overfittear
    - MLP con buenas features TF-IDF es competitivo y mucho más liviano
    - No requiere GPU ni torch (evita el problema de instalación)
"""

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler


def build_mlp_pipeline(
    max_features_word: int = 5000,
    max_features_char: int = 3000,
    k_features: int = 3000,
    hidden_layer_sizes: tuple = (256, 128),
    alpha: float = 0.001,
    learning_rate_init: float = 0.001,
) -> Pipeline:
    """
    Pipeline avanzado con TF-IDF word + char n-grams + MLP.

    Usa FeatureUnion para combinar dos vectorizadores:
        1. TF-IDF a nivel de palabra  (word n-grams)
        2. TF-IDF a nivel de caracter (char n-grams)

    Los char n-grams capturan morfología y son robustos
    a errores de escritura — útil para comandos en lenguaje natural.

    Args:
        max_features_word  : features del vectorizador de palabras
        max_features_char  : features del vectorizador de caracteres
        k_features         : features a conservar tras chi2
        hidden_layer_sizes : arquitectura de la red (capas ocultas)
        alpha              : regularización L2 del MLP
        learning_rate_init : tasa de aprendizaje inicial

    Returns:
        Pipeline sklearn listo para fit/predict
    """
    # Combinar word n-grams y char n-grams
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            max_features=max_features_word,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            analyzer="word",
            strip_accents="unicode",
            lowercase=True,
        )),
        ("char_tfidf", TfidfVectorizer(
            max_features=max_features_char,
            ngram_range=(2, 4),     # bigramas a 4-gramas de caracteres
            sublinear_tf=True,
            min_df=3,
            analyzer="char_wb",    # char dentro de límites de palabra
            strip_accents="unicode",
            lowercase=True,
        )),
    ])

    return Pipeline([
        ("features", feature_union),
        ("chi2",     SelectKBest(chi2, k=k_features)),
        ("scaler",   MaxAbsScaler()),    # normalizar para MLP (no afecta sparsity)
        ("clf",      MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate="adaptive",
            learning_rate_init=learning_rate_init,
            max_iter=500,
            early_stopping=True,        # detiene si val_loss no mejora
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )),
    ])
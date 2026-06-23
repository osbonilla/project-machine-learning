"""
selection.py — Feature selection para clasificación de texto
Implementa métodos de tipo filter para reducir dimensionalidad.

Métodos implementados:
    - Chi-cuadrado (chi2)        : mide dependencia entre feature e intent
    - Mutual Information (MI)    : mide información compartida
    - Variance Threshold         : elimina features con varianza casi cero

Nota: Para transfer learning o texto con la rúbrica lo marca como opcional.
      Se puede omitir si se usa TF-IDF + SVM (que maneja bien alta dimensionalidad).

Uso:
    from src.features.selection import select_features, compare_selection_methods
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import FIGURES_DIR

# ─────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  Métodos filter de selección
# ─────────────────────────────────────────────────────────
def select_chi2(X_train, y_train, X_test, k: int = 2000):
    """
    Selección por Chi-cuadrado.
    Mide la dependencia estadística entre cada feature y la clase.
    Solo funciona con valores no negativos (TF-IDF / BoW son válidos).

    Args:
        X_train: matriz de features de entrenamiento (sparse)
        y_train: labels de entrenamiento
        X_test : matriz de features de prueba
        k      : número de features a conservar

    Returns:
        Tupla (X_train_sel, X_test_sel, selector)
    """
    selector = SelectKBest(chi2, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    logger.info(f"Chi2 selection: {X_train.shape[1]} → {X_train_sel.shape[1]} features")
    return X_train_sel, X_test_sel, selector


def select_mutual_info(X_train, y_train, X_test, k: int = 2000):
    """
    Selección por Mutual Information.
    Mide cuánta información aporta cada feature sobre la clase.
    Más robusto que chi2 pero más lento.

    Args:
        X_train: matriz de features de entrenamiento
        y_train: labels de entrenamiento
        X_test : matriz de features de prueba
        k      : número de features a conservar

    Returns:
        Tupla (X_train_sel, X_test_sel, selector)
    """
    # Convertir sparse a dense si es necesario para MI
    X_train_dense = X_train.toarray() if issparse(X_train) else X_train
    X_test_dense  = X_test.toarray()  if issparse(X_test)  else X_test

    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_dense, y_train)
    X_test_sel  = selector.transform(X_test_dense)

    logger.info(f"Mutual Info selection: {X_train.shape[1]} → {X_train_sel.shape[1]} features")
    return X_train_sel, X_test_sel, selector


def select_variance_threshold(X_train, X_test, threshold: float = 0.0):
    """
    Elimina features con varianza igual o menor al threshold.
    Con threshold=0.0 elimina features constantes (siempre el mismo valor).

    Args:
        X_train  : matriz de features de entrenamiento
        X_test   : matriz de features de prueba
        threshold: umbral de varianza mínima

    Returns:
        Tupla (X_train_sel, X_test_sel, selector)
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel  = selector.transform(X_test)

    removed = X_train.shape[1] - X_train_sel.shape[1]
    logger.info(
        f"Variance Threshold (threshold={threshold}): "
        f"{X_train.shape[1]} → {X_train_sel.shape[1]} features "
        f"({removed} eliminadas)"
    )
    return X_train_sel, X_test_sel, selector


# ─────────────────────────────────────────────────────────
#  Función principal de selección
# ─────────────────────────────────────────────────────────
def select_features(
    X_train,
    y_train,
    X_test,
    method: str = "chi2",
    k: int = 2000,
):
    """
    Aplica el método de selección indicado.

    Args:
        X_train: matriz de features de entrenamiento
        y_train: labels de entrenamiento
        X_test : matriz de features de prueba
        method : "chi2" | "mutual_info" | "variance"
        k      : número de features a conservar (no aplica a variance)

    Returns:
        Tupla (X_train_sel, X_test_sel, selector)
    """
    methods = {
        "chi2":        lambda: select_chi2(X_train, y_train, X_test, k),
        "mutual_info": lambda: select_mutual_info(X_train, y_train, X_test, k),
        "variance":    lambda: select_variance_threshold(X_train, X_test),
    }

    if method not in methods:
        raise ValueError(f"method debe ser uno de: {list(methods.keys())}")

    logger.info(f"Aplicando feature selection: {method} (k={k})")
    return methods[method]()


# ─────────────────────────────────────────────────────────
#  Comparación de métodos (para notebook 05)
# ─────────────────────────────────────────────────────────
def compare_selection_methods(
    X_train,
    y_train,
    X_test,
    k_values: list[int] = [500, 1000, 2000, 3000],
) -> dict:
    """
    Compara chi2 y mutual_info con distintos valores de k.
    Útil para el notebook 05_feature_selection.ipynb.

    Args:
        X_train : matriz de features de entrenamiento
        y_train : labels
        X_test  : matriz de features de prueba
        k_values: lista de valores de k a probar

    Returns:
        Dict con shapes resultantes por método y k
    """
    print("\n" + "=" * 55)
    print("COMPARACIÓN DE MÉTODOS DE FEATURE SELECTION")
    print("=" * 55)
    print(f"{'Método':<18} {'k':>6} {'Features orig':>14} {'Features sel':>13}")
    print("-" * 55)

    results = {}

    for k in k_values:
        # Chi2
        X_tr_chi2, _, _ = select_chi2(X_train, y_train, X_test, k=k)
        results[f"chi2_k{k}"] = X_tr_chi2.shape[1]
        print(f"{'chi2':<18} {k:>6} {X_train.shape[1]:>14} {X_tr_chi2.shape[1]:>13}")

        # Mutual Info
        X_tr_mi, _, _ = select_mutual_info(X_train, y_train, X_test, k=k)
        results[f"mi_k{k}"] = X_tr_mi.shape[1]
        print(f"{'mutual_info':<18} {k:>6} {X_train.shape[1]:>14} {X_tr_mi.shape[1]:>13}")

    print("=" * 55 + "\n")
    return results


# ─────────────────────────────────────────────────────────
#  Visualización de scores (para notebook 05)
# ─────────────────────────────────────────────────────────
def plot_feature_scores(
    X_train,
    y_train,
    vectorizer,
    method: str = "chi2",
    top_n: int = 20,
    save: bool = True,
) -> None:
    """
    Grafica los top_n features con mayor score según el método.

    Args:
        X_train   : matriz de features de entrenamiento
        y_train   : labels
        vectorizer: vectorizador ajustado (para obtener nombres de features)
        method    : "chi2" | "mutual_info"
        top_n     : cuántos features mostrar
        save      : guardar figura en reports/figures/
    """
    feature_names = vectorizer.get_feature_names_out()

    if method == "chi2":
        scores, _ = chi2(X_train, y_train)
        title = "Top features por Chi-cuadrado"
    else:
        scores = mutual_info_classif(
            X_train.toarray() if issparse(X_train) else X_train,
            y_train,
        )
        title = "Top features por Mutual Information"

    # Ordenar y tomar top_n
    top_indices = np.argsort(scores)[-top_n:][::-1]
    top_scores  = scores[top_indices]
    top_names   = feature_names[top_indices]

    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_scores[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"feature_scores_{method}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Figura guardada: {path}")

    plt.show()
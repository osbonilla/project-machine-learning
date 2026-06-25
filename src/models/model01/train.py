"""
train.py — Entrenamiento, tuning y evaluación de model01
Pipeline: TF-IDF + chi2 + SVM / Naive Bayes

Uso:
    uv run train-model01
    python -m src.models.model01.train
"""

import logging
import joblib
import pandas as pd
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_val_score,
    learning_curve,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.config import (
    MODEL01_PATH,
    LABEL_ENCODER_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    CV_N_SPLITS,
    CV_N_REPEATS,
    RANDOM_STATE,
    INTENT_LABELS,
)
from src.data.dataset import load_processed
from src.models.model01.model01 import build_svm_pipeline, build_nb_pipeline

# ─────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  Hiperparámetros a buscar con GridSearchCV
# ─────────────────────────────────────────────────────────
PARAM_GRID_SVM = {
    "tfidf__max_features":  [3000, 5000, 8000],
    "tfidf__ngram_range":   [(1, 1), (1, 2)],
    "chi2__k":              [1000, 2000, 3000],
    "clf__estimator__C":    [0.1, 1.0, 10.0],
}

PARAM_GRID_NB = {
    "tfidf__max_features": [3000, 5000, 8000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "chi2__k":             [1000, 2000, 3000],
    "clf__alpha":          [0.1, 0.5, 1.0, 2.0],
}


# ─────────────────────────────────────────────────────────
#  Tuning con Repeated K-Fold CV
# ─────────────────────────────────────────────────────────
def tune_pipeline(pipeline, param_grid, X_train, y_train, pipeline_name: str):
    """
    Busca los mejores hiperparámetros con GridSearchCV
    usando RepeatedStratifiedKFold (exigido por la rúbrica).
    """
    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    logger.info(f"Iniciando GridSearchCV para {pipeline_name}...")
    logger.info(f"CV: {CV_N_REPEATS} repeticiones × {CV_N_SPLITS} folds")
    logger.info(f"Combinaciones a probar: {_count_combinations(param_grid)}")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)

    logger.info(f"✓ Mejores hiperparámetros ({pipeline_name}):")
    for param, value in search.best_params_.items():
        logger.info(f"    {param}: {value}")
    logger.info(f"  Mejor F1-macro (CV): {search.best_score_:.4f}")

    return search


def _count_combinations(param_grid: dict) -> int:
    """Cuenta el número total de combinaciones en la grilla."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


# ─────────────────────────────────────────────────────────
#  Curva de aprendizaje
# ─────────────────────────────────────────────────────────
def plot_learning_curve(pipeline, X_train, y_train, model_name: str = "model01") -> None:
    """Genera y guarda la curva de aprendizaje."""
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=RANDOM_STATE
    )

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_sizes, train_mean, "o-", color="royalblue",
            label="Train score", linewidth=2)
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color="royalblue")

    ax.plot(train_sizes, val_mean, "o-", color="tomato",
            label="Validation score", linewidth=2)
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color="tomato")

    ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=11)
    ax.set_ylabel("F1-macro", fontsize=11)
    ax.set_title(f"Curva de aprendizaje — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"learning_curve_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Curva de aprendizaje guardada: {path}")
    plt.show()


# ─────────────────────────────────────────────────────────
#  Evaluación final
# ─────────────────────────────────────────────────────────
def evaluate(pipeline, X_test, y_test, label_encoder) -> dict:
    """Evalúa el pipeline en el conjunto de test."""
    y_pred = pipeline.predict(X_test)
    class_names = label_encoder.classes_

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
    )

    print("\n" + "=" * 60)
    print("EVALUACIÓN EN TEST SET — MODEL01")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Matriz de Confusión — Model01", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "confusion_matrix_model01.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Matriz de confusión guardada: {path}")
    plt.show()

    return report


# ─────────────────────────────────────────────────────────
#  Entrypoint principal
# ─────────────────────────────────────────────────────────
def main():
    logger.info("=" * 55)
    logger.info("ENTRENAMIENTO MODEL01 — TF-IDF + chi2 + SVM/NB")
    logger.info("=" * 55)

    # 1. Cargar datos procesados
    X_train, X_test, y_train, y_test, le = load_processed()

    # ── SVM ──────────────────────────────────────────────
    logger.info("\n[1/2] Tuning SVM pipeline...")
    svm_search = tune_pipeline(
        pipeline=build_svm_pipeline(),
        param_grid=PARAM_GRID_SVM,
        X_train=X_train,
        y_train=y_train,
        pipeline_name="SVM",
    )
    best_svm = svm_search.best_estimator_

    # Guardar resultados del CV para graficar en el notebook
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results_svm = pd.DataFrame(svm_search.cv_results_)
    cv_results_svm.to_csv(MODELS_DIR / "cv_results_model01.csv", index=False)
    logger.info(f"✓ Resultados CV-SVM guardados en: {MODELS_DIR / 'cv_results_model01.csv'}")

    # ── Naive Bayes ───────────────────────────────────────
    logger.info("\n[2/2] Tuning Naive Bayes pipeline...")
    nb_search = tune_pipeline(
        pipeline=build_nb_pipeline(),
        param_grid=PARAM_GRID_NB,
        X_train=X_train,
        y_train=y_train,
        pipeline_name="NaiveBayes",
    )
    best_nb = nb_search.best_estimator_

    # Guardar resultados del CV de NB también
    cv_results_nb = pd.DataFrame(nb_search.cv_results_)
    cv_results_nb.to_csv(MODELS_DIR / "cv_results_nb.csv", index=False)
    logger.info(f"✓ Resultados CV-NB guardados en: {MODELS_DIR / 'cv_results_nb.csv'}")

    # ── Seleccionar el mejor entre SVM y NB ───────────────
    logger.info("\nComparando SVM vs NaiveBayes...")
    logger.info(f"  SVM  F1-macro CV: {svm_search.best_score_:.4f}")
    logger.info(f"  NB   F1-macro CV: {nb_search.best_score_:.4f}")

    if svm_search.best_score_ >= nb_search.best_score_:
        best_pipeline = best_svm
        best_name = "SVM"
    else:
        best_pipeline = best_nb
        best_name = "NaiveBayes"

    logger.info(f"✓ Mejor modelo seleccionado: {best_name}")

    # Guardar el nombre del mejor clasificador
    with open(MODELS_DIR / "best_model01_name.txt", "w") as f:
        f.write(best_name)

    # Guardar scores finales para el notebook
    scores_summary = {
        "svm_best_score":  svm_search.best_score_,
        "nb_best_score":   nb_search.best_score_,
        "best_classifier": best_name,
        "svm_best_params": str(svm_search.best_params_),
        "nb_best_params":  str(nb_search.best_params_),
    }
    pd.DataFrame([scores_summary]).to_csv(
        MODELS_DIR / "model01_summary.csv", index=False
    )
    logger.info(f"✓ Resumen guardado en: {MODELS_DIR / 'model01_summary.csv'}")

    # ── Curva de aprendizaje ──────────────────────────────
    logger.info("\nGenerando curva de aprendizaje...")
    plot_learning_curve(best_pipeline, X_train, y_train, model_name="model01")

    # ── Evaluación en test ────────────────────────────────
    logger.info("\nEvaluando en test set...")
    evaluate(best_pipeline, X_test, y_test, le)

    # ── Guardar modelo ────────────────────────────────────
    joblib.dump(best_pipeline, MODEL01_PATH)
    logger.info(f"\n✓ Model01 guardado en: {MODEL01_PATH}")


if __name__ == "__main__":
    main()
"""
train.py — Entrenamiento, tuning y evaluación de model02
Pipeline: TF-IDF (word+char) + chi2 + MLP

Uso:
    uv run train-model02
    python -m src.models.model02.train
"""

import logging
import joblib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV,
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
    MODEL02_PATH,
    LABEL_ENCODER_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    CV_N_SPLITS,
    CV_N_REPEATS,
    RANDOM_STATE,
)
from src.data.dataset import load_processed
from src.models.model02.model02 import build_mlp_pipeline

# ─────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  Grilla de hiperparámetros
# ─────────────────────────────────────────────────────────
PARAM_GRID_MLP = {
    "features__word_tfidf__max_features": [3000, 5000],
    "features__char_tfidf__max_features": [2000, 3000],
    "chi2__k":                            [2000, 3000],
    "clf__hidden_layer_sizes":            [(128, 64), (256, 128)],
    "clf__alpha":                         [0.0001, 0.001, 0.01],
}


# ─────────────────────────────────────────────────────────
#  Tuning con Repeated K-Fold CV
# ─────────────────────────────────────────────────────────
def tune_pipeline(pipeline, param_grid, X_train, y_train):
    """
    Busca los mejores hiperparámetros con GridSearchCV
    usando RepeatedStratifiedKFold (10×10).
    """
    cv = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    total = 1
    for v in param_grid.values():
        total *= len(v)

    logger.info(f"Iniciando GridSearchCV MLP — {total} combinaciones × {CV_N_REPEATS}×{CV_N_SPLITS} CV")

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

    logger.info("✓ Mejores hiperparámetros MLP:")
    for param, value in search.best_params_.items():
        logger.info(f"    {param}: {value}")
    logger.info(f"  Mejor F1-macro (CV): {search.best_score_:.4f}")

    return search


# ─────────────────────────────────────────────────────────
#  Curva de aprendizaje
# ─────────────────────────────────────────────────────────
def plot_learning_curve(pipeline, X_train, y_train, model_name: str = "model02") -> None:
    """Genera y guarda la curva de aprendizaje."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

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
    y_pred     = pipeline.predict(X_test)
    class_names = label_encoder.classes_

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
    )

    print("\n" + "=" * 60)
    print("EVALUACIÓN EN TEST SET — MODEL02")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    fig, ax = plt.subplots(figsize=(10, 8))
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Matriz de Confusión — Model02", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = FIGURES_DIR / "confusion_matrix_model02.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"✓ Matriz de confusión guardada: {path}")
    plt.show()

    return report


# ─────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────
def main():
    logger.info("=" * 55)
    logger.info("ENTRENAMIENTO MODEL02 — TF-IDF (word+char) + MLP")
    logger.info("=" * 55)

    # 1. Cargar datos
    X_train, X_test, y_train, y_test, le = load_processed()

    # 2. Tuning
    search = tune_pipeline(
        pipeline=build_mlp_pipeline(),
        param_grid=PARAM_GRID_MLP,
        X_train=X_train,
        y_train=y_train,
    )
    best_pipeline = search.best_estimator_

    # 3. Curva de aprendizaje
    logger.info("\nGenerando curva de aprendizaje...")
    plot_learning_curve(best_pipeline, X_train, y_train, model_name="model02")

    # 4. Evaluación en test
    logger.info("\nEvaluando en test set...")
    evaluate(best_pipeline, X_test, y_test, le)

    # 5. Guardar modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL02_PATH)
    logger.info(f"\n✓ Model02 guardado en: {MODEL02_PATH}")


if __name__ == "__main__":
    main()
"""
dataset.py — Carga, valida y splitea el corpus
Lee el archivo limpio y produce X_train, X_test, y_train, y_test

Uso:
    from src.data.dataset import load_dataset, split_and_save
"""

import json
import logging
import joblib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CLEAN_DATA_PATH,
    RAW_DATA_PATH,
    PROCESSED_DIR,
    PROCESSED_FILES,
    LABEL_ENCODER_PATH,
    INTENT_LABELS,
    TEST_SIZE,
    RANDOM_STATE,
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
#  Carga
# ─────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> pd.DataFrame:
    """
    Lee un archivo JSONL y lo convierte en DataFrame.

    Args:
        path: ruta al archivo .jsonl

    Returns:
        DataFrame con columnas [text, intent]
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {path}\n"
            "Asegúrate de haber ejecutado primero: uv run generate-data"
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Línea {line_num} inválida, ignorada: {line[:50]}")

    df = pd.DataFrame(records)
    logger.info(f"Cargados {len(df)} registros desde {path.name}")
    return df


def load_dataset(use_clean: bool = True) -> pd.DataFrame:
    """
    Carga el corpus. Usa el archivo limpio si existe,
    si no usa el crudo.

    Args:
        use_clean: si True intenta cargar intents_clean.jsonl primero

    Returns:
        DataFrame con columnas [text, intent]
    """
    if use_clean and CLEAN_DATA_PATH.exists():
        logger.info("Cargando datos limpios (interim)...")
        df = load_jsonl(CLEAN_DATA_PATH)
    else:
        logger.info("Datos limpios no encontrados, cargando datos crudos (raw)...")
        df = load_jsonl(RAW_DATA_PATH)

    return df


# ─────────────────────────────────────────────────────────
#  Validación
# ─────────────────────────────────────────────────────────
def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Valida que el DataFrame tenga la estructura y calidad esperada.

    Args:
        df: DataFrame a validar

    Returns:
        True si es válido, lanza ValueError si no
    """
    errors = []

    # Columnas requeridas
    required_cols = {"text", "intent"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Columnas faltantes: {missing}")

    # Sin valores nulos
    nulls = df[list(required_cols & set(df.columns))].isnull().sum()
    if nulls.any():
        errors.append(f"Valores nulos encontrados:\n{nulls[nulls > 0]}")

    # Intents válidos
    if "intent" in df.columns:
        unknown = set(df["intent"].unique()) - set(INTENT_LABELS)
        if unknown:
            errors.append(f"Intents desconocidos: {unknown}")

    # Textos no vacíos
    if "text" in df.columns:
        empty = (df["text"].str.strip() == "").sum()
        if empty > 0:
            errors.append(f"{empty} textos vacíos encontrados")

    if errors:
        raise ValueError("Dataset inválido:\n" + "\n".join(errors))

    logger.info("✓ Validación del dataset: OK")
    return True


# ─────────────────────────────────────────────────────────
#  Estadísticas
# ─────────────────────────────────────────────────────────
def describe_dataset(df: pd.DataFrame) -> None:
    """Imprime un resumen del dataset."""
    print("\n" + "=" * 50)
    print("RESUMEN DEL DATASET")
    print("=" * 50)
    print(f"Total ejemplos  : {len(df)}")
    print(f"Intents únicos  : {df['intent'].nunique()}")
    print(f"Longitud media  : {df['text'].str.split().str.len().mean():.1f} palabras")
    print(f"Longitud mínima : {df['text'].str.split().str.len().min()} palabras")
    print(f"Longitud máxima : {df['text'].str.split().str.len().max()} palabras")
    print("\nDistribución por intent:")
    counts = df["intent"].value_counts()
    for intent, count in counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {intent:<22} {count:>4} ({pct:.1f}%)  {bar}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────
#  Split y guardado
# ─────────────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> tuple:
    """
    Divide el corpus en train/test, codifica los labels
    y guarda los archivos .pkl en data/processed/.

    Args:
        df: DataFrame con columnas [text, intent]

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    # Separar features y labels
    X = df["text"].values          # array de strings
    y_raw = df["intent"].values    # array de strings con nombre del intent

    # Codificar labels a enteros (LabelEncoder)
    # query_layer → 0, spatial_filter → 1, etc.
    le = LabelEncoder()
    le.fit(INTENT_LABELS)          # ajustar con todos los intents conocidos
    y = le.transform(y_raw)        # transformar a enteros

    # Split estratificado (mantiene proporciones de clases)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,                # importante: balancear clases en train y test
    )

    logger.info(f"Split completado:")
    logger.info(f"  Train : {len(X_train)} ejemplos ({(1-TEST_SIZE)*100:.0f}%)")
    logger.info(f"  Test  : {len(X_test)} ejemplos ({TEST_SIZE*100:.0f}%)")

    # Guardar en data/processed/
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train, PROCESSED_FILES["X_train"])
    joblib.dump(X_test,  PROCESSED_FILES["X_test"])
    joblib.dump(y_train, PROCESSED_FILES["y_train"])
    joblib.dump(y_test,  PROCESSED_FILES["y_test"])
    joblib.dump(le,      LABEL_ENCODER_PATH)

    logger.info(f"✓ Archivos guardados en: {PROCESSED_DIR}")
    logger.info(f"✓ LabelEncoder guardado en: {LABEL_ENCODER_PATH}")
    logger.info(f"  Clases: {list(le.classes_)}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────
#  Función de conveniencia: carga los .pkl ya procesados
# ─────────────────────────────────────────────────────────
def load_processed() -> tuple:
    """
    Carga los splits ya guardados en data/processed/.
    Útil para los módulos de entrenamiento sin reprocesar.

    Returns:
        Tupla (X_train, X_test, y_train, y_test, label_encoder)
    """
    for key, path in PROCESSED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{key}.pkl no encontrado en {path}\n"
                "Ejecuta primero el pipeline de preprocesamiento."
            )

    X_train = joblib.load(PROCESSED_FILES["X_train"])
    X_test  = joblib.load(PROCESSED_FILES["X_test"])
    y_train = joblib.load(PROCESSED_FILES["y_train"])
    y_test  = joblib.load(PROCESSED_FILES["y_test"])
    le      = joblib.load(LABEL_ENCODER_PATH)

    logger.info(f"✓ Datos procesados cargados:")
    logger.info(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, le


# ─────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────
def main():
    logger.info("Iniciando pipeline de dataset...")

    # 1. Cargar
    df = load_dataset(use_clean=True)

    # 2. Validar
    validate_dataset(df)

    # 3. Describir
    describe_dataset(df)

    # 4. Split y guardar
    X_train, X_test, y_train, y_test = split_and_save(df)

    logger.info("✓ Pipeline de dataset completado")


if __name__ == "__main__":
    main()
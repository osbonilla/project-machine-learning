"""
dataloader.py — Carga de datos específica para model01
Wrapper sobre dataset.py con validaciones propias del modelo.
"""

import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.data.dataset import load_dataset, load_processed, validate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_train_data() -> tuple:
    """
    Devuelve X_train, y_train listos para entrenar model01.
    Carga desde los .pkl procesados si existen.
    """
    try:
        X_train, X_test, y_train, y_test, le = load_processed()
        logger.info("✓ Datos cargados desde processed/")
        return X_train, X_test, y_train, y_test, le
    except FileNotFoundError:
        logger.warning("Processed no encontrado, cargando desde raw/interim...")
        df = load_dataset(use_clean=True)
        validate_dataset(df)
        from src.data.dataset import split_and_save
        X_train, X_test, y_train, y_test = split_and_save(df)
        import joblib
        from src.config import LABEL_ENCODER_PATH
        le = joblib.load(LABEL_ENCODER_PATH)
        return X_train, X_test, y_train, y_test, le
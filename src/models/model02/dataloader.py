"""
dataloader.py — Carga de datos para model02
Igual que model01/dataloader.py, reutiliza load_processed.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.models.model01.dataloader import get_train_data  # reutilizar lógica

# Reexportar para que model02 tenga su propio dataloader
__all__ = ["get_train_data"]
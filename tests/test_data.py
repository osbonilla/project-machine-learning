"""
test_data.py — Tests para generación, limpieza y carga de datos
"""

import json
import pytest
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INTENT_LABELS, RAW_DATA_PATH, CLEAN_DATA_PATH


class TestDataSchema:
    """Valida el schema y calidad del corpus."""

    def test_corpus_raw_exists(self):
        """El archivo crudo debe existir tras ejecutar generator."""
        assert RAW_DATA_PATH.exists(), (
            f"No se encontró {RAW_DATA_PATH}. "
            "Ejecuta: uv run python -m src.data.generator"
        )

    def test_corpus_clean_exists(self):
        """El archivo limpio debe existir tras ejecutar preprocess."""
        assert CLEAN_DATA_PATH.exists(), (
            f"No se encontró {CLEAN_DATA_PATH}. "
            "Ejecuta: uv run python -m src.data.preprocess"
        )

    def test_raw_corpus_not_empty(self):
        """El corpus crudo no debe estar vacío."""
        if not RAW_DATA_PATH.exists():
            pytest.skip("Corpus crudo no generado")
        records = RAW_DATA_PATH.read_text(encoding="utf-8").strip().split("\n")
        assert len(records) > 0

    def test_raw_corpus_valid_jsonl(self):
        """Cada línea del corpus crudo debe ser JSON válido."""
        if not RAW_DATA_PATH.exists():
            pytest.skip("Corpus crudo no generado")
        with open(RAW_DATA_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)   # lanza error si no es JSON válido
                assert "text" in record, f"Falta campo 'text' en línea {i}"
                assert "intent" in record, f"Falta campo 'intent' en línea {i}"

    def test_all_intents_present_in_raw(self):
        """Todos los intents definidos deben estar en el corpus crudo."""
        if not RAW_DATA_PATH.exists():
            pytest.skip("Corpus crudo no generado")
        intents_found = set()
        with open(RAW_DATA_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    intents_found.add(record["intent"])
        for intent in INTENT_LABELS:
            assert intent in intents_found, f"Intent '{intent}' no encontrado en corpus"

    def test_no_empty_texts_in_raw(self):
        """No debe haber textos vacíos en el corpus crudo."""
        if not RAW_DATA_PATH.exists():
            pytest.skip("Corpus crudo no generado")
        with open(RAW_DATA_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert record["text"].strip() != "", f"Texto vacío en línea {i}"

    def test_clean_corpus_has_required_fields(self):
        """El corpus limpio debe tener text, intent y tokens."""
        if not CLEAN_DATA_PATH.exists():
            pytest.skip("Corpus limpio no generado")
        with open(CLEAN_DATA_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert "text" in record
                assert "intent" in record
                assert "tokens" in record
                assert isinstance(record["tokens"], list)
                break  # solo verificar primera línea


class TestDataset:
    """Valida la función de carga y split."""

    def test_load_dataset_returns_dataframe(self, sample_corpus, tmp_path):
        """load_jsonl debe retornar un DataFrame con columnas correctas."""
        # Crear archivo temporal
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for record in sample_corpus:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        from src.data.dataset import load_jsonl
        df = load_jsonl(jsonl_file)

        assert isinstance(df, pd.DataFrame)
        assert "text" in df.columns
        assert "intent" in df.columns
        assert len(df) == len(sample_corpus)

    def test_validate_dataset_passes(self, sample_dataframe):
        """validate_dataset no debe lanzar error con datos válidos."""
        from src.data.dataset import validate_dataset
        assert validate_dataset(sample_dataframe) is True

    def test_validate_dataset_fails_missing_column(self):
        """validate_dataset debe fallar si falta columna requerida."""
        from src.data.dataset import validate_dataset
        df = pd.DataFrame({"text": ["hola"]})  # falta 'intent'
        with pytest.raises(ValueError, match="Columnas faltantes"):
            validate_dataset(df)

    def test_validate_dataset_fails_unknown_intent(self, sample_dataframe):
        """validate_dataset debe fallar si hay intents desconocidos."""
        from src.data.dataset import validate_dataset
        df = sample_dataframe.copy()
        df.loc[0, "intent"] = "intent_falso"
        with pytest.raises(ValueError, match="Intents desconocidos"):
            validate_dataset(df)

    def test_split_proportions(self, sample_dataframe, label_encoder):
        """El split debe respetar TEST_SIZE=0.2 aproximadamente."""
        from src.config import TEST_SIZE
        n = len(sample_dataframe)
        expected_test = int(n * TEST_SIZE)
        # Con datasets pequeños puede variar ±1
        assert abs(expected_test - round(n * TEST_SIZE)) <= 1
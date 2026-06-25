"""
preprocess.py — Limpieza y normalización del corpus de texto
Lee intents_raw.jsonl y produce intents_clean.jsonl

Pasos:
    1. Eliminar numeración al inicio de la frase
    2. Minúsculas
    3. Eliminar caracteres especiales y puntuación
    4. Normalizar espacios
    5. Eliminar stopwords (spaCy)
    6. Lematización (spaCy)

Uso:
    from src.data.preprocess import clean_text, preprocess_corpus
"""

import json
import re
import logging
from pathlib import Path

import spacy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    INTERIM_DIR,
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
#  Cargar modelo spaCy
# ─────────────────────────────────────────────────────────
def load_spacy_model():
    """Carga el modelo de spaCy para español."""
    try:
        nlp = spacy.load("es_core_news_sm")
        logger.info("✓ Modelo spaCy 'es_core_news_sm' cargado")
        return nlp
    except OSError:
        logger.error(
            "Modelo spaCy no encontrado.\n"
            "Ejecuta: uv run python -m spacy download es_core_news_sm"
        )
        raise


# ─────────────────────────────────────────────────────────
#  Limpieza de texto
# ─────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Normalización básica antes de pasar por spaCy.

    Pasos:
        - Eliminar numeración al inicio: "18. texto" → "texto"
        - Minúsculas
        - Eliminar caracteres especiales (conserva letras, números y espacios)
        - Normalizar espacios múltiples

    Args:
        text: texto crudo

    Returns:
        texto normalizado
    """
    # Eliminar numeración al inicio de la frase: "18. texto" o "18) texto" → "texto"
    text = re.sub(r'^\d+[\.\)]\s*', '', text)

    # Minúsculas
    text = text.lower()

    # Eliminar signos de puntuación y caracteres especiales
    # Conserva letras (incluyendo tildes y ñ), números y espacios
    text = re.sub(r"[^a-záéíóúüñà-ÿ0-9\s]", " ", text)

    # Normalizar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    return text


def lemmatize_and_filter(text: str, nlp) -> str:
    """
    Lematiza el texto y elimina stopwords usando spaCy.

    Args:
        text: texto normalizado
        nlp:  modelo spaCy cargado

    Returns:
        texto lematizado sin stopwords
    """
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop        # eliminar stopwords
        and not token.is_punct      # eliminar puntuación
        and not token.is_space      # eliminar espacios
        and len(token.lemma_) > 1   # eliminar tokens de 1 caracter
    ]

    return " ".join(tokens)


def clean_text(text: str, nlp) -> str:
    """
    Pipeline completo de limpieza para un texto.

    Args:
        text: texto crudo
        nlp:  modelo spaCy

    Returns:
        texto limpio y lematizado
    """
    text = normalize_text(text)
    text = lemmatize_and_filter(text, nlp)
    return text


# ─────────────────────────────────────────────────────────
#  Pipeline de preprocesamiento del corpus
# ─────────────────────────────────────────────────────────
def preprocess_corpus() -> list[dict]:
    """
    Lee intents_raw.jsonl, limpia cada texto y guarda
    el resultado en intents_clean.jsonl.

    Returns:
        Lista de dicts con texto limpio y tokens
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró: {RAW_DATA_PATH}\n"
            "Ejecuta primero: uv run generate-data"
        )

    nlp = load_spacy_model()

    raw_records = []
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    logger.info(f"Registros crudos cargados: {len(raw_records)}")

    clean_records = []
    skipped = 0

    for i, record in enumerate(raw_records):
        raw_text = record["text"]
        intent   = record["intent"]

        clean  = clean_text(raw_text, nlp)
        tokens = clean.split()

        if len(tokens) < 2:
            skipped += 1
            logger.debug(f"Texto descartado (muy corto): '{raw_text}' → '{clean}'")
            continue

        clean_records.append({
            "text":     clean,      # texto lematizado
            "text_raw": raw_text,   # texto original
            "tokens":   tokens,     # lista de tokens
            "n_tokens": len(tokens),
            "intent":   intent,
        })

        if (i + 1) % 100 == 0:
            logger.info(f"  Procesados {i + 1}/{len(raw_records)} registros...")

    logger.info(f"\n✓ Preprocesamiento completado:")
    logger.info(f"  Total procesados : {len(clean_records)}")
    logger.info(f"  Descartados      : {skipped} (texto muy corto tras limpieza)")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        for record in clean_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"✓ Corpus limpio guardado en: {CLEAN_DATA_PATH}")

    return clean_records


# ─────────────────────────────────────────────────────────
#  Ejemplo visual del efecto de la limpieza
# ─────────────────────────────────────────────────────────
def show_cleaning_examples(raw_records: list[dict], clean_records: list[dict], n: int = 5) -> None:
    """Muestra n ejemplos comparando texto crudo vs limpio."""
    print("\n" + "=" * 60)
    print("EJEMPLOS DE LIMPIEZA")
    print("=" * 60)

    for raw, clean in zip(raw_records[:n], clean_records[:n]):
        print(f"\nIntent  : {raw['intent']}")
        print(f"  Crudo : {raw['text']}")
        print(f"  Limpio: {clean['text']}")
        print(f"  Tokens: {clean['tokens']}")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────
def main():
    logger.info("Iniciando preprocesamiento del corpus...")

    clean_records = preprocess_corpus()

    raw_records = []
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    show_cleaning_examples(raw_records, clean_records, n=5)

    logger.info("✓ Preprocesamiento finalizado")


if __name__ == "__main__":
    main()
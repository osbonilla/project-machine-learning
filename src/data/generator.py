"""
generator.py — Generación de corpus sintético con GROQ API
Genera utterances etiquetados por intent usando un LLM.

Uso:
    uv run generate-data
    python -m src.data.generator
"""

import json
import time
import logging
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

# Importar configuración central
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    INTENTS,
    UTTERANCES_PER_INTENT,
    UTTERANCES_PER_BATCH,
    GENERATION_TEMPERATURE,
    RAW_DIR,
    RAW_DATA_PATH,
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
#  Prompt de generación
# ─────────────────────────────────────────────────────────
def build_prompt(intent_key: str, intent_info: dict, n: int) -> str:
    """
    Construye el prompt para pedirle a GROQ que genere
    utterances para un intent específico.
    """
    examples_str = "\n".join(f"  - {ex}" for ex in intent_info["examples"])

    return f"""Eres un experto en generación de datos de entrenamiento para sistemas NLP.

Tu tarea es generar exactamente {n} frases en ESPAÑOL que un usuario diría
para expresar la siguiente intención geoespacial:

INTENCIÓN: {intent_key}
DESCRIPCIÓN: {intent_info["description"]}

EJEMPLOS DE REFERENCIA (no los repitas, úsalos solo como guía de estilo):
{examples_str}

REGLAS IMPORTANTES:
1. Genera exactamente {n} frases, una por línea
2. Varía el vocabulario, estructura y longitud de las frases
3. Usa lenguaje natural y cotidiano en español
4. Algunas pueden ser cortas (3-5 palabras), otras más largas
5. NO numeres las frases
6. NO incluyas explicaciones ni comentarios
7. Solo devuelve las frases, una por línea

Genera las {n} frases ahora:"""


# ─────────────────────────────────────────────────────────
#  Generador principal
# ─────────────────────────────────────────────────────────
class IntentCorpusGenerator:
    """
    Genera un corpus de utterances etiquetados por intent
    usando la API de GROQ.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY no encontrada. "
                "Asegúrate de tenerla en tu archivo .env"
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"Cliente GROQ inicializado con modelo: {GROQ_MODEL}")

    def generate_batch(self, intent_key: str, intent_info: dict, n: int) -> list[str]:
        """
        Llama a GROQ y devuelve una lista de utterances para un intent.

        Args:
            intent_key:  nombre del intent (ej: "query_layer")
            intent_info: dict con description y examples
            n:           número de utterances a generar

        Returns:
            Lista de strings con los utterances generados
        """
        prompt = build_prompt(intent_key, intent_info, n)

        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GENERATION_TEMPERATURE,
                max_tokens=2048,
            )

            raw_text = response.choices[0].message.content.strip()

            # Parsear líneas y limpiar
            utterances = [
                line.strip()
                for line in raw_text.split("\n")
                if line.strip() and len(line.strip()) > 3
            ]

            logger.info(f"  [{intent_key}] generados {len(utterances)}/{n} utterances")
            return utterances

        except Exception as e:
            logger.error(f"  [{intent_key}] Error en GROQ API: {e}")
            return []

    def generate_all(self) -> list[dict]:
        """
        Itera sobre todos los intents y genera el corpus completo.

        Returns:
            Lista de dicts {"text": ..., "intent": ...}
        """
        corpus = []
        total_intents = len(INTENTS)

        logger.info(f"Iniciando generación: {total_intents} intents × {UTTERANCES_PER_INTENT} utterances")
        logger.info(f"Total esperado: {total_intents * UTTERANCES_PER_INTENT} ejemplos\n")

        for idx, (intent_key, intent_info) in enumerate(INTENTS.items(), 1):
            logger.info(f"[{idx}/{total_intents}] Generando intent: {intent_key}")

            intent_utterances = []
            batches_needed = UTTERANCES_PER_INTENT // UTTERANCES_PER_BATCH

            for batch_num in range(batches_needed):
                batch = self.generate_batch(intent_key, intent_info, UTTERANCES_PER_BATCH)
                intent_utterances.extend(batch)

                # Pausa entre llamadas para respetar rate limits de GROQ
                if batch_num < batches_needed - 1:
                    time.sleep(1)

            # Limitar exactamente a UTTERANCES_PER_INTENT
            intent_utterances = intent_utterances[:UTTERANCES_PER_INTENT]

            # Agregar al corpus con su label
            for text in intent_utterances:
                corpus.append({
                    "text": text,
                    "intent": intent_key,
                })

            logger.info(f"  ✓ {len(intent_utterances)} utterances agregados para '{intent_key}'\n")

        return corpus

    def save(self, corpus: list[dict]) -> Path:
        """
        Guarda el corpus en formato JSONL (una línea = un ejemplo).

        Args:
            corpus: lista de dicts {"text": ..., "intent": ...}

        Returns:
            Path del archivo guardado
        """
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
            for record in corpus:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"✓ Corpus guardado en: {RAW_DATA_PATH}")
        logger.info(f"  Total de ejemplos: {len(corpus)}")
        return RAW_DATA_PATH


# ─────────────────────────────────────────────────────────
#  Estadísticas del corpus generado
# ─────────────────────────────────────────────────────────
def print_stats(corpus: list[dict]) -> None:
    """Imprime estadísticas básicas del corpus generado."""
    from collections import Counter

    intent_counts = Counter(r["intent"] for r in corpus)
    avg_len = sum(len(r["text"].split()) for r in corpus) / len(corpus)

    print("\n" + "=" * 50)
    print("ESTADÍSTICAS DEL CORPUS GENERADO")
    print("=" * 50)
    print(f"Total de ejemplos : {len(corpus)}")
    print(f"Longitud promedio : {avg_len:.1f} palabras")
    print(f"Número de intents : {len(intent_counts)}")
    print("\nDistribución por intent:")
    for intent, count in sorted(intent_counts.items()):
        bar = "█" * (count // 4)
        print(f"  {intent:<20} {count:>4}  {bar}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────
def main():
    load_dotenv()

    logger.info("=" * 50)
    logger.info("GEO-INTENT CORPUS GENERATOR")
    logger.info("=" * 50)

    generator = IntentCorpusGenerator()

    # Generar corpus completo
    corpus = generator.generate_all()

    if not corpus:
        logger.error("No se generaron datos. Revisa tu GROQ_API_KEY.")
        return

    # Guardar
    generator.save(corpus)

    # Mostrar estadísticas
    print_stats(corpus)

    logger.info("✓ Generación completada exitosamente")


if __name__ == "__main__":
    main()
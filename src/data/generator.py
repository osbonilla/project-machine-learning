"""
generator.py — Generación de corpus sintético con GROQ API + Ollama (Mistral local)
Genera utterances etiquetados por intent usando dos LLMs para mayor diversidad.

Fuentes:
    - GROQ API (LLaMA 3.3 70B)  → utterances en la nube
    - Ollama / Mistral local      → utterances adicionales locales

Uso:
    uv run generate-data
    python -m src.data.generator
"""

import json
import time
import logging
import requests
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    INTENTS,
    UTTERANCES_PER_BATCH,
    GENERATION_TEMPERATURE,
    GROQ_UTTERANCES_PER_INTENT,
    OLLAMA_UTTERANCES_PER_INTENT,
    OLLAMA_URL,
    OLLAMA_MODEL,
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
def build_prompt(intent_key: str, intent_info: dict, n: int, style: str = "variado") -> str:
    """
    Construye el prompt para generar utterances.

    Args:
        intent_key : nombre del intent
        intent_info: dict con description y examples
        n          : número de utterances a generar
        style      : "variado" | "corto" | "largo" para diversificar
    """
    examples_str = "\n".join(f"  - {ex}" for ex in intent_info["examples"])

    style_instruction = {
        "variado": "Mezcla frases cortas (3-5 palabras) y largas (8-15 palabras).",
        "corto":   "Genera frases MUY CORTAS y directas (2-5 palabras máximo).",
        "largo":   "Genera frases LARGAS con contexto completo (10-15 palabras).",
    }.get(style, "Varía la longitud de las frases.")

    return f"""Eres un experto en generación de datos de entrenamiento para sistemas NLP geoespacial.

Tu tarea es generar exactamente {n} frases en ESPAÑOL que un usuario diría
para expresar la siguiente intención en un sistema GIS:

INTENCIÓN: {intent_key}
DESCRIPCIÓN: {intent_info["description"]}

EJEMPLOS DE REFERENCIA (no los repitas):
{examples_str}

INSTRUCCIONES:
1. Genera exactamente {n} frases, una por línea
2. {style_instruction}
3. Usa vocabulario técnico GIS Y lenguaje cotidiano
4. Incluye sinónimos y variaciones: "mostrar", "ver", "listar", "visualizar"
5. Algunas frases pueden tener contexto adicional: "en mi proyecto", "de esta zona"
6. NO numeres las frases
7. NO incluyas explicaciones ni comentarios
8. Solo devuelve las frases, una por línea

Genera las {n} frases ahora:"""


# ─────────────────────────────────────────────────────────
#  Generador GROQ
# ─────────────────────────────────────────────────────────
class GroqGenerator:
    """Genera utterances usando la API de GROQ (LLaMA 3.3 70B)."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY no encontrada en .env")
        self.client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"✓ GROQ inicializado: {GROQ_MODEL}")

    def generate_batch(
        self,
        intent_key: str,
        intent_info: dict,
        n: int,
        style: str = "variado",
    ) -> list[str]:
        """Genera n utterances para un intent con GROQ."""
        prompt = build_prompt(intent_key, intent_info, n, style)
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GENERATION_TEMPERATURE,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()
            utterances = [
                line.strip()
                for line in raw.split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            return utterances
        except Exception as e:
            logger.error(f"  [GROQ] Error: {e}")
            return []

    def generate_intent(
        self,
        intent_key: str,
        intent_info: dict,
        total: int,
    ) -> list[str]:
        """Genera total utterances para un intent en batches."""
        utterances = []
        batch_size = UTTERANCES_PER_BATCH
        batches    = total // batch_size
        styles     = ["variado", "corto", "largo", "variado"]  # rotar estilos

        for b in range(batches):
            style = styles[b % len(styles)]
            batch = self.generate_batch(intent_key, intent_info, batch_size, style)
            utterances.extend(batch)
            logger.info(f"  [GROQ] batch {b+1}/{batches} ({style}): {len(batch)} utterances")
            if b < batches - 1:
                time.sleep(1.5)  # respetar rate limit

        return utterances[:total]


# ─────────────────────────────────────────────────────────
#  Generador Ollama (Mistral local)
# ─────────────────────────────────────────────────────────

class OllamaGenerator:
    """Genera utterances usando Mistral local via Ollama."""

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.url   = url
        self.model = model
        self._check_connection()

    def _check_connection(self):
        """Verifica que Ollama esté corriendo."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models   = [m["name"] for m in response.json().get("models", [])]
            if any(self.model in m for m in models):
                logger.info(f"✓ Ollama inicializado: {self.model}")
            else:
                logger.warning(f"⚠ Modelo '{self.model}' no encontrado en Ollama. Modelos disponibles: {models}")
        except Exception as e:
            logger.error(f"✗ Ollama no disponible: {e}")
            raise ConnectionError(
                "Ollama no está corriendo. "
                "Inicia con: ollama serve"
            )

    def generate_batch(
        self,
        intent_key: str,
        intent_info: dict,
        n: int,
        style: str = "variado",
    ) -> list[str]:
        """Genera n utterances para un intent con Mistral."""
        prompt = build_prompt(intent_key, intent_info, n, style)
        try:
            response = requests.post(
                self.url,
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": GENERATION_TEMPERATURE,
                        "num_predict": 1024,
                    },
                },
                timeout=500,
            )
            raw = response.json().get("response", "").strip()
            utterances = [
                line.strip()
                for line in raw.split("\n")
                if line.strip() and len(line.strip()) > 3
                and not line.strip().startswith("#")  # filtrar comentarios
                and not line.strip().startswith("-")  # filtrar listas con guión
            ]
            return utterances
        except Exception as e:
            logger.error(f"  [Ollama] Error: {e}")
            return []

    def generate_intent(
        self,
        intent_key: str,
        intent_info: dict,
        total: int,
    ) -> list[str]:
        """Genera total utterances para un intent en batches."""
        utterances = []
        batch_size = 25
        batches    = max(1, total // batch_size)
        styles     = ["corto", "largo"]

        for b in range(batches):
            style = styles[b % len(styles)]
            batch = self.generate_batch(intent_key, intent_info, batch_size, style)
            utterances.extend(batch)
            logger.info(f"  [Ollama] batch {b+1}/{batches} ({style}): {len(batch)} utterances")
            time.sleep(0.5)

        return utterances[:total]


# ─────────────────────────────────────────────────────────
#  Pipeline principal de generación
# ─────────────────────────────────────────────────────────
def generate_corpus(
    use_groq: bool = True,
    use_ollama: bool = True,
) -> list[dict]:
    """
    Genera el corpus completo combinando GROQ y Ollama.

    Args:
        use_groq  : usar GROQ API
        use_ollama: usar Mistral local via Ollama

    Returns:
        Lista de dicts {"text": ..., "intent": ...}
    """
    corpus = []
    total_intents = len(INTENTS)

    # Inicializar generadores
    groq_gen   = GroqGenerator() if use_groq else None
    ollama_gen = None
    if use_ollama:
        try:
            ollama_gen = OllamaGenerator()
        except ConnectionError as e:
            logger.warning(f"Ollama no disponible, usando solo GROQ: {e}")

    logger.info(f"\nIniciando generación: {total_intents} intents")
    logger.info(f"  GROQ    : {GROQ_UTTERANCES_PER_INTENT} utterances/intent")
    if ollama_gen:
        logger.info(f"  Ollama  : {OLLAMA_UTTERANCES_PER_INTENT} utterances/intent")
    total_esperado = total_intents * (
        GROQ_UTTERANCES_PER_INTENT + (OLLAMA_UTTERANCES_PER_INTENT if ollama_gen else 0)
    )
    logger.info(f"  Total esperado: ~{total_esperado} ejemplos\n")

    for idx, (intent_key, intent_info) in enumerate(INTENTS.items(), 1):
        logger.info(f"[{idx}/{total_intents}] Generando intent: {intent_key}")
        intent_utterances = []

        # ── GROQ ──────────────────────────────────────────
        if groq_gen:
            groq_utterances = groq_gen.generate_intent(
                intent_key, intent_info, GROQ_UTTERANCES_PER_INTENT
            )
            intent_utterances.extend(groq_utterances)
            logger.info(f"  GROQ total: {len(groq_utterances)} utterances")

        # ── Ollama ────────────────────────────────────────
        if ollama_gen:
            ollama_utterances = ollama_gen.generate_intent(
                intent_key, intent_info, OLLAMA_UTTERANCES_PER_INTENT
            )
            intent_utterances.extend(ollama_utterances)
            logger.info(f"  Ollama total: {len(ollama_utterances)} utterances")

        # Eliminar duplicados manteniendo orden
        seen = set()
        unique_utterances = []
        for text in intent_utterances:
            text_lower = text.lower().strip()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_utterances.append(text)

        # Agregar al corpus
        for text in unique_utterances:
            corpus.append({"text": text, "intent": intent_key})

        logger.info(f"  ✓ {len(unique_utterances)} utterances únicos para '{intent_key}'\n")

    return corpus


# ─────────────────────────────────────────────────────────
#  Estadísticas
# ─────────────────────────────────────────────────────────
def print_stats(corpus: list[dict]) -> None:
    """Imprime estadísticas del corpus generado."""
    from collections import Counter
    intent_counts = Counter(r["intent"] for r in corpus)
    avg_len = sum(len(r["text"].split()) for r in corpus) / len(corpus)

    print("\n" + "=" * 55)
    print("ESTADÍSTICAS DEL CORPUS GENERADO")
    print("=" * 55)
    print(f"Total de ejemplos : {len(corpus)}")
    print(f"Longitud promedio : {avg_len:.1f} palabras")
    print(f"Número de intents : {len(intent_counts)}")
    print("\nDistribución por intent:")
    for intent, count in sorted(intent_counts.items()):
        bar = "█" * (count // 5)
        print(f"  {intent:<22} {count:>5}  {bar}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────
#  Guardado
# ─────────────────────────────────────────────────────────
def save_corpus(corpus: list[dict]) -> Path:
    """Guarda el corpus en formato JSONL."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
        for record in corpus:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"✓ Corpus guardado en: {RAW_DATA_PATH}")
    logger.info(f"  Total: {len(corpus)} ejemplos")
    return RAW_DATA_PATH


# ─────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────
def main():
    load_dotenv()
    logger.info("=" * 55)
    logger.info("GEO-INTENT CORPUS GENERATOR")
    logger.info("Fuentes: GROQ API + Mistral (Ollama local)")
    logger.info("=" * 55)

    corpus = generate_corpus(use_groq=True, use_ollama=True)

    if not corpus:
        logger.error("No se generaron datos.")
        return

    save_corpus(corpus)
    print_stats(corpus)
    logger.info("✓ Generación completada")


if __name__ == "__main__":
    main()
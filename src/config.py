"""
config.py — Configuración central del proyecto
Todos los paths, constantes e intents se definen aquí.
Nunca hardcodear paths en otros módulos, siempre importar desde aquí.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ─────────────────────────────────────────────────────────
#  Entorno
# ─────────────────────────────────────────────────────────
load_dotenv()

ENV         = os.getenv("ENV", "development")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "info")

# ─────────────────────────────────────────────────────────
#  GROQ
# ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# ─────────────────────────────────────────────────────────
#  Paths del proyecto
# ─────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent  # raíz del proyecto

DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR  = DATA_DIR / "external"

MODELS_DIR    = ROOT_DIR / "models"
REPORTS_DIR   = ROOT_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"

# ─────────────────────────────────────────────────────────
#  Archivos de datos
# ─────────────────────────────────────────────────────────
RAW_DATA_PATH     = RAW_DIR / "intents_raw.jsonl"
CLEAN_DATA_PATH   = INTERIM_DIR / "intents_clean.jsonl"

PROCESSED_FILES = {
    "X_train": PROCESSED_DIR / "X_train.pkl",
    "X_test":  PROCESSED_DIR / "X_test.pkl",
    "y_train": PROCESSED_DIR / "y_train.pkl",
    "y_test":  PROCESSED_DIR / "y_test.pkl",
}

# ─────────────────────────────────────────────────────────
#  Archivos de modelos
# ─────────────────────────────────────────────────────────
MODEL01_PATH        = MODELS_DIR / "model01.pkl"
MODEL02_PATH        = MODELS_DIR / "model02.pkl"
LABEL_ENCODER_PATH  = MODELS_DIR / "label_encoder.pkl"

ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "model01")

MODEL_PATHS = {
    "model01": MODEL01_PATH,
    "model02": MODEL02_PATH,
}

# ─────────────────────────────────────────────────────────
#  API
# ─────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# ─────────────────────────────────────────────────────────
#  Intents geoespaciales
#  Cada intent tiene:
#   - label     : nombre de la clase (lo que predice el modelo)
#   - description: descripción para generar utterances con GROQ
#   - examples  : 3 ejemplos semilla para guiar al LLM
# ─────────────────────────────────────────────────────────
INTENTS = {
    "query_layer": {
        "label": "query_layer",
        "description": "El usuario quiere consultar, ver o listar capas geoespaciales disponibles",
        "examples": [
            "muéstrame las capas disponibles",
            "qué capas hay en el mapa",
            "quiero ver la capa de uso de suelo",
        ],
    },
    "spatial_filter": {
        "label": "spatial_filter",
        "description": "El usuario quiere filtrar, seleccionar o recortar features por ubicación o geometría",
        "examples": [
            "filtra los polígonos dentro del área de estudio",
            "selecciona los puntos que están dentro de la provincia",
            "recorta la capa por el límite municipal",
        ],
    },
    "calculate_area": {
        "label": "calculate_area",
        "description": "El usuario quiere calcular áreas, perímetros o medidas geométricas",
        "examples": [
            "calcula el área total de los parques",
            "cuántos metros cuadrados tiene este polígono",
            "dame el perímetro de la zona de estudio",
        ],
    },
    "get_attributes": {
        "label": "get_attributes",
        "description": "El usuario quiere consultar los atributos, campos o propiedades de una capa o feature",
        "examples": [
            "qué atributos tiene la capa de ríos",
            "muéstrame los campos de la tabla de parcelas",
            "cuáles son las propiedades de este polígono",
        ],
    },
    "export_data": {
        "label": "export_data",
        "description": "El usuario quiere exportar, descargar o guardar datos en algún formato",
        "examples": [
            "exporta los resultados a shapefile",
            "descarga la capa en formato GeoJSON",
            "guarda el mapa como archivo KML",
        ],
    },
    "visualize_map": {
        "label": "visualize_map",
        "description": "El usuario quiere visualizar, renderizar o cambiar la simbología del mapa",
        "examples": [
            "muestra el mapa con simbología por categoría",
            "cambia el color de la capa de vegetación",
            "visualiza las zonas de riesgo en el mapa",
        ],
    },
    "spatial_join": {
        "label": "spatial_join",
        "description": "El usuario quiere unir o cruzar dos capas espacialmente",
        "examples": [
            "une la capa de puntos con la de polígonos",
            "cruza los datos de población con los límites provinciales",
            "combina las capas por intersección espacial",
        ],
    },
    "buffer_analysis": {
        "label": "buffer_analysis",
        "description": "El usuario quiere crear zonas de influencia o buffers alrededor de features",
        "examples": [
            "crea un buffer de 500 metros alrededor de los ríos",
            "genera una zona de influencia de 1 km de las escuelas",
            "aplica un buffer a los puntos de interés",
        ],
    },
}

# Lista plana de labels (útil para el LabelEncoder)
INTENT_LABELS = list(INTENTS.keys())
NUM_CLASSES    = len(INTENT_LABELS)

# ─────────────────────────────────────────────────────────
#  Parámetros de generación de datos
# ─────────────────────────────────────────────────────────
UTTERANCES_PER_INTENT = 80   # utterances a generar por cada intent
UTTERANCES_PER_BATCH  = 20   # cuántos pedir a GROQ por llamada (evita timeouts)
GENERATION_TEMPERATURE = 0.85

# ─────────────────────────────────────────────────────────
#  Parámetros de entrenamiento
# ─────────────────────────────────────────────────────────
TEST_SIZE         = 0.2
RANDOM_STATE      = 42
CV_N_SPLITS       = 10   # folds para k-fold
CV_N_REPEATS      = 10   # repeticiones para repeated k-fold
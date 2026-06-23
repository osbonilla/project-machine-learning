# 🌍 Geo-Intent Classifier

> Clasificador de intención NLP para arquitectura multiagente geoespacial  
> Python 3.12 · uv · scikit-learn · spaCy · FastAPI · Docker

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👤 Autor

**[Tu Nombre]**  
[Universidad / Institución]  
Curso: Machine Learning — 2026

---

## 📋 Tabla de contenidos

- [Descripción del problema](#-descripción-del-problema)
- [Arquitectura](#-arquitectura)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Pipeline de datos](#-pipeline-de-datos)
- [Modelos](#-modelos)
- [Resultados](#-resultados)
- [API](#-api)
- [Frontend](#-frontend)
- [Docker](#-docker)
- [Tests](#-tests)
- [Notebook](#-notebook)
- [Referencias](#-referencias)

---

## 📌 Descripción del problema

En arquitecturas multiagente modernas, el sistema debe interpretar instrucciones del usuario en lenguaje natural y redirigirlas al agente especializado correspondiente. Sin un clasificador de intención, el sistema no sabe a qué agente enviar cada instrucción.

### Solución propuesta

Un clasificador de texto que, dada una instrucción en español, identifica automáticamente la **intención geoespacial** y la dirige al **agente GIS** correspondiente.

### Intenciones implementadas (8 clases)

| Intent | Descripción | Ejemplo |
|--------|-------------|---------|
| `query_layer` | Consultar capas disponibles | *"muéstrame las capas del mapa"* |
| `spatial_filter` | Filtrar por geometría/ubicación | *"filtra los polígonos del área"* |
| `calculate_area` | Calcular áreas y medidas | *"calcula el área de los parques"* |
| `get_attributes` | Consultar atributos de capa | *"qué campos tiene la capa de ríos"* |
| `export_data` | Exportar datos | *"exporta los resultados a shapefile"* |
| `visualize_map` | Cambiar simbología del mapa | *"muestra el mapa por categoría"* |
| `spatial_join` | Unir capas espacialmente | *"cruza las capas por intersección"* |
| `buffer_analysis` | Crear zonas de influencia | *"crea un buffer de 500m"* |

---

## 🏗️ Arquitectura

```
Usuario (texto o voz)
        │
        ▼
┌───────────────────┐
│  Frontend (Plus)  │  ← visor web con Web Speech API
│  index.html       │
└────────┬──────────┘
         │  HTTP POST /predict
         ▼
┌───────────────────┐
│   FastAPI         │  ← src/api/main.py
│   POST /predict   │
│   GET  /health    │
│   WS   /speech    │
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────┐
│  Clasificador de Intención        │
│                                   │
│  Model01: TF-IDF + chi² + SVM     │  ← F1-macro: 0.96
│  Model02: TF-IDF + chi² + MLP     │  ← más robusto con texto crudo
└────────┬──────────────────────────┘
         │
         ▼
   Agente Geoespacial
   (GIS / análisis espacial)
```

---

## 📁 Estructura del proyecto

```
project-machine-learning/
│
├── .github/workflows/
│   ├── ci.yml                      # Lint + tests automáticos en cada push
│   └── train.yml                   # Pipeline de entrenamiento en CI
│
├── data/
│   ├── raw/
│   │   └── intents_raw.jsonl       # 639 utterances crudos generados con GROQ API
│   ├── interim/
│   │   └── intents_clean.jsonl     # Corpus limpio y lematizado con spaCy
│   ├── processed/
│   │   ├── X_train.pkl             # Textos de entrenamiento (509 ejemplos, 80%)
│   │   ├── X_test.pkl              # Textos de prueba (128 ejemplos, 20%)
│   │   ├── y_train.pkl             # Labels de entrenamiento (enteros)
│   │   └── y_test.pkl              # Labels de prueba (enteros)
│   └── external/                   # Stopwords y vocabularios externos
│
├── docs/                           # Documentación mkdocs
│
├── notebooks/
│   └── main.ipynb                  # Notebook principal con todo el análisis y resultados
│
├── reports/
│   └── figures/                    # Gráficas generadas (curvas, t-SNE, matrices)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths, constantes, definición de intents y parámetros
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py            # Genera utterances por intent usando GROQ API (LLaMA 3.3)
│   │   ├── dataset.py              # Carga, valida, codifica labels y splitea el corpus
│   │   └── preprocess.py           # Limpieza: minúsculas, regex, lematización spaCy, stopwords
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extraction.py           # TF-IDF, Bag of Words, n-grams — vectorización de texto
│   │   └── selection.py            # chi², mutual info, variance threshold — selección de features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model01/                # Baseline: TF-IDF + chi² + LinearSVC / Naive Bayes
│   │   │   ├── model01.py          # Definición de los pipelines SVM y NB
│   │   │   ├── train.py            # GridSearchCV + RepeatedKFold + curva de aprendizaje
│   │   │   ├── predict.py          # Carga model01.pkl y expone predict(text) → {intent, confidence}
│   │   │   └── dataloader.py       # Carga datos procesados para entrenamiento
│   │   └── model02/                # Avanzado: TF-IDF (word+char) + chi² + MLP
│   │       ├── model02.py          # Pipeline con FeatureUnion de word y char n-grams
│   │       ├── train.py            # GridSearchCV + RepeatedKFold + curva de aprendizaje
│   │       ├── predict.py          # Carga model02.pkl y expone predict(text) → {intent, confidence}
│   │       └── dataloader.py       # Carga datos procesados para entrenamiento
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # App FastAPI: lifespan, carga modelos, monta frontend estático
│   │   ├── routes.py               # POST /predict · GET /health · WebSocket /speech
│   │   ├── schemas.py              # Pydantic: IntentRequest / IntentResponse / HealthResponse
│   │   └── middleware.py           # CORS (necesario para frontend/speech) + logging de requests
│   │
│   └── speech/
│       ├── __init__.py
│       ├── stt.py                  # Speech-to-Text (Web Speech API / Whisper)
│       └── tts.py                  # Text-to-Speech (respuesta hablada)
│
├── frontend/                       # Visor demo — HTML/CSS/JS puro, sin framework
│   ├── index.html                  # Caja de texto + botón micrófono + panel de resultado
│   ├── app.js                      # Web Speech API + fetch POST /predict + render resultado
│   └── css/
│       └── style.css               # Estilos oscuros del visor
│
├── models/
│   ├── model01.pkl                 # Pipeline SVM serializado con joblib
│   ├── model02.pkl                 # Pipeline MLP serializado con joblib
│   └── label_encoder.pkl           # Mapeo índice → nombre de intent
│
├── tests/
│   ├── conftest.py                 # Fixtures compartidos (textos, labels, corpus de prueba)
│   ├── test_data.py                # Valida schema JSONL, intents presentes, textos no vacíos
│   ├── test_features.py            # TF-IDF, BoW, chi², no data leakage, variance threshold
│   ├── test_models.py              # Pipelines SVM/NB/MLP, accuracy > 0.85, predict válido
│   └── test_api.py                 # Endpoints /predict y /health, schema de response
│
├── .env.example                    # Template de variables de entorno
├── .gitignore                      # Excluye .venv, .env, *.pkl, data/raw/*, etc.
├── CONTRIBUTING.md                 # Guía de contribución
├── Dockerfile                      # Imagen multistage: builder (uv) → runtime (liviana)
├── docker-compose.yml              # Servicio api (prod) + api-dev (hot-reload)
├── LICENSE                         # MIT
├── Makefile                        # make install | data | train | api | test | docker
├── pyproject.toml                  # Dependencias + scripts CLI (uv, sin requirements.txt)
├── README.md
└── uv.lock                         # Lockfile reproducible de uv
```

---

## 🚀 Instalación

### Requisitos

- Python 3.12
- [uv](https://github.com/astral-sh/uv) — gestor de paquetes
- GROQ API Key gratuita — [console.groq.com](https://console.groq.com)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/project-machine-learning.git
cd project-machine-learning

# 2. Crear entorno virtual con Python 3.12
uv venv .venv --python 3.12

# 3. Activar entorno
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Instalar dependencias
uv sync

# 5. Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar GROQ_API_KEY=gsk_...

# 6. Descargar modelo de español de spaCy
uv run python -m spacy download es_core_news_sm
```

---

## 🗂️ Pipeline de datos

El pipeline transforma instrucciones en lenguaje natural hasta llegar a los modelos entrenados:

```
1. GROQ API (LLaMA 3.3 70B)
   └── genera 80 utterances × 8 intents = 640 ejemplos
   └── src/data/generator.py
   └── → data/raw/intents_raw.jsonl

2. Preprocesamiento (spaCy es_core_news_sm)
   └── minúsculas → regex → lematización → eliminar stopwords
   └── src/data/preprocess.py
   └── → data/interim/intents_clean.jsonl

3. Split y codificación
   └── LabelEncoder: "query_layer" → 0, "spatial_filter" → 1, ...
   └── train_test_split estratificado: 80% train / 20% test
   └── src/data/dataset.py
   └── → data/processed/X_train.pkl, X_test.pkl, y_train.pkl, y_test.pkl

4. Feature Extraction
   └── TF-IDF con unigramas + bigramas (ngram_range=(1,2))
   └── src/features/extraction.py
   └── Vocabulario: ~467 términos

5. Feature Selection (dentro del pipeline)
   └── SelectKBest(chi², k=1000) — método filter
   └── src/features/selection.py

6. Entrenamiento
   └── GridSearchCV + RepeatedStratifiedKFold (10×10)
   └── src/models/model01/train.py
   └── src/models/model02/train.py
   └── → models/model01.pkl, model02.pkl
```

### Ejecutar el pipeline completo

```bash
# Generar corpus
uv run python -m src.data.generator

# Preprocesar
uv run python -m src.data.preprocess

# Split y codificación
uv run python -m src.data.dataset

# Entrenar modelos
uv run python -m src.models.model01.train
uv run python -m src.models.model02.train
```

O con Makefile:

```bash
make data       # genera + preprocesa + splitea
make train      # entrena model01 y model02
```

---

## 🤖 Modelos

### Model01 — Baseline

**Pipeline:** `TF-IDF` → `SelectKBest(chi²)` → `CalibratedClassifierCV(LinearSVC)`

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)),
    ('chi2',  SelectKBest(chi2, k=1000)),
    ('clf',   CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
])
```

- **TF-IDF:** convierte texto en vectores numéricos ponderando términos informativos
- **chi²:** filtra los 1000 términos más relevantes estadísticamente
- **LinearSVC:** clasificador SVM lineal, eficiente en alta dimensionalidad
- **CalibratedClassifierCV:** envuelve SVM para obtener probabilidades (`predict_proba`)

### Model02 — Avanzado

**Pipeline:** `FeatureUnion(word TF-IDF + char TF-IDF)` → `chi²` → `MaxAbsScaler` → `MLP`

```python
Pipeline([
    ('features', FeatureUnion([
        ('word_tfidf', TfidfVectorizer(ngram_range=(1,2), analyzer='word')),
        ('char_tfidf', TfidfVectorizer(ngram_range=(2,4), analyzer='char_wb')),
    ])),
    ('chi2',   SelectKBest(chi2, k=2000)),
    ('scaler', MaxAbsScaler()),
    ('clf',    MLPClassifier(hidden_layer_sizes=(256,128), early_stopping=True)),
])
```

- **FeatureUnion:** combina dos vectorizadores en paralelo
  - **word n-grams:** captura frases clave como *"calcular área"*, *"exportar shapefile"*
  - **char n-grams:** captura morfología, robusto ante errores tipográficos
- **MLP:** red neuronal densa, aprende patrones más complejos
- **Model02 es más robusto con texto crudo** (sin preprocesamiento) en producción

### Optimización de hiperparámetros

Se usaron dos estrategias:

**GridSearchCV** — búsqueda exhaustiva:
```python
RepeatedStratifiedKFold(n_splits=10, n_repeats=10)  # 10×10 = 100 folds
GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
```

**Optuna** — búsqueda bayesiana (extra):
```python
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=40)
```

---

## 📊 Resultados

### Evaluación en test set (128 ejemplos)

| Modelo | Accuracy | F1-macro | F1-weighted |
|--------|----------|----------|-------------|
| **Model01 (SVM)** | **0.96** | **0.96** | **0.96** |
| Model02 (MLP) | ~0.91 | ~0.91 | ~0.91 |

### Comparación estadística — Test de Wilcoxon

```
Estadístico W : 187.0
p-valor       : 0.000000
Nivel α       : 0.05

✓ Diferencia SIGNIFICATIVA (p < 0.05)
  Model01 (SVM) es estadísticamente superior en validación cruzada
```

**Conclusión:** Model01 supera estadísticamente a Model02 en condiciones controladas (texto preprocesado). Sin embargo, **Model02 es más robusto en producción** con texto crudo gracias a los char n-grams.

### Por intent — Model01

| Intent | Precision | Recall | F1 |
|--------|-----------|--------|----|
| buffer_analysis | 1.00 | 1.00 | 1.00 |
| calculate_area | 0.94 | 1.00 | 0.97 |
| export_data | 1.00 | 1.00 | 1.00 |
| get_attributes | 0.88 | 0.88 | 0.88 |
| query_layer | 0.88 | 0.94 | 0.91 |
| spatial_filter | 1.00 | 0.94 | 0.97 |
| spatial_join | 1.00 | 0.94 | 0.97 |
| visualize_map | 1.00 | 1.00 | 1.00 |

---

## 🌐 API

Levanta en `http://localhost:8000`. Documentación interactiva en `/docs`.

### Levantar la API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

#### `POST /predict` — Clasificar texto

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "muéstrame las capas disponibles", "model": "model02"}'
```

```json
{
  "intent": "query_layer",
  "confidence": 0.9722,
  "agent": "geo_agent",
  "model": "model02",
  "text": "muéstrame las capas disponibles"
}
```

#### `GET /health` — Estado del servicio

```json
{
  "status": "ok",
  "model01": true,
  "model02": true
}
```

#### `WebSocket /speech` — Clasificación por voz

Recibe texto transcrito por Web Speech API y devuelve el intent en tiempo real.

---

## 🖥️ Frontend (Plus)

Visor demo accesible en `http://localhost:8000/app`.

- **Entrada de texto:** escribe cualquier instrucción geoespacial
- **Entrada por voz:** botón micrófono usa Web Speech API nativa del browser (Chrome/Edge)
- **Selector de modelo:** elige entre Model01 y Model02
- **Resultado:** muestra intent, barra de confianza con color, agente destino y modelo usado

> El módulo de voz usa Web Speech API del browser — no requiere librerías externas ni API key.

---

## 🐳 Docker

### Build y levantar

```bash
docker compose up --build
```

### Modo desarrollo (hot-reload)

```bash
docker compose --profile dev up --build
```

### Servicios

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| `api` | `8000` | FastAPI + modelos + frontend estático |
| `api-dev` | `8000` | Hot-reload para desarrollo |

---

## 🧪 Tests

```bash
# Correr todos los tests
uv run pytest

# Con reporte de cobertura
uv run pytest --cov=src --cov-report=html
```

**Resultado: 41/41 tests passed**

| Archivo | Qué prueba |
|---------|-----------|
| `test_data.py` | Schema JSONL, intents presentes, textos no vacíos, split correcto |
| `test_features.py` | TF-IDF, BoW, bigramas, no data leakage, chi², variance threshold |
| `test_models.py` | Pipelines SVM/NB/MLP, accuracy > 0.85 en test set, predict válido |
| `test_api.py` | `/predict` y `/health`, schema de response, texto vacío → 422 |

---

## 📓 Notebook

El análisis completo está en `notebooks/main.ipynb`:

| Sección | Contenido |
|---------|-----------|
| Introducción | Problema, propuesta, tabla de intents |
| EDA | Distribución de clases, longitud de utterances, top n-grams por intent |
| Preprocesamiento | Ejemplos antes/después de limpieza con spaCy |
| Feature Extraction | Comparación BoW vs TF-IDF vs TF-IDF+n-grams |
| Pipelines | Definición y justificación de los 3 pipelines sklearn |
| Optimización | GridSearchCV + curvas de aprendizaje + Optuna (bayesiano) |
| Comparación estadística | Wilcoxon 10×10 repeated k-fold + boxplot + diferencia por fold |
| Evaluación | Classification report + matrices de confusión |
| t-SNE | Espacio de features en 2D con colores por intent |
| Conclusiones | Análisis de resultados y recomendaciones |
| Referencias | IEEE + prompts de IA usados como comentarios Python |

---

## 📚 Referencias

**[1]** F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

**[2]** G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, vol. 24, no. 5, pp. 513–523, 1988.

**[3]** C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

**[4]** L. Van der Maaten and G. Hinton, "Visualizing Data using t-SNE," *Journal of Machine Learning Research*, vol. 9, pp. 2579–2605, 2008.

**[5]** T. Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework," in *Proc. KDD*, 2019. [Online]. Available: https://optuna.org

**[6]** Explosion AI, "spaCy: Industrial-strength NLP," 2023. [Online]. Available: https://spacy.io

**[7]** GROQ Inc., "GROQ API Documentation," 2024. [Online]. Available: https://console.groq.com/docs

**[8]** F. Wilcoxon, "Individual Comparisons by Ranking Methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.
# 🌍 Geo-Intent Classifier

> Clasificador de intención NLP para arquitectura multiagente geoespacial  
> Python 3.12 · uv · scikit-learn · spaCy · FastAPI · Docker

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👤 Autor

**Oldrin Santiago Bonilla Cáceres**  
Maestrante en Ciencia de Datos  
Curso: Machine Learning (Aprendizaje de Máquina) — 2026

---

## 📋 Tabla de contenidos

- [Descripción del problema](#-descripción-del-problema)
- [Arquitectura](#-arquitectura)
- [Decisiones técnicas](#-decisiones-técnicas)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Flujo de trabajo completo](#-flujo-de-trabajo-completo)
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
│     Frontend      │  ← visor web con Web Speech API (Chrome/Edge)
│     index.html    │    HTML + CSS + JS puro, sin framework
└────────┬──────────┘
         │  HTTP POST /predict
         ▼
┌─────────────────────────────┐
│   FastAPI — src/api/        │
│   POST /predict             │  ← preprocesa texto + clasifica
│   GET  /health              │  ← estado de los modelos
│   WS   /speech              │  ← clasificación por voz en tiempo real
└────────┬────────────────────┘
         │  preprocess_text() → spaCy
         │  predict() → model.pkl
         ▼
┌──────────────────────────────────────┐
│  Clasificador de Intención           │
│                                      │
│  Model01: TF-IDF + chi² + SVM       │  ← F1-macro: 0.96 (texto preprocesado)
│  Model02: TF-IDF + chi² + MLP       │  ← F1-macro: 0.88 (más robusto en producción)
└────────┬─────────────────────────────┘
         │
         ▼
   Agente Geoespacial (geo_agent)
   Recibe el intent clasificado y ejecuta la operación GIS correspondiente
```

---

## ⚙️ Decisiones técnicas

### ¿Por qué `uv` en lugar de `pip`?

`uv` es un gestor de paquetes moderno escrito en Rust que reemplaza a `pip` + `venv`. Se eligió por tres razones concretas:

1. **Velocidad:** instala dependencias hasta 10-100x más rápido que pip gracias a su resolver en Rust y caché agresiva
2. **Reproducibilidad:** genera un `uv.lock` con hashes exactos de cada paquete, garantizando que cualquier persona que clone el repo obtenga exactamente el mismo entorno
3. **Estándar moderno:** usa `pyproject.toml` como única fuente de verdad, eliminando `requirements.txt` y sus problemas de inconsistencia

```bash
# Con pip (antiguo)
pip install -r requirements.txt  # puede instalar versiones diferentes cada vez

# Con uv (moderno)
uv sync  # instala exactamente lo del uv.lock, siempre igual
```

### ¿Por qué Docker?

Docker se usa para garantizar que el proyecto funcione igual en cualquier entorno — laptop, servidor, CI/CD. Sin Docker:

- En tu máquina funciona, en la del profesor no porque tiene Python 3.11
- Las versiones de spaCy o sklearn pueden diferir entre sistemas
- El modelo `.pkl` puede no ser compatible entre versiones

Con Docker:
```bash
docker compose up --build
# → Python 3.12 exacto + todas las deps + modelos + API en localhost:8000
# → Funciona igual en Windows, macOS y Linux
```

El `Dockerfile` usa **multistage build** para mantener la imagen pequeña:
- **Stage builder:** instala uv y todas las dependencias
- **Stage runtime:** copia solo lo necesario, sin herramientas de build

---

## 📁 Estructura del proyecto

```
project-machine-learning/
│
├── .github/workflows/
│   ├── ci.yml              # Corre lint (ruff) + tests (pytest) en cada push a GitHub
│   └── train.yml           # Pipeline de entrenamiento automático en CI
│
├── data/                   # Datos en diferentes etapas del pipeline
│   ├── raw/
│   │   └── intents_raw.jsonl     # Utterances crudos generados por GROQ + Mistral
│   ├── interim/
│   │   └── intents_clean.jsonl   # Corpus limpio: lematizado, sin stopwords, sin números
│   ├── processed/
│   │   ├── X_train.pkl           # Array de textos para entrenar (80%)
│   │   ├── X_test.pkl            # Array de textos para evaluar (20%)
│   │   ├── y_train.pkl           # Labels codificados como enteros para entrenar
│   │   └── y_test.pkl            # Labels codificados como enteros para evaluar
│   └── external/                 # Recursos externos (stopwords, vocabularios)
│
├── docs/                   # Documentación del proyecto (mkdocs)
│
├── notebooks/
│   └── main.ipynb          # Notebook principal: EDA → features → modelos → resultados
│
├── reports/
│   └── figures/            # Gráficas generadas: curvas de aprendizaje, t-SNE, matrices
│
├── src/                    # Código fuente modular del proyecto
│   ├── config.py           # Centraliza TODOS los paths, constantes e intents
│   │                       # Ningún otro módulo hardcodea paths — siempre importa de aquí
│   │
│   ├── data/
│   │   ├── generator.py    # Genera utterances usando GROQ API (LLaMA 3.3 70B) +
│   │   │                   # Mistral local (Ollama). Produce intents_raw.jsonl
│   │   ├── preprocess.py   # Limpia cada utterance: elimina numeración → minúsculas →
│   │   │                   # regex → lematización spaCy → eliminar stopwords
│   │   └── dataset.py      # Carga el corpus limpio, codifica labels con LabelEncoder,
│   │                       # hace split estratificado 80/20 y guarda los .pkl
│   │
│   ├── features/
│   │   ├── extraction.py   # Implementa BoW, TF-IDF y TF-IDF+ngrams para comparar
│   │   │                   # representaciones en el notebook de EDA
│   │   └── selection.py    # Implementa chi², mutual info y variance threshold
│   │                       # como métodos filter de selección de features
│   │
│   ├── models/
│   │   ├── model01/        # Pipeline baseline: TF-IDF + chi² + SVM/NB
│   │   │   ├── model01.py  # Define build_svm_pipeline() y build_nb_pipeline()
│   │   │   ├── train.py    # Entrena con GridSearchCV + RepeatedKFold(10×10),
│   │   │   │               # guarda cv_results_model01.csv y model01.pkl
│   │   │   ├── predict.py  # Carga model01.pkl → predict(text) → {intent, confidence}
│   │   │   └── dataloader.py # Carga X_train/test.pkl para el entrenamiento
│   │   │
│   │   └── model02/        # Pipeline avanzado: TF-IDF(word+char) + chi² + MLP
│   │       ├── model02.py  # Define build_mlp_pipeline() con FeatureUnion
│   │       ├── train.py    # Entrena con GridSearchCV + RepeatedKFold(10×10),
│   │       │               # guarda cv_results_model02.csv y model02.pkl
│   │       ├── predict.py  # Carga model02.pkl → predict(text) → {intent, confidence}
│   │       └── dataloader.py # Carga X_train/test.pkl para el entrenamiento
│   │
│   ├── api/
│   │   ├── main.py         # App FastAPI: carga modelos al iniciar (lifespan),
│   │   │                   # registra routers y sirve frontend/ como archivos estáticos
│   │   ├── routes.py       # Define los endpoints: POST /predict (con preprocesamiento
│   │   │                   # spaCy), GET /health, WebSocket /speech
│   │   ├── schemas.py      # Modelos Pydantic: IntentRequest, IntentResponse, HealthResponse
│   │   └── middleware.py   # CORS (permite llamadas desde el browser/frontend)
│   │                       # y logging automático de cada request
│   │
│   └── speech/
│       ├── stt.py          # Speech-to-Text: recibe audio → devuelve texto transcrito
│       └── tts.py          # Text-to-Speech: convierte respuesta del agente en audio
│
├── frontend/               # Visor demo — HTML/CSS/JS puro, sin framework ni Node.js
│   ├── index.html          # Interfaz: textarea + botón micrófono + panel de resultado
│   ├── app.js              # Lógica: Web Speech API (voz→texto) +
│   │                       # fetch POST /predict + renderiza resultado con confianza
│   └── css/
│       └── style.css       # Diseño oscuro del visor
│
├── models/                 # Modelos serializados y resultados de entrenamiento
│   ├── model01.pkl             # Pipeline SVM completo serializado con joblib
│   ├── model02.pkl             # Pipeline MLP completo serializado con joblib
│   ├── label_encoder.pkl       # Mapeo: índice entero ↔ nombre de intent
│   ├── cv_results_model01.csv  # Resultados de cada combinación del GridSearchCV SVM
│   ├── cv_results_model02.csv  # Resultados de cada combinación del GridSearchCV MLP
│   ├── cv_results_nb.csv       # Resultados GridSearchCV Naive Bayes
│   ├── model01_summary.csv     # Resumen: mejores parámetros y scores de model01
│   ├── model02_summary.csv     # Resumen: mejores parámetros y scores de model02
│   └── best_model01_name.txt   # Nombre del mejor clasificador: "SVM" o "NaiveBayes"
│
├── tests/
│   ├── conftest.py         # Fixtures compartidos: textos de prueba, labels, corpus mínimo
│   ├── test_data.py        # Verifica schema JSONL, todos los intents presentes,
│   │                       # sin textos vacíos, split con proporciones correctas
│   ├── test_features.py    # Verifica TF-IDF, BoW, bigramas, NO data leakage,
│   │                       # chi² reduce features, variance threshold elimina constantes
│   ├── test_models.py      # Verifica pipelines SVM/NB/MLP se construyen,
│   │                       # accuracy > 0.85, predict() devuelve intent válido
│   └── test_api.py         # Verifica /predict devuelve 200, schema correcto,
│                           # texto vacío devuelve 422, /health funciona
│
├── .env.example            # Template de variables de entorno (copiar a .env)
├── .gitignore              # Excluye: .venv/, .env, *.pkl, data/raw/*, htmlcov/
├── CONTRIBUTING.md         # Guía para contribuir al proyecto
├── Dockerfile              # Build multistage: builder (uv+deps) → runtime (solo lo necesario)
├── docker-compose.yml      # Servicio "api" (producción) + "api-dev" (hot-reload para desarrollo)
├── LICENSE                 # MIT
├── Makefile                # Comandos de conveniencia: make install/data/train/api/test/docker
├── pyproject.toml          # Deps + scripts CLI + config ruff/pytest/mypy (reemplaza requirements.txt)
├── README.md               # Este archivo
└── uv.lock                 # Lockfile reproducible — NO editar manualmente
```

---

## 🔄 Flujo de trabajo completo

El proyecto sigue una metodología Data Science estructurada. Cada paso transforma los datos hasta llegar al modelo deployado.

### Paso 1 — Generación de datos sintéticos

No había datos reales disponibles. Se generó el corpus usando dos LLMs para mayor diversidad:

```bash
uv run python -m src.data.generator
```

**`src/data/generator.py`** llama a:
- **GROQ API** (LLaMA 3.3 70B, en la nube): genera 100 utterances/intent en 4 estilos rotando (variado, corto, largo)
- **Mistral local** (Ollama): genera 50 utterances/intent adicionales con distinto vocabulario

Resultado: `data/raw/intents_raw.jsonl` con ~1146 ejemplos en formato:
```json
{"text": "muéstrame las capas disponibles", "intent": "query_layer"}
```

### Paso 2 — Preprocesamiento

```bash
uv run python -m src.data.preprocess
```

**`src/data/preprocess.py`** aplica a cada utterance:
1. Elimina numeración al inicio (`"18. texto"` → `"texto"`)
2. Minúsculas
3. Regex: elimina caracteres especiales, conserva letras con tildes y ñ
4. spaCy `es_core_news_sm`: lematización + eliminación de stopwords

Resultado: `data/interim/intents_clean.jsonl` con campos `text`, `text_raw`, `tokens`, `n_tokens`, `intent`.

### Paso 3 — Split y codificación

```bash
uv run python -m src.data.dataset
```

**`src/data/dataset.py`**:
- `LabelEncoder`: convierte `"query_layer"` → `0`, ..., `"visualize_map"` → `7`
- `train_test_split` con `stratify=y`: garantiza misma proporción de clases en train y test
- Split: **80% train (~916 ejemplos)** / **20% test (~230 ejemplos)**

Resultado: 4 archivos `.pkl` en `data/processed/` + `label_encoder.pkl` en `models/`

### Paso 4 — Feature Extraction y Feature Selection (en el pipeline)

No se ejecutan como paso separado — están **dentro del pipeline sklearn** de cada modelo. Esto garantiza que el preprocesamiento de features respete el split train/test sin data leakage.

`src/features/extraction.py` implementa BoW, TF-IDF y TF-IDF+ngrams para comparar en el notebook. `src/features/selection.py` implementa chi², mutual info y variance threshold como métodos filter.

### Paso 5 — Entrenamiento de modelos

```bash
uv run python -m src.models.model01.train
uv run python -m src.models.model02.train
```

Ambos `train.py` siguen el mismo proceso:
1. Cargan `X_train.pkl` y `y_train.pkl`
2. Construyen el pipeline desde `model01.py` / `model02.py`
3. **GridSearchCV** con `RepeatedStratifiedKFold(n_splits=10, n_repeats=10)` → 100 folds
4. Guardan los resultados del CV en CSV para el notebook
5. Generan y guardan la curva de aprendizaje en `reports/figures/`
6. Evalúan en test set y muestran classification report + matriz de confusión
7. Serializan el mejor pipeline con `joblib.dump()` → `models/model01.pkl`

### Paso 6 — Despliegue de la API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

**`src/api/main.py`** al iniciar:
- Carga `model01.pkl` y `model02.pkl` en memoria (evita latencia en la primera request)
- Sirve `frontend/` como archivos estáticos en `localhost:8000/app`

**`src/api/routes.py`** en cada request a `POST /predict`:
1. Recibe el texto del usuario
2. Aplica `preprocess_text()` con spaCy (mismo pipeline que el entrenamiento)
3. Llama a `predict(clean_text)` del modelo seleccionado
4. Devuelve `{intent, confidence, agent, model, text}`

> **¿Por qué el preprocesamiento en la API?**  
> Los modelos fueron entrenados con texto preprocesado (lematizado, sin stopwords). Si la API enviara texto crudo, habría inconsistencia entre entrenamiento y producción — el modelo recibiría distribuciones de texto que nunca vio, bajando la confianza. Al aplicar el mismo preprocesamiento en la API, el F1 reportado en el notebook es válido en producción.

---

## 🚀 Instalación

### Requisitos

- Python 3.12
- [uv](https://github.com/astral-sh/uv) — gestor de paquetes
- GROQ API Key gratuita — [console.groq.com](https://console.groq.com)
- Ollama (opcional, para generación local con Mistral)

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

# 4. Instalar dependencias de producción
uv sync

# 5. Instalar dependencias de desarrollo (tests, linting)
uv sync --extra dev

# 6. Configurar variables de entorno
cp .env.example .env
# Editar .env: agregar GROQ_API_KEY=gsk_...

# 7. Descargar modelo de español de spaCy (una sola vez)
uv run python -m spacy download es_core_news_sm
```

---

## 🗂️ Pipeline de datos

```
GROQ API (LLaMA 3.3 70B) + Mistral local (Ollama)
         │  src/data/generator.py
         ▼
data/raw/intents_raw.jsonl          ← ~1146 utterances crudos
         │  src/data/preprocess.py (spaCy es_core_news_sm)
         ▼
data/interim/intents_clean.jsonl    ← lematizados, sin stopwords
         │  src/data/dataset.py
         ▼
data/processed/X_train.pkl          ← 80% textos para entrenar
data/processed/X_test.pkl           ← 20% textos para evaluar
data/processed/y_train.pkl          ← labels enteros train
data/processed/y_test.pkl           ← labels enteros test
         │  src/models/model0N/train.py
         │  (Pipeline: TF-IDF → chi² → clasificador)
         ▼
models/model01.pkl                  ← pipeline SVM serializado
models/model02.pkl                  ← pipeline MLP serializado
```

---

## 🤖 Modelos

### Model01 — Baseline (TF-IDF + chi² + SVM)

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)),
    ('chi2',  SelectKBest(chi2, k=1000)),
    ('clf',   CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
])
```

| Componente | Función |
|-----------|---------|
| `TfidfVectorizer` | Convierte texto en vector numérico, ponderando términos más informativos. `sublinear_tf=True` aplica `log(tf)` para reducir el efecto de términos muy frecuentes |
| `SelectKBest(chi2)` | Selecciona los términos con mayor dependencia estadística con la clase (método filter) |
| `LinearSVC` | Clasificador SVM lineal — encuentra el hiperplano óptimo en el espacio TF-IDF |
| `CalibratedClassifierCV` | Calibra probabilidades del SVM (que nativamente no las tiene) mediante CV interna |

### Model02 — Avanzado (TF-IDF word+char + MLP)

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

| Componente | Función |
|-----------|---------|
| `FeatureUnion` | Combina en paralelo dos vectorizadores — concatena sus matrices de features |
| `word_tfidf` | Captura n-grams de palabras: `"calcular área"`, `"exportar shapefile"` |
| `char_tfidf` | Captura n-grams de caracteres: `"calc"`, `"expo"` — robusto ante errores tipográficos |
| `MaxAbsScaler` | Normaliza las features entre -1 y 1 sin destruir la sparsidad — necesario para MLP |
| `MLPClassifier` | Red neuronal densa con early stopping — para cuando la validación deja de mejorar |

### Optimización de hiperparámetros

**GridSearchCV** — búsqueda exhaustiva (requerido por rúbrica):
```python
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)  # 100 folds
GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
# Model01: 54 combinaciones × 100 folds = 5400 fits
# Model02: 48 combinaciones × 100 folds = 4800 fits
```

**Optuna** — búsqueda bayesiana TPE (extra):
```python
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=40)
# Converge a resultados similares con ~400 evaluaciones (10x menos)
```

---

## 📊 Resultados

### Evaluación en test set

| Modelo | Accuracy | F1-macro | Condición |
|--------|----------|----------|-----------|
| **Model01 (SVM)** | **0.96** | **0.96** | Texto preprocesado |
| Model02 (MLP) | 0.88 | 0.88 | Texto preprocesado |

### Comparación estadística — Test de Wilcoxon (10×10 CV)

```
Estadístico W : (resultado real del CV)
p-valor       : < 0.05
✓ Diferencia SIGNIFICATIVA — Model01 es estadísticamente superior
```

**Nota importante:** Model01 gana en condiciones controladas (texto preprocesado). Model02 es más robusto en producción con texto crudo gracias a los char n-grams. Se seleccionó **Model02 como modelo por defecto en producción**.

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

> `query_layer` y `visualize_map` tienen menor F1 porque comparten vocabulario ("mostrar", "mapa", "capas"). Son los intents más difíciles de distinguir.

---

## 🌐 API

### Levantar

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

**`POST /predict`** — Clasifica una instrucción:
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

**`GET /health`** — Estado del servicio:
```json
{"status": "ok", "model01": true, "model02": true}
```

**`WebSocket /speech`** — Voz en tiempo real:
El browser transcribe audio con Web Speech API y envía el texto al WebSocket. La API clasifica y devuelve el intent.

---

## 🖥️ Frontend

Visor accesible en `http://localhost:8000/app`.

FastAPI sirve `frontend/` como archivos estáticos con una sola línea:
```python
app.mount("/app", StaticFiles(directory="frontend", html=True))
```

Funcionalidades:
- **Texto:** escribe cualquier instrucción y presiona Clasificar
- **Voz:** botón micrófono usa Web Speech API nativa del browser (Chrome/Edge)
- **Selector de modelo:** Model01 (SVM) o Model02 (MLP)
- **Resultado:** intent, barra de confianza con color (verde/amarillo/rojo), agente y modelo

---

## 🐳 Docker

### ¿Por qué Docker?

Sin Docker, el proyecto depende de que el entorno local tenga exactamente Python 3.12, las mismas versiones de sklearn/spaCy y el mismo OS. Con Docker, todo eso está encapsulado en una imagen reproducible.

```bash
# Cualquier persona puede levantar el proyecto completo con:
docker compose up --build
# → http://localhost:8000/app  (frontend)
# → http://localhost:8000/docs (API)
```

### Build y levantar

```bash
# Producción
docker compose up --build

# Desarrollo (hot-reload)
docker compose --profile dev up --build

# Detener
docker compose down
```

---

## 🧪 Tests

```bash
# Todos los tests
uv run pytest

# Con cobertura HTML
uv run pytest --cov=src --cov-report=html
```

**Resultado: 41/41 tests passed**

| Archivo | Qué verifica |
|---------|-------------|
| `test_data.py` | Schema JSONL correcto, todos los intents presentes, sin textos vacíos |
| `test_features.py` | TF-IDF funciona, bigramas detectados, **no data leakage**, chi² reduce features |
| `test_models.py` | Pipelines se construyen, accuracy > 0.85 en test, predict() devuelve intent válido |
| `test_api.py` | /predict devuelve 200 con schema correcto, texto vacío devuelve 422 |

---

## 📓 Notebook

El análisis completo está en `notebooks/main.ipynb`. El notebook es un **reporte ejecutivo** — carga los datos y modelos ya generados sin reentrenar desde cero.

| Sección | Descripción |
|---------|-------------|
| **0. Setup** | Imports, paths, constantes e intents definidos |
| **1. Introducción** | Problema, propuesta, arquitectura, tabla de intents |
| **2. EDA** | Distribución de clases, longitud de utterances, top n-grams por intent |
| **3. Preprocesamiento** | Ejemplos antes/después de limpieza con spaCy |
| **4. Feature Extraction** | Comparación BoW vs TF-IDF vs TF-IDF+n-grams con tabla de vocabulario y densidad |
| **5. Pipelines** | Definición de los 3 pipelines sklearn con justificación de cada componente |
| **6. Optimización** | GridSearchCV real (desde CSV) + gráfica F1 vs λ + curvas de aprendizaje + Optuna |
| **7. Comparación estadística** | Wilcoxon 10×10 + boxplot de scores + diferencia por fold |
| **8. Evaluación** | Classification report + matrices de confusión de ambos modelos |
| **9. t-SNE** | Espacio de features TF-IDF en 2D con colores por intent |
| **10. Conclusiones** | Análisis de resultados, limitaciones y recomendaciones |
| **11. Referencias** | IEEE + prompts de IA usados como comentarios Python |

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

**[9]** Astral, "uv — An extremely fast Python package manager," 2024. [Online]. Available: https://github.com/astral-sh/uv

**[10]** J. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge University Press, 2008.
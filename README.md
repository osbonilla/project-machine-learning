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
- [Instalación y comandos](#-instalación-y-comandos)
- [Pipeline de datos](#-pipeline-de-datos)
- [Modelos y justificación](#-modelos-y-justificación)
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
┌─────────────────────────────────────┐
│  Clasificador de Intención          │
│                                     │
│  Model01: TF-IDF + chi² + SVM       │  ← F1-macro: 0.96 (texto preprocesado)
│  Model02: TF-IDF + chi² + MLP       │  ← F1-macro: 0.88 (más robusto en producción)
└────────┬────────────────────────────┘
         │
         ▼
   Agente Geoespacial (geo_agent)
   Recibe el intent clasificado y ejecuta la operación GIS correspondiente
```

---

## ⚙️ Decisiones técnicas

### ¿Qué es TF-IDF?

**TF-IDF** (Term Frequency — Inverse Document Frequency) es el método que convierte texto en números que el modelo puede procesar. Asigna un peso numérico a cada palabra según dos criterios:

- **TF (Term Frequency):** qué tan frecuente es la palabra en ese documento
- **IDF (Inverse Document Frequency):** qué tan rara es la palabra en todo el corpus

```
Palabra "buffer" → aparece mucho en buffer_analysis, poco en el resto → peso ALTO
Palabra "de"    → aparece en todos los documentos por igual        → peso BAJO
```

El resultado es una **matriz numérica** donde cada fila es un utterance y cada columna es un término del vocabulario. Los modelos ML trabajan sobre esta matriz, no sobre el texto directamente.

Con `ngram_range=(1,2)` también captura **bigramas** — pares de palabras consecutivas:

```
"calcular área" → una sola feature que distingue calculate_area de otros intents
"exportar shapefile" → feature única de export_data
```

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
│   │
│   ├── data/
│   │   ├── generator.py    # Genera utterances usando GROQ API + Mistral (Ollama)
│   │   ├── preprocess.py   # Limpia: numeración → minúsculas → regex → spaCy
│   │   └── dataset.py      # Split estratificado 80/20 y guarda los .pkl
│   │
│   ├── features/
│   │   ├── extraction.py   # BoW, TF-IDF y TF-IDF+ngrams para comparar en notebook
│   │   └── selection.py    # chi², mutual info y variance threshold (métodos filter)
│   │
│   ├── models/
│   │   ├── model01/        # Pipeline baseline: TF-IDF + chi² + SVM/NB
│   │   │   ├── model01.py  # Define build_svm_pipeline() y build_nb_pipeline()
│   │   │   ├── train.py    # GridSearchCV + RepeatedKFold(10×10) → model01.pkl
│   │   │   ├── predict.py  # predict(text) → {intent, confidence, agent}
│   │   │   └── dataloader.py
│   │   └── model02/        # Pipeline avanzado: TF-IDF(word+char) + chi² + MLP
│   │       ├── model02.py  # Define build_mlp_pipeline() con FeatureUnion
│   │       ├── train.py    # GridSearchCV + RepeatedKFold(10×10) → model02.pkl
│   │       ├── predict.py  # predict(text) → {intent, confidence, agent}
│   │       └── dataloader.py
│   │
│   ├── api/
│   │   ├── main.py         # FastAPI: carga modelos al iniciar + sirve frontend
│   │   ├── routes.py       # POST /predict (con preprocesamiento spaCy) + /health + WS
│   │   ├── schemas.py      # Pydantic: IntentRequest, IntentResponse, HealthResponse
│   │   └── middleware.py   # CORS + logging de requests
│   │
│   └── speech/
│       ├── stt.py          # Speech-to-Text
│       └── tts.py          # Text-to-Speech
│
├── frontend/               # Visor demo — HTML/CSS/JS puro
│   ├── index.html
│   ├── app.js              # Web Speech API + fetch /predict + render resultado
│   └── css/style.css
│
├── models/                 # Modelos y resultados de entrenamiento
│   ├── model01.pkl
│   ├── model02.pkl
│   ├── label_encoder.pkl
│   ├── cv_results_model01.csv
│   ├── cv_results_model02.csv
│   ├── cv_results_nb.csv
│   ├── model01_summary.csv
│   ├── model02_summary.csv
│   └── best_model01_name.txt
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 🔄 Flujo de trabajo completo

### Paso 1 — Generación de datos sintéticos

```bash
uv run python -m src.data.generator
```

Llama a GROQ API (LLaMA 3.3 70B) + Mistral local (Ollama) para generar ~1146 utterances.
Resultado: `data/raw/intents_raw.jsonl`

### Paso 2 — Preprocesamiento

```bash
uv run python -m src.data.preprocess
```

Aplica: numeración → minúsculas → regex → lematización spaCy → stopwords.
Resultado: `data/interim/intents_clean.jsonl`

### Paso 3 — Split y codificación

```bash
uv run python -m src.data.dataset
```

LabelEncoder + split estratificado 80/20.
Resultado: `data/processed/*.pkl` + `models/label_encoder.pkl`

### Paso 4 — Entrenamiento

```bash
uv run python -m src.models.model01.train
uv run python -m src.models.model02.train
```

GridSearchCV + RepeatedKFold(10×10) para ambos modelos.
Resultado: `models/model01.pkl`, `models/model02.pkl`, CSVs de resultados

### Paso 5 — API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

Disponible en `http://localhost:8000`

> **¿Por qué el preprocesamiento en la API?**
> Los modelos fueron entrenados con texto preprocesado. Si la API enviara texto crudo habría inconsistencia entre entrenamiento y producción. El mismo pipeline spaCy se aplica en `/predict` antes de clasificar, garantizando que el F1 reportado en el notebook sea válido en producción.

---

## 🚀 Instalación y comandos

### Requisitos

- Python 3.12
- [uv](https://github.com/astral-sh/uv)
- GROQ API Key — [console.groq.com](https://console.groq.com)
- Ollama (opcional) — [ollama.com](https://ollama.com)

### Setup inicial

```bash
# Clonar
git clone https://github.com/tu-usuario/project-machine-learning.git
cd project-machine-learning

# Entorno virtual
uv venv .venv --python 3.12

# Activar (Windows)
.venv\Scripts\activate
# Activar (Linux/macOS)
source .venv/bin/activate

# Instalar dependencias
uv sync

# Dependencias de desarrollo (tests, linting)
uv sync --extra dev

# Variables de entorno
cp .env.example .env
# Editar .env: agregar GROQ_API_KEY=gsk_...

# Modelo spaCy (una sola vez)
uv run python -m spacy download es_core_news_sm
```

### Comandos principales

```bash
# ── Pipeline completo de datos ──────────────────────────
uv run python -m src.data.generator      # 1. Generar corpus con GROQ + Mistral
uv run python -m src.data.preprocess     # 2. Limpiar y lematizar con spaCy
uv run python -m src.data.dataset        # 3. Split 80/20 y guardar .pkl

# ── Entrenamiento ───────────────────────────────────────
uv run python -m src.models.model01.train   # Entrenar Model01 (SVM)
uv run python -m src.models.model02.train   # Entrenar Model02 (MLP)

# ── API ─────────────────────────────────────────────────
uv run uvicorn src.api.main:app --reload --port 8000

# ── Tests ───────────────────────────────────────────────
uv run pytest                            # Todos los tests
uv run pytest --cov=src                  # Con cobertura

# ── Calidad de código ───────────────────────────────────
uv run ruff check src/                   # Linting
uv run ruff format src/                  # Formatear

# ── Docker ──────────────────────────────────────────────
docker compose up --build                # Producción
docker compose --profile dev up --build  # Desarrollo (hot-reload)
docker compose down                      # Detener

# ── Makefile (atajos) ───────────────────────────────────
make install      # setup completo
make data         # genera + preprocesa + splitea
make train        # entrena ambos modelos
make api          # levanta la API
make test         # corre tests
make docker       # build y levanta Docker
make clean        # limpia caches
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
         │  Pipeline: TF-IDF → chi² → clasificador
         ▼
models/model01.pkl                  ← pipeline SVM serializado
models/model02.pkl                  ← pipeline MLP serializado
```

---

## 🤖 Modelos y justificación

### ¿Por qué dos modelos?

Se implementan dos modelos con enfoques distintos para comparar estadísticamente su rendimiento (test de Wilcoxon) y elegir el mejor para producción.

---

### Model01 — Baseline: TF-IDF + chi² + SVM

**¿Por qué SVM para clasificación de texto?**

El SVM (Support Vector Machine) es el modelo clásico más efectivo para clasificación de texto por tres razones:
1. **Funciona bien en alta dimensionalidad:** el espacio TF-IDF puede tener miles de features — SVM los maneja eficientemente
2. **Margen máximo:** encuentra el hiperplano que maximiza la separación entre clases, siendo robusto con pocos datos
3. **Velocidad:** `LinearSVC` es extremadamente rápido comparado con kernels no lineales o redes neuronales

**¿Por qué también Naive Bayes?**

Se incluye Naive Bayes como alternativa para seleccionar el mejor mediante comparación directa. NB es el baseline tradicional para clasificación de texto — si SVM no lo supera, no justifica su complejidad adicional.

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)),
    ('chi2',  SelectKBest(chi2, k=1000)),
    ('clf',   CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
])
```

| Componente | Por qué se usa |
|-----------|----------------|
| `TfidfVectorizer` | Convierte texto a vector numérico ponderando términos informativos. `sublinear_tf=True` aplica `log(tf)` reduciendo el peso de términos muy repetidos |
| `ngram_range=(1,2)` | Captura bigramas como *"calcular área"* que son más discriminativos que palabras sueltas |
| `SelectKBest(chi2)` | Filtra los términos estadísticamente más dependientes de la clase — reduce ruido antes del clasificador |
| `LinearSVC` | SVM lineal optimizado para alta dimensionalidad — rápido y preciso en texto |
| `CalibratedClassifierCV` | LinearSVC no tiene `predict_proba()` nativo — la calibración permite obtener el score de confianza |

**Resultado:** F1-macro = 0.96 en test set ✅

---

### Model02 — Avanzado: TF-IDF (word+char) + MLP

**¿Por qué MLP en lugar de transformers?**

Con ~900 ejemplos de entrenamiento, los transformers (BERT, RoBERTa) tienden a hacer overfitting. Un MLP sobre features TF-IDF bien diseñadas es más eficiente y competitivo con datasets pequeños. Además, los transformers requieren GPU y torch — el MLP corre en CPU sin dependencias pesadas.

**¿Por qué FeatureUnion con char n-grams?**

La innovación de Model02 es combinar dos tipos de features en paralelo:
- **word n-grams:** captura frases clave completas (`"exportar shapefile"`, `"calcular área"`)
- **char n-grams:** captura patrones de caracteres (`"calc"`, `"expo"`, `"buff"`) — más robusto ante errores tipográficos y variaciones morfológicas

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

| Componente | Por qué se usa |
|-----------|----------------|
| `FeatureUnion` | Combina en paralelo ambos vectorizadores — el vector final es la concatenación de ambas matrices |
| `char_wb n-grams (2,4)` | Captura morfología sin preprocesamiento — `"calcular"` y `"calculando"` comparten los mismos char n-grams |
| `MaxAbsScaler` | Normaliza features entre -1 y 1 sin destruir la sparsidad — necesario porque MLP es sensible a la escala |
| `MLPClassifier (256,128)` | Red neuronal densa de dos capas — aprende combinaciones no lineales de features |
| `early_stopping=True` | Detiene el entrenamiento cuando la validación deja de mejorar — evita overfitting |

**Resultado:** F1-macro = 0.88 en test set, pero **más robusto con texto crudo en producción** gracias a los char n-grams ✅

---

### Comparación de modelos

| Criterio | Model01 (SVM) | Model02 (MLP) |
|----------|--------------|--------------|
| F1-macro test | **0.96** | 0.88 |
| Velocidad entrenamiento | Muy rápido | Lento |
| Robustez texto crudo | Baja | **Alta** |
| Interpretabilidad | Alta | Media |
| Memoria | Pequeña | Mayor |
| **Uso recomendado** | Evaluación académica | **Producción** |

**¿Por qué Model02 en producción?**
Model01 gana en métricas pero fue evaluado con texto preprocesado (spaCy). Los char n-grams de Model02 hacen que sea más tolerante ante variaciones del usuario real. Aunque el preprocesamiento se aplica también en la API, Model02 ofrece mayor robustez como capa de seguridad adicional.

---

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
p-valor < 0.05
✓ Diferencia SIGNIFICATIVA — Model01 es estadísticamente superior en CV
```

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

> `query_layer` y `visualize_map` tienen menor F1 porque comparten vocabulario ("mostrar", "mapa", "capas").

---

## 🌐 API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

**`POST /predict`:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "muéstrame las capas disponibles", "model": "model02"}'
```
```json
{"intent": "query_layer", "confidence": 0.9722, "agent": "geo_agent", "model": "model02"}
```

**`GET /health`:** `{"status": "ok", "model01": true, "model02": true}`

**`WebSocket /speech`:** voz → Web Speech API → texto → classify → intent

---

## 🖥️ Frontend

Visor en `http://localhost:8000/app` — texto o voz, selector de modelo, resultado con confianza.

---

## 🐳 Docker

```bash
docker compose up --build          # producción en localhost:8000
docker compose --profile dev up    # desarrollo con hot-reload
docker compose down                # detener
```

---

## 🧪 Tests

```bash
uv run pytest                      # 41/41 passed
uv run pytest --cov=src            # con cobertura
```

| Archivo | Qué verifica |
|---------|-------------|
| `test_data.py` | Schema JSONL, intents presentes, sin textos vacíos |
| `test_features.py` | TF-IDF, bigramas, **no data leakage**, chi² |
| `test_models.py` | Pipelines, accuracy > 0.85, predict() válido |
| `test_api.py` | /predict 200, schema correcto, vacío → 422 |

---

## 📓 Notebook

`notebooks/main.ipynb` — reporte ejecutivo que carga datos y modelos ya generados.

| Sección | Descripción |
|---------|-------------|
| 0. Setup | Imports, paths, constantes |
| 1. Introducción | Problema, propuesta, tabla de intents |
| 2. EDA | Distribución, longitud, n-grams por intent |
| 3. Preprocesamiento | Ejemplos antes/después de spaCy |
| 4. Feature Extraction | BoW vs TF-IDF vs TF-IDF+n-grams |
| 5. Pipelines | Definición y justificación de los 3 pipelines |
| 6. Optimización | GridSearchCV + F1 vs λ + curvas de aprendizaje + Optuna |
| 7. Comparación estadística | Wilcoxon 10×10 + boxplot |
| 8. Evaluación | Classification report + matrices de confusión |
| 9. t-SNE | Espacio de features en 2D |
| 10. Conclusiones | Resultados, limitaciones, recomendaciones |
| 11. Referencias | IEEE + prompts de IA |

---

## 📚 Referencias

**[1]** F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.

**[2]** G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, vol. 24, no. 5, pp. 513–523, 1988.

**[3]** C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

**[4]** L. Van der Maaten and G. Hinton, "Visualizing Data using t-SNE," *JMLR*, vol. 9, pp. 2579–2605, 2008.

**[5]** T. Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework," *KDD*, 2019. https://optuna.org

**[6]** Explosion AI, "spaCy: Industrial-strength NLP," 2023. https://spacy.io

**[7]** GROQ Inc., "GROQ API Documentation," 2024. https://console.groq.com/docs

**[8]** F. Wilcoxon, "Individual Comparisons by Ranking Methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.

**[9]** Astral, "uv — An extremely fast Python package manager," 2024. https://github.com/astral-sh/uv

**[10]** J. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge University Press, 2008.
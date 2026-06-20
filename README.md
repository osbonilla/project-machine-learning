# 📁 Estructura del proyecto

```
geo-intent-classifier/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Lint + tests
│       └── train.yml                   # Pipeline de entrenamiento en CI
│
├── data/
│   ├── raw/
│   │   └── intents_raw.jsonl           # Utterances crudos generados por GROQ
│   ├── interim/
│   │   └── intents_clean.jsonl         # Datos limpios y tokenizados
│   ├── processed/
│   │   ├── X_train.pkl
│   │   ├── X_test.pkl
│   │   ├── y_train.pkl
│   │   └── y_test.pkl
│   └── external/                       # Stopwords, vocabularios externos
│
├── docs/                               # Documentación mkdocs
│
├── notebooks/
│   ├── 01_data_generation.ipynb        # Generación de corpus con GROQ API
│   ├── 02_eda.ipynb                    # Exploratory Data Analysis
│   ├── 03_preprocessing.ipynb          # Limpieza, tokenización, lematización
│   ├── 04_feature_extraction.ipynb     # TF-IDF, embeddings, n-grams
│   ├── 05_feature_selection.ipynb      # chi2, mutual info (filter methods)
│   ├── 06_model_baseline.ipynb         # Naive Bayes / SVM — model01
│   ├── 07_model_advanced.ipynb         # Embeddings + MLP — model02
│   ├── 08_hyperparameter_tuning.ipynb  # Repeated k-fold CV + curva de aprendizaje
│   ├── 09_statistical_comparison.ipynb # Wilcoxon / t-test entre modelos
│   └── 10_tsne_visualization.ipynb     # t-SNE con labels de intención
│
├── reports/
│   ├── figures/                        # t-SNE, curvas de aprendizaje, matrices de confusión
│   └── report.md                       # Reporte final
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, clases de intención, constantes
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py                # GROQ API → utterances etiquetados por intent
│   │   ├── dataset.py                  # Carga, valida y splitea el corpus
│   │   └── preprocess.py              # Tokenización, stemming, lematización (spaCy)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extraction.py              # TF-IDF, BoW, sentence embeddings
│   │   └── selection.py               # chi2, mutual info, variance threshold
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model01/                   # Baseline: TF-IDF + Naive Bayes / SVM
│   │   │   ├── dataloader.py
│   │   │   ├── model01.py
│   │   │   ├── train.py
│   │   │   └── predict.py
│   │   └── model02/                   # Avanzado: embeddings + MLP / transformer
│   │       ├── dataloader.py
│   │       ├── model02.py
│   │       ├── train.py
│   │       └── predict.py
│   │
│   └── api/                           # [plus] Expone el modelo + sirve el visor
│       ├── __init__.py
│       ├── main.py                    # FastAPI app + monta frontend/ como static
│       ├── routes.py                  # POST /predict · WebSocket /speech
│       ├── schemas.py                 # IntentRequest / IntentResponse (Pydantic)
│       └── middleware.py              # CORS, logging
│
├── frontend/                          # [plus] Visor demo — HTML/CSS/JS puro, sin framework
│   ├── index.html                     # Caja de texto + botón micrófono
│   └── app.js                         # Web Speech API + fetch /predict + render resultado
│
├── models/
│   ├── model01.pkl                    # Pipeline sklearn serializado
│   ├── model02.pkl
│   └── label_encoder.pkl              # Mapeo índice → nombre de intent
│
├── tests/
│   ├── conftest.py                    # Fixtures compartidos de pytest
│   ├── test_data.py                   # Valida schema y calidad del corpus
│   ├── test_features.py               # Extracción y selección de features
│   ├── test_models.py                 # Predicción, umbrales de accuracy
│   └── test_api.py                    # Endpoints /predict y /speech
│
├── .env.example                       # Variables: GROQ_API_KEY, MODEL_PATH, etc.
├── .gitignore
├── AGENTS.md                          # Instrucciones para IDE agéntico
├── CONTRIBUTING.md
├── Dockerfile                         # Imagen de producción (API + frontend estático)
├── docker-compose.yml                 # Orquesta API + servicios auxiliares
├── LICENSE
├── Makefile                           # make data | make train | make api | make docker
├── pyproject.toml                     # Deps + scripts (uv — sin requirements.txt)
├── README.md
└── uv.lock                            # Lockfile reproducible de uv
```


# ─────────────────────────────────────────────────────────
#  project-machine-learning — Makefile
# ─────────────────────────────────────────────────────────

.PHONY: help install install-dev data train-01 train-02 train api docker docker-dev docker-down test test-cov lint format clean

PYTHON := .venv/bin/python
UV     := uv

RESET  := \033[0m
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m

help:  ## Muestra esta ayuda
	@echo ""
	@echo "$(BOLD)project-machine-learning$(RESET)"
	@echo "──────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""

install:  ## Crea .venv Python 3.12 e instala dependencias
	$(UV) venv .venv --python 3.12
	$(UV) sync
	@echo "$(GREEN)✓ Entorno listo. Activa con: source .venv/bin/activate$(RESET)"

install-dev:  ## Instala dependencias de desarrollo
	$(UV) sync --extra dev
	@echo "$(GREEN)✓ Dev deps instaladas$(RESET)"

data:  ## Genera corpus de utterances con GROQ API
	@echo "$(YELLOW)→ Generando datos con GROQ...$(RESET)"
	$(UV) run generate-data

train-01:  ## Entrena model01 (TF-IDF + Naive Bayes / SVM)
	@echo "$(YELLOW)→ Entrenando model01...$(RESET)"
	$(UV) run train-model01

train-02:  ## Entrena model02 (embeddings + MLP)
	@echo "$(YELLOW)→ Entrenando model02...$(RESET)"
	$(UV) run train-model02

train: train-01 train-02  ## Entrena ambos modelos en secuencia

api:  ## Levanta la API FastAPI en localhost:8000
	@echo "$(YELLOW)→ Iniciando API en http://localhost:8000$(RESET)"
	$(UV) run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker:  ## Build y levanta contenedor de producción
	docker compose up --build

docker-dev:  ## Levanta contenedor en modo desarrollo
	docker compose --profile dev up --build

docker-down:  ## Detiene contenedores
	docker compose down

test:  ## Corre suite completa de tests
	$(UV) run pytest

test-cov:  ## Tests con reporte de cobertura HTML
	$(UV) run pytest --cov=src --cov-report=html
	@echo "$(GREEN)✓ Reporte en htmlcov/index.html$(RESET)"

lint:  ## Linting con ruff
	$(UV) run ruff check src/ tests/

format:  ## Formatea código con ruff
	$(UV) run ruff format src/ tests/

clean:  ## Elimina caches y archivos temporales
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "$(GREEN)✓ Limpio$(RESET)"
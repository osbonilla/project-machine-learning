# Contribuciones — project-machine-learning

## Setup

```bash
git clone https://github.com/tu-usuario/project-machine-learning.git
cd project-machine-learning
make install-dev
cp .env.example .env   # completar con tu GROQ_API_KEY
```

## Flujo de trabajo

1. Crear rama desde `main`: `git checkout -b feat/nombre`
2. Asegurarse que `make lint` y `make test` pasen antes del PR
3. Abrir Pull Request con descripción clara

## Estándares

- Formato con `make format` (ruff)
- Tests para todo código nuevo en `tests/`
- Commits en inglés: `feat:`, `fix:`, `docs:`, `refactor:`
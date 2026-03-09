# RathBones / Investor ML — common tasks
# Usage: make [target]. On Windows you may need: make (with Make installed) or run commands manually.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: all help install install-dev test test-cov lint run train clean fix

all: install lint test

help:
	@echo "RathBones ML — targets:"
	@echo "  make / make all   — install-dev, lint, test"
	@echo "  make install      — install package (no dev deps)"
	@echo "  make install-dev  — install with dev deps (pytest, ruff)"
	@echo "  make test         — run pytest"
	@echo "  make test-cov     — run pytest with coverage; opens htmlcov/index.html (coverage doc)"
	@echo "  make lint         — run ruff check"
	@echo "  make run          — start FastAPI server (uvicorn)"
	@echo "  make train        — run train pipeline once (no server)"
	@echo "  make clean        — remove artifacts, caches, build dirs"
	@echo "  make fix          — run ruff auto-fix"


install:
	@echo "Installing package..."
	$(PIP) install -e .

install-dev:
	@echo "Installing dev dependencies..."
	$(PIP) install -e ".[dev]"

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest tests -v --tb=short --cov=src/investor_ml --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	@echo "=============================================== Running Ruff checks ================================================"
	$(PYTHON) -m ruff check src tests

fix:
	@echo "Running Ruff auto-fix..."
	$(PYTHON) -m ruff check . --fix

run:
	@echo "Running FastAPI server..."
	$(PYTHON) -m uvicorn investor_ml.web.api:app --host 0.0.0.0 --port 8000

train:
	@echo "Running train pipeline..."
	$(PYTHON) -c "from investor_ml.pipeline.run import run_train_evaluate_pipeline; run_train_evaluate_pipeline()"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf src/investor_ml.egg-info
	rm -rf build dist .eggs
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov


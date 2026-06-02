# =============================================================================
# Manifold DB — Makefile
# =============================================================================
# Usage: make <target>
# =============================================================================

.PHONY: help install lint format test test-cov test-integration test-benchmark \
        server cli docker-build docker-up docker-down clean docs release check \
        typecheck isort-check black-check ruff-check

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PYTHON   ?= python
PIP      ?= pip
PYTEST   ?= pytest
DOCKER   ?= docker
MYPY     ?= mypy
RUFF     ?= ruff
BLACK    ?= black
ISORT    ?= isort
MKDOCS   ?= mkdocs

# ---------------------------------------------------------------------------
# Help (default target)
# ---------------------------------------------------------------------------
help: ## Show this help message
	@echo "Manifold DB — Available targets"
	@echo "==============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install: ## Install package with dev dependencies
	$(PIP) install -e ".[dev]"

install-all: ## Install package with all optional dependencies
	$(PIP) install -e ".[all]"

# ---------------------------------------------------------------------------
# Linting & formatting
# ---------------------------------------------------------------------------
lint: ruff-check typecheck ## Run ruff lint + mypy type check
	@echo "✅ All linting checks passed."

ruff-check: ## Run ruff linter
	$(RUFF) check manifold_db/

black-check: ## Run black in check-only mode
	$(BLACK) --check manifold_db/ tests/

isort-check: ## Run isort in check-only mode
	$(ISORT) --check-only manifold_db/ tests/

typecheck: ## Run mypy type checker
	$(MYPY) manifold_db/

format: ## Auto-format code with black + isort
	$(BLACK) manifold_db/ tests/
	$(ISORT) manifold_db/ tests/
	@echo "✅ Code formatted."

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: ## Run test suite
	$(PYTEST) tests/ -m "not slow and not gpu" --tb=short

test-all: ## Run all tests including slow ones
	$(PYTEST) tests/ --tb=short

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=manifold_db --cov-report=term-missing --cov-report=html:htmlcov -m "not slow and not gpu"

test-integration: ## Run integration tests
	$(PYTEST) tests/ -m integration --tb=short

test-benchmark: ## Run performance benchmarks
	$(PYTEST) tests/ -m benchmark --benchmark-only --benchmark-sort=name

test-slow: ## Run slow tests
	$(PYTEST) tests/ -m slow --tb=short

# ---------------------------------------------------------------------------
# Server & CLI
# ---------------------------------------------------------------------------
server: ## Start the API server
	$(PYTHON) -m uvicorn manifold_db.api.main:app --host 0.0.0.0 --port 8000 --reload

cli: ## Run CLI help
	$(PYTHON) -m manifold_db.cli.main --help

worker: ## Start background worker
	$(PYTHON) -m manifold_db.cli.worker

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build Docker image
	$(DOCKER) build -t manifold-db:latest .

docker-up: ## Start docker-compose stack
	docker-compose up -d

docker-up-prod: ## Start docker-compose with production profile (includes PostgreSQL)
	docker-compose --profile production up -d

docker-down: ## Stop docker-compose stack
	docker-compose down

docker-logs: ## Tail docker-compose logs
	docker-compose logs -f

docker-clean: ## Stop docker-compose and remove volumes
	docker-compose down -v

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------
docs: ## Build documentation with mkdocs
	$(MKDOCS) build

docs-serve: ## Serve docs locally with live reload
	$(MKDOCS) serve

# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------
release: clean lint test-cov ## Run all checks, then build release artifacts
	$(PYTHON) -m build
	@echo "✅ Release artifacts in dist/"

# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
check: lint test ## Run linting + tests
	@echo "✅ All checks passed."

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove build artifacts, caches, and generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"      -delete               2>/dev/null || true
	find . -type f -name "*.egg-info"  -exec rm -rf {} +     2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage manifold.db
	@echo "✅ Cleaned."

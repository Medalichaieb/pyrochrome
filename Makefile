# Pyrochrome — reproducible pipeline. Each target is one command, in order:
#   data -> train -> eval -> export
# All Python runs through `uv run` so the pinned environment (uv.lock) is used.

GLAZY_REPO := https://github.com/derekphilipau/glazy-data.git
DATA_RAW   := data/raw
DATA_PROC  := data/processed

.PHONY: help setup data train baseline compare eval export lint format typecheck test check clean

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtual env and install all deps (incl. dev).
	uv sync --extra dev

data: ## Download the Glazy dataset (CSV + YAML dump) into data/raw/.
	@test -d $(DATA_RAW)/glazy-data || git clone --depth 1 $(GLAZY_REPO) $(DATA_RAW)/glazy-data
	uv run python -m pyrochrome.pipeline.download

train: ## Train and persist the selected model (HistGradientBoosting) per target.
	uv run python -m pyrochrome.models.train

baseline: ## Reproduce the reference baseline (RF / GradBoost vs naïve).
	uv run python -m pyrochrome.models.baseline

compare: ## Cross-validate candidate models per target (model selection).
	uv run python -m pyrochrome.models.compare

eval: ## Evaluate trained models and regenerate report metrics.
	uv run python -m pyrochrome.models.evaluate

export: ## Export the compact model to JSON for the in-browser predictor.
	uv run python -m pyrochrome.models.export

# --- quality gates -----------------------------------------------------------
lint: ## Run ruff linter.
	uv run ruff check src tests

format: ## Auto-format with black + ruff import sorting.
	uv run black src tests
	uv run ruff check --fix src tests

typecheck: ## Run mypy.
	uv run mypy

test: ## Run the test suite.
	uv run pytest

check: lint typecheck test ## Run all quality gates (what CI runs).

clean: ## Remove caches and build artifacts.
	rm -rf .mypy_cache .ruff_cache .pytest_cache **/__pycache__ models_out

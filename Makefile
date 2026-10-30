.DEFAULT_GOAL := help
.PHONY: help init sync lock add-dev lint format test build validate \
        deploy-dev connect-test clean

# ---- setup -----------------------------------------------------------

init: ## First-time setup: pin Python, sync deps
	uv python pin 3.12.3
	uv sync --all-groups
	@echo "Done. Run 'make connect-test' once you've configured auth (see README)."

sync: ## Re-sync the venv to match pyproject.toml / uv.lock exactly
	uv sync --all-groups --locked

lock: ## Re-resolve and write uv.lock after changing pyproject.toml
	uv lock

add-dev: ## Example: make add-dev PKG="some-package>=1.0"
	uv add --dev $(PKG)

# ---- code quality ------------------------------------------------------

lint: ## Static checks, no fixes
	uv run ruff check .

format: ## Auto-fix formatting + lint issues
	uv run ruff format .
	uv run ruff check . --fix

test: ## Run the test suite
	uv run pytest tests/ -v

# ---- build / bundle -----------------------------------------------------

build: ## Build the wheel locally (same command DAB runs on deploy)
	uv build --wheel

validate: ## Validate the bundle against dev without deploying
	databricks bundle validate -t dev

deploy-dev: ## Deploy this project to YOUR dev identity (not CI/CD)
	databricks bundle deploy -t dev
run-train-dev: ## Trigger train_job in dev and tail the run
	databricks bundle run train_job -t dev

run-score-dev: ## Trigger score_job in dev and tail the run
	databricks bundle run score_job -t dev

# ---- local debugging against a real cluster ------------------------------

connect-test: ## Smoke-test Databricks Connect (set DATABRICKS_SERVERLESS_COMPUTE_ID=auto, or DATABRICKS_CLUSTER_ID for classic)
	uv run python scripts/connect_test.py

clean: ## Remove build/test artifacts
	rm -rf dist build .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

help: ## Show this list
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

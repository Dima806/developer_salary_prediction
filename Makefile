.PHONY: lint format test coverage complexity maintainability audit security \
        tune pre-process train app smoke-test guardrails check all

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

coverage:
	uv run pytest --cov=src --cov-report=term-missing

complexity:
	uv run radon cc . -a -s -nb

maintainability:
	uv run radon mi . -s

audit:
	uv run pip-audit

# --severity-level medium: only MEDIUM/HIGH severity fails the build.
# LOW severity findings (e.g. B403 pickle import) are suppressed
# regardless of their confidence level.
security:
	uv run bandit -r . -x ./.venv,./tests --severity-level medium

tune:
	uv run python -m src.tune

# Requires data/survey_results_public.csv
# Validates columns, filters salaries, reduces cardinality, and writes
# config/valid_categories.yaml and config/currency_rates.yaml
pre-process:
	uv run python -m src.preprocess

# Requires data/survey_results_public.csv (run pre-process first)
train:
	uv run python -m src.train

# Requires a trained model (run `make train` first)
app:
	uv run streamlit run app.py

smoke-test:
	uv run python example_inference.py

# Requires training data and a trained model
guardrails:
	uv run python guardrail_evaluation.py

# CI gate: fast checks that require no model or training data
check: lint test complexity maintainability audit security

# Complete workflow: quality checks → pre-process data → train → evaluate
all: format lint test coverage complexity maintainability audit security pre-process train smoke-test guardrails

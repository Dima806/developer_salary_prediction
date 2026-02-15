.PHONY: lint format test coverage complexity maintainability audit security check all

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

security:
	uv run bandit -r . -x ./.venv

check: lint test complexity maintainability audit security

all: check

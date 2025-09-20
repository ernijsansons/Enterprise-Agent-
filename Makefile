setup:
	poetry install
	npm init -y && npm i -D @openai/codex
	# SAFETY: Verify CLI without API call
	npx codex --version

lint:
	poetry run black .
	poetry run isort .
	poetry run ruff check .

test:
	poetry run pytest tests/ -v --cov=src --cov-report=html

typecheck:
	poetry run mypy src/

format:
	poetry run black .
	poetry run isort .

run:
	poetry run python src/agent_orchestrator.py $(DOMAIN) --input "$(INPUT)"

bench:
	poetry run pytest tests/ --benchmark-only

export:
	poetry run python -c "from src.utils.safety import scrub_pii; import json; print(json.dumps({'logs': 'anonymized'}, indent=2))"

ci: lint test typecheck bench

.PHONY: setup lint test typecheck format run bench ci export
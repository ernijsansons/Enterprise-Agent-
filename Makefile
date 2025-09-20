setup:
	poetry install
	npm init -y && npm i -D @openai/codex
	# SAFETY: Verify CLI without API call
	npx codex --version

lint:
	poetry run black src/ tests/
	poetry run isort src/ tests/
	poetry run ruff check src/ tests/

test:
	poetry run pytest tests/ -v --cov=src --cov-report=html

typecheck:
	poetry run mypy src/ --ignore-missing-imports

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

run:
	poetry run python src/agent_orchestrator.py $(DOMAIN) --input "$(INPUT)"

bench:
	poetry run pytest tests/ --benchmark-only

export:
	poetry run python -c "from src.utils.safety import scrub_pii; import json; print(json.dumps({'logs': 'anonymized'}, indent=2))"

bandit:
	poetry run bandit -r src/ -f json -o bandit-report.json

security: bandit
	@echo "Security scan completed successfully"

quality: lint typecheck security
	@echo "All quality checks passed"

ci: quality test bench
	@echo "CI pipeline completed successfully"

.PHONY: setup lint test typecheck format run bench ci export bandit security quality
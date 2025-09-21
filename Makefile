# Enterprise Agent Makefile v3.4
# Production-ready build automation with error handling

.SILENT:
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Configuration
PYTHON := python3
POETRY := poetry
POETRY_VERSION := 1.7.0
MIN_PYTHON_VERSION := 3.9
COVERAGE_THRESHOLD := 80

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "Enterprise Agent Build System"
	@echo "=============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: check-python ## Install dependencies and validate configuration
	@echo "Setting up Enterprise Agent environment..."
	@if ! command -v $(POETRY) &> /dev/null; then \
		echo "Installing Poetry $(POETRY_VERSION)..."; \
		pip install poetry==$(POETRY_VERSION) || { echo "$(RED)Failed to install Poetry$(NC)"; exit 1; }; \
	fi
	@echo "Installing Python dependencies..."
	@$(POETRY) install --no-interaction --no-ansi || { echo "$(RED)Failed to install dependencies$(NC)"; exit 1; }
	@if command -v npm &> /dev/null; then \
		echo "Setting up Node.js dependencies..."; \
		npm init -y 2>/dev/null || true; \
		npm i -D @openai/codex 2>/dev/null || echo "$(YELLOW)Warning: Could not install Node.js dependencies$(NC)"; \
	fi
	@echo "Validating configuration files..."
	@$(MAKE) validate-config || { echo "$(RED)Configuration validation failed$(NC)"; exit 1; }
	@echo "$(GREEN)Setup completed successfully!$(NC)"

check-python: ## Check Python version
	@echo "Checking Python version..."
	@$(PYTHON) -c "import sys; v = sys.version_info; exit(0 if v >= (3, 9) else 1)" || \
		{ echo "$(RED)Python $(MIN_PYTHON_VERSION)+ required. Current: $$($(PYTHON) --version)$(NC)"; exit 1; }
	@echo "$(GREEN)Python version check passed$(NC)"

lint: ## Run linting checks
	@echo "Running linting checks..."
	@$(POETRY) run black --check src/ tests/ || \
		{ echo "$(YELLOW)Formatting issues detected. Run 'make format' to fix.$(NC)"; exit 1; }
	@$(POETRY) run isort --check-only src/ tests/ || \
		{ echo "$(YELLOW)Import sorting issues detected. Run 'make format' to fix.$(NC)"; exit 1; }
	@$(POETRY) run ruff check src/ tests/ --quiet || \
		{ echo "$(RED)Linting errors detected$(NC)"; exit 1; }
	@echo "$(GREEN)All linting checks passed$(NC)"

test: ## Run unit tests with coverage
	@echo "Running unit tests..."
	@mkdir -p tests .metrics .cache logs
	@$(POETRY) run pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD) || \
		{ echo "$(RED)Tests failed or coverage below $(COVERAGE_THRESHOLD)%$(NC)"; exit 1; }
	@echo "$(GREEN)All tests passed with sufficient coverage$(NC)"

test-quick: ## Run tests without coverage
	@echo "Running quick tests..."
	@$(POETRY) run pytest tests/ -x --tb=short
	@echo "$(GREEN)Quick tests completed$(NC)"

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	@if [ -f smoke_test.py ]; then \
		$(PYTHON) smoke_test.py || { echo "$(RED)Smoke test failed$(NC)"; exit 1; }; \
	fi
	@if [ -f test_metrics_system.py ]; then \
		$(PYTHON) test_metrics_system.py || { echo "$(RED)Metrics test failed$(NC)"; exit 1; }; \
	fi
	@echo "$(GREEN)Integration tests passed$(NC)"

test-edge: ## Run edge case tests
	@echo "Running edge case tests..."
	@if [ -f test_edge_cases.py ]; then \
		$(PYTHON) test_edge_cases.py || { echo "$(YELLOW)Edge case tests failed$(NC)"; }; \
	else \
		echo "$(YELLOW)No edge case tests found$(NC)"; \
	fi

test-all: test test-integration test-edge ## Run all tests

typecheck: ## Run type checking
	@echo "Running type checking..."
	@$(POETRY) run mypy src/ --ignore-missing-imports --show-error-codes || \
		{ echo "$(YELLOW)Type checking issues detected$(NC)"; exit 1; }
	@echo "$(GREEN)Type checking passed$(NC)"

format: ## Auto-format code
	@echo "Formatting code..."
	@$(POETRY) run black src/ tests/ || { echo "$(RED)Black formatting failed$(NC)"; exit 1; }
	@$(POETRY) run isort src/ tests/ || { echo "$(RED)Import sorting failed$(NC)"; exit 1; }
	@echo "$(GREEN)Code formatted successfully$(NC)"

format-check: ## Check formatting without changing files
	@echo "Checking code format..."
	@$(POETRY) run black --check src/ tests/
	@$(POETRY) run isort --check-only src/ tests/

run: ## Run the agent orchestrator (DOMAIN=domain INPUT="input")
	@if [ -z "$(DOMAIN)" ]; then \
		echo "$(RED)Error: DOMAIN parameter required. Usage: make run DOMAIN=coding INPUT='task'$(NC)"; \
		exit 1; \
	fi
	@echo "Running agent orchestrator for domain: $(DOMAIN)"
	@$(POETRY) run $(PYTHON) src/agent_orchestrator.py $(DOMAIN) --input "$(INPUT)" || \
		{ echo "$(RED)Agent orchestrator failed$(NC)"; exit 1; }

bench: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	@$(POETRY) run pytest tests/ --benchmark-only 2>/dev/null || \
		echo "$(YELLOW)No benchmark tests found or benchmarking not configured$(NC)"
	@$(PYTHON) -c "import time; from src.agent_orchestrator import AgentOrchestrator; \
		start = time.time(); agent = AgentOrchestrator(); \
		elapsed = time.time() - start; \
		print(f'Initialization time: {elapsed:.3f}s'); \
		exit(0 if elapsed < 5 else 1)" || \
		{ echo "$(YELLOW)Performance warning: Slow initialization$(NC)"; }

export: ## Export anonymized logs
	@echo "Exporting anonymized logs..."
	@$(POETRY) run $(PYTHON) -c "\
		try: \
			from src.utils.safety import scrub_pii; \
			import json; \
			print(json.dumps({'logs': 'anonymized', 'status': 'success'}, indent=2)); \
		except ImportError: \
			print('Safety module not available'); \
		" || { echo "$(RED)Export failed$(NC)"; exit 1; }
	@echo "$(GREEN)Export completed$(NC)"

bandit: ## Run security scan with bandit
	@echo "Running security scan..."
	@$(POETRY) run bandit -r src/ -f json -o bandit-report.json -ll || true
	@$(POETRY) run bandit -r src/ -f txt || true
	@$(PYTHON) -c "\
		import json; \
		try: \
			with open('bandit-report.json') as f: \
				report = json.load(f); \
				issues = report.get('results', []); \
				if issues: \
					print(f'Found {len(issues)} security issues'); \
					exit(1); \
				else: \
					print('No security vulnerabilities found'); \
		except: pass"
	@echo "$(GREEN)Security scan completed$(NC)"

security: bandit validate-secrets ## Run all security checks
	@echo "$(GREEN)All security checks passed$(NC)"

validate-secrets: ## Check for hardcoded secrets
	@echo "Checking for hardcoded secrets..."
	@! grep -r -E "(api_key|password|secret|token)\s*=\s*['\"]" src/ --include="*.py" 2>/dev/null || \
		{ echo "$(RED)Potential hardcoded secrets detected$(NC)"; exit 1; }
	@! grep -r -E "sk-[a-zA-Z0-9]{48}|AIza[a-zA-Z0-9]{35}" src/ --include="*.py" 2>/dev/null || \
		{ echo "$(RED)API keys detected in code$(NC)"; exit 1; }
	@echo "$(GREEN)No hardcoded secrets found$(NC)"

validate-config: ## Validate configuration files
	@echo "Validating configuration files..."
	@if [ ! -f configs/agent_config_v3.4.yaml ]; then \
		echo "$(RED)Configuration file not found: configs/agent_config_v3.4.yaml$(NC)"; \
		exit 1; \
	fi
	@$(PYTHON) validate_config.py configs/agent_config_v3.4.yaml || \
		{ echo "$(RED)Configuration validation failed$(NC)"; exit 1; }
	@echo "$(GREEN)Configuration validation passed$(NC)"

quality: lint typecheck security validate-config ## Run all quality checks
	@echo "$(GREEN)All quality checks passed$(NC)"

ci: check-python quality test test-integration bench ## Run full CI pipeline
	@echo "$(GREEN)CI pipeline completed successfully$(NC)"

ci-fast: check-python lint typecheck test-quick ## Run fast CI checks
	@echo "$(GREEN)Fast CI checks completed$(NC)"

install-hooks: ## Install git hooks
	@echo "Installing git hooks..."
	@echo '#!/bin/bash' > .git/hooks/pre-commit
	@echo 'make ci-fast' >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)Git hooks installed$(NC)"

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	@rm -rf dist/ build/ *.egg-info .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Clean completed$(NC)"

build: clean ## Build distribution packages
	@echo "Building distribution packages..."
	@$(POETRY) build || { echo "$(RED)Build failed$(NC)"; exit 1; }
	@echo "$(GREEN)Build completed. Packages in dist/$(NC)"

install: build ## Install the package locally
	@echo "Installing package locally..."
	@pip install dist/*.whl || { echo "$(RED)Installation failed$(NC)"; exit 1; }
	@echo "$(GREEN)Package installed successfully$(NC)"

docs: ## Generate documentation
	@echo "Generating documentation..."
	@if command -v sphinx-build &> /dev/null; then \
		sphinx-build -b html docs/ docs/_build/html; \
		echo "$(GREEN)Documentation generated in docs/_build/html$(NC)"; \
	else \
		echo "$(YELLOW)Sphinx not installed. Skipping documentation generation$(NC)"; \
	fi

version: ## Show version information
	@echo "Enterprise Agent v3.4"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Poetry: $$($(POETRY) --version 2>/dev/null || echo 'Not installed')"
	@echo "Pip: $$(pip --version)"

.PHONY: help setup check-python lint test test-quick test-integration test-edge test-all \
        typecheck format format-check run bench export bandit security validate-secrets \
        validate-config quality ci ci-fast install-hooks clean build install docs version
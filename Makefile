# Isaac Sim MCP Server Makefile

.PHONY: help install install-dev test test-cov lint format clean build docs run-server run-headless setup-isaac

# Default target
help:
	@echo "Simul MCP Server Development Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install       Install package and dependencies"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  setup-isaac   Setup Isaac Sim environment integration"
	@echo ""
	@echo "Development Commands:"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Run linting with flake8 and mypy"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  clean         Clean build artifacts and cache"
	@echo ""
	@echo "Build Commands:"
	@echo "  build         Build package"
	@echo "  docs          Generate documentation"
	@echo ""
	@echo "Run Commands:"
	@echo "  run-server    Run MCP server in development mode"
	@echo "  run-headless  Run MCP server in headless mode"
	@echo "  run-isaac     Run with Isaac Sim integration"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

setup-isaac:
	@echo "Setting up Isaac Sim environment..."
	@if [ -z "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: ISAAC_SIM_PATH environment variable not set"; \
		echo "Please set ISAAC_SIM_PATH to your Isaac Sim installation directory"; \
		exit 1; \
	fi
	@echo "Isaac Sim path: $(ISAAC_SIM_PATH)"
	@if [ ! -d "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: Isaac Sim directory not found at $(ISAAC_SIM_PATH)"; \
		exit 1; \
	fi
	@echo "Isaac Sim environment setup complete"

# Development
format:
	black src/ tests/ examples/
	isort src/ tests/ examples/

lint:
	flake8 src/ tests/
	mypy src/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=simul_mcp --cov-report=html --cov-report=term-missing

test-isaac:
	pytest tests/ -v -m isaac

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build

docs:
	@echo "Documentation generation not yet implemented"

# Running
run-server:
	python -m simul_mcp.cli.main server --config config/isaac/default.yaml

run-headless:
	python -m simul_mcp.cli.main server --config config/isaac/default.yaml

run-isaac:
	@if [ -z "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: ISAAC_SIM_PATH not set. Run 'make setup-isaac' first"; \
		exit 1; \
	fi
	$(ISAAC_SIM_PATH)/python.sh -m simul_mcp.cli.main server --config config/isaac/default.yaml

# Development utilities
dev-setup: install-dev setup-isaac
	@echo "Development environment setup complete"

check: format lint test
	@echo "All checks passed!"

# Isaac Sim specific commands
isaac-shell:
	@if [ -z "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: ISAAC_SIM_PATH not set"; \
		exit 1; \
	fi
	$(ISAAC_SIM_PATH)/python.sh

isaac-test:
	@if [ -z "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: ISAAC_SIM_PATH not set"; \
		exit 1; \
	fi
	$(ISAAC_SIM_PATH)/python.sh -m pytest tests/ -v -m isaac

# Extension development
ext-install:
	@echo "Installing Isaac Sim extension..."
	@if [ -z "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: ISAAC_SIM_PATH not set"; \
		exit 1; \
	fi
	cp -r exts/khemoo.simul.mcp $(ISAAC_SIM_PATH)/exts/

ext-uninstall:
	@echo "Uninstalling Isaac Sim extension..."
	@if [ -z "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: ISAAC_SIM_PATH not set"; \
		exit 1; \
	fi
	rm -rf $(ISAAC_SIM_PATH)/exts/khemoo.simul.mcp

# Debugging
debug-server:
	python -m debugpy --listen 5678 --wait-for-client -m simul_mcp.cli.main server

debug-isaac:
	@if [ -z "$(ISAAC_SIM_PATH)" ]; then \
		echo "Error: ISAAC_SIM_PATH not set"; \
		exit 1; \
	fi
	$(ISAAC_SIM_PATH)/python.sh -m debugpy --listen 5678 --wait-for-client -m simul_mcp.cli.main server --config config/isaac/default.yaml

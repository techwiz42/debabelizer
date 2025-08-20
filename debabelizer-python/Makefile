.PHONY: help build develop test clean lint format check-all

help:
	@echo "Available commands:"
	@echo "  build     - Build the Python wheel"
	@echo "  develop   - Install in development mode"
	@echo "  test      - Run tests"
	@echo "  clean     - Clean build artifacts"
	@echo "  lint      - Run linting"
	@echo "  format    - Format code"
	@echo "  check-all - Run all checks (format, lint, test)"

build:
	maturin build --release

develop:
	maturin develop

test:
	python -m pytest tests/

clean:
	cargo clean
	rm -rf target/
	rm -rf python/debabelizer.egg-info/
	rm -rf python/debabelizer/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

lint:
	ruff check python/

format:
	black python/
	ruff format python/

check-all: format lint test
	@echo "All checks passed!"
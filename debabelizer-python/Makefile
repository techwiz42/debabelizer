.PHONY: help build build-release install install-dev test test-release clean format lint check-all dev-install publish

help:
	@echo "Available commands:"
	@echo "  build         - Build the extension module in debug mode"
	@echo "  build-release - Build the extension module in release mode"
	@echo "  install       - Install in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  test          - Run Python tests"
	@echo "  test-release  - Run tests with release build"
	@echo "  clean         - Clean build artifacts"
	@echo "  format        - Format Python code with black"
	@echo "  lint          - Lint Python code with ruff"
	@echo "  check-all     - Run all checks (format, lint, test)"
	@echo "  dev-install   - Full development setup"
	@echo "  publish       - Publish to PyPI (requires authentication)"

build:
	maturin develop

build-release:
	maturin develop --release

install: build

install-dev: install
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-release: build-release test

clean:
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf build/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

format:
	black python/ tests/ examples/

lint:
	ruff check python/ tests/ examples/

check-all: format lint test-release

dev-install: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

publish: build-release
	maturin build --release
	maturin publish
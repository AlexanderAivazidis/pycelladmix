# pycelladmix dev tasks. Run `just <task>` or `just --list`.

# Default: list available tasks
default:
    @just --list

# Sync dependencies with uv (creates .venv if missing)
sync:
    uv sync --all-extras

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run tests with coverage report
cov:
    uv run pytest --cov=pycelladmix --cov-report=term-missing --cov-report=html

# Lint with ruff (no fixes)
lint:
    uv run ruff check src tests

# Format with ruff
format:
    uv run ruff format src tests
    uv run ruff check --fix src tests

# Check formatting without modifying
format-check:
    uv run ruff format --check src tests

# Build the docs site locally
docs:
    uv run sphinx-build -b html docs docs/_build/html

# Live-reload docs server
docs-serve:
    uv run sphinx-autobuild docs docs/_build/html

# Build wheel + sdist
build:
    uv build

# Clean build / cache artefacts
clean:
    rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov docs/_build
    find . -type d -name __pycache__ -exec rm -rf {} +

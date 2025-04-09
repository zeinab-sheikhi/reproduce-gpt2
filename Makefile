.PHONY: install-uv install

install-uv:
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv ..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv is already installed!"; \
	fi

install: install-uv
	uv sync --all-extras --python 3.11

lint:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy .

pre-commit-checks: lint typecheck

clean:
	rm -rf .mypy_cache __pycache__ .ruff_cache
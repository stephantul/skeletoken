install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest --cov=skeletoken --cov-report=term-missing

install-no-pre-commit:
	uv sync --all-extras

type-check:
	uv run mypy skeletoken
	uv run ty check skeletoken

lint:
	uv run ruff check skeletoken
	uv run ruff format skeletoken

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest --cov=skeletoken --cov-report=term-missing

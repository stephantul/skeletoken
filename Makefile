install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest --cov=skeletoken --cov-report=term-missing

test-integration:
	# model2vec has no conflicting pins so it's a real "integration" extra (see pyproject.toml).
	# torch/sentence-transformers/pylate DO force a transformers downgrade if added the same way
	# (sentence-transformers/pylate cap transformers well below what skeletoken needs), so they're
	# installed ephemerally here instead, keeping the main dependency resolution untouched.
	uv run --with torch --with sentence-transformers --with pylate pytest tests/integration

install-no-pre-commit:
	uv sync --all-extras

type-check:
	uv run mypy skeletoken
	uv run ty check skeletoken

lint:
	uv run ruff check skeletoken
	uv run ruff format skeletoken

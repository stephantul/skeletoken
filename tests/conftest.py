import json
from typing import Any

from pytest import fixture


@fixture(scope="function")
def small_tokenizer_json() -> dict[str, Any]:
    """Load an extremely small tokenizer for testing purposes."""
    return json.load(open("tests/data/small_tokenizer.json", "r", encoding="utf-8"))

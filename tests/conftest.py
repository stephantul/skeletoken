import json
from typing import Any

from pytest import fixture
from tokenizers import Tokenizer


@fixture(scope="function")
def small_tokenizer_json() -> dict[str, Any]:
    """Load an extremely small tokenizer for testing purposes."""
    return json.load(open("tests/data/small_tokenizer.json", "r", encoding="utf-8"))


@fixture(scope="module")
def small_tokenizer() -> Tokenizer:
    """Load an extremely small tokenizer for testing purposes."""
    return Tokenizer.from_file("tests/data/small_tokenizer.json")

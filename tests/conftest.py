import json
from typing import Any

from pytest import fixture
from tokenizers import Tokenizer

from skeletoken.merges import Merges


@fixture(scope="function")
def small_tokenizer_json() -> dict[str, Any]:
    """Load an extremely small tokenizer for testing purposes."""
    return json.load(open("tests/data/small_tokenizer.json", "r", encoding="utf-8"))


@fixture(scope="module")
def small_tokenizer() -> Tokenizer:
    """Load an extremely small tokenizer for testing purposes."""
    return Tokenizer.from_file("tests/data/small_tokenizer.json")


@fixture(scope="function")
def small_merges() -> Merges:
    """Load an extremely small merges for testing purposes."""
    return Merges([("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc")])

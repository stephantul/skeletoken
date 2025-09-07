import json
from typing import Any

from pytest import FixtureRequest, fixture
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from skeletoken.merges import Merges


@fixture(scope="function")
def small_tokenizer_json(request: FixtureRequest) -> dict[str, Any]:
    """Load an extremely small tokenizer for testing purposes."""
    name = getattr(request, "param", None) or "bpe"
    return json.load(open(f"tests/data/{name}/tokenizer.json", "r", encoding="utf-8"))


@fixture(scope="module")
def small_tokenizer(request: FixtureRequest) -> Tokenizer:
    """Load an extremely small tokenizer for testing purposes."""
    name = getattr(request, "param", None) or "bpe"
    return Tokenizer.from_file(f"tests/data/{name}/tokenizer.json")


@fixture(scope="function")
def small_merges() -> Merges:
    """Load an extremely small set of merges for testing purposes."""
    return Merges([("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc")])


@fixture(scope="function")
def transformers_tokenizer(request: FixtureRequest) -> PreTrainedTokenizerFast:
    """Load a small transformers tokenizer for testing purposes."""
    name = getattr(request, "param", None) or "bpe"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(f"tests/data/{name}")
    return tokenizer

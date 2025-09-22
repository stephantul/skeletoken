import json
from pathlib import Path
from typing import Any

from pytest import FixtureRequest, fixture
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from skeletoken.merges import Merges


def _get_path(name: str) -> Path:
    """Gets the path to a test resource."""
    return Path("tests") / "data" / name


def _get_tokenizers_path(name: str) -> str:
    """Gets the path to a test tokenizer resource."""
    return str(_get_path(name) / "tokenizer.json")


@fixture(scope="function")
def small_tokenizer_json(request: FixtureRequest) -> dict[str, Any]:
    """Load an extremely small tokenizer for testing purposes."""
    name = getattr(request, "param", None) or "wordpiece"
    return json.load(open(_get_tokenizers_path(name), "r", encoding="utf-8"))


@fixture(scope="function")
def small_tokenizer(request: FixtureRequest) -> Tokenizer:
    """Load an extremely small tokenizer for testing purposes."""
    name = getattr(request, "param", None) or "wordpiece"
    return Tokenizer.from_file(_get_tokenizers_path(name))


@fixture(scope="function")
def small_merges() -> Merges:
    """Load an extremely small set of merges for testing purposes."""
    return Merges([("a", "b"), ("c", "d"), ("ab", "c"), ("a", "bc")])


@fixture(scope="function")
def transformers_tokenizer(request: FixtureRequest) -> PreTrainedTokenizerFast:
    """Load a small transformers tokenizer for testing purposes."""
    name = getattr(request, "param", None) or "wordpiece"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(_get_path(name))
    return tokenizer

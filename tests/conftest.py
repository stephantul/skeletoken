import json
from pathlib import Path
from typing import Any

from pytest import FixtureRequest, fixture
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from skeletoken.base import TokenizerModel
from skeletoken.merges import Merges
from skeletoken.postprocessors import TemplatePostProcessor


def call_tokenizer(model: TokenizerModel) -> None:
    """Call the tokenizer to ensure the model is correct."""
    tokenizer = model.to_tokenizer()
    tokenizer.encode("some text")


def assert_vocabulary_consistent(model: TokenizerModel) -> None:
    """Assert that the model's vocabulary is internally consistent."""
    vocab = model.vocabulary
    size = model.vocabulary_size
    # IDs form a contiguous range [0, size)
    assert sorted(vocab.values()) == list(range(size))
    assert len(model.sorted_vocabulary) == size
    # Every added token lives in the vocabulary with a matching ID
    for token in model.added_tokens.root:
        assert token.content in vocab, f"Added token {token.content!r} missing from vocabulary"
        assert vocab[token.content] == token.id, (
            f"Added token {token.content!r} has id={token.id} but vocabulary says {vocab[token.content]}"
        )
    # Special token IDs are self-consistent
    if model.unk_token:
        assert model.unk_token in vocab
        assert vocab[model.unk_token] == model.unk_token_id
    if model.pad_token:
        assert model.pad_token in vocab
        assert vocab[model.pad_token] == model.pad_token_id


def _get_path(name: str) -> Path:
    """Get the path to a test resource."""
    return Path("tests") / "data" / name


def _get_tokenizers_path(name: str) -> str:
    """Get the path to a test tokenizer resource."""
    return str(_get_path(name) / "tokenizer.json")


@fixture(scope="function")
def small_tokenizer_json(request: FixtureRequest) -> dict[str, Any]:
    """Load an extremely small tokenizer for testing purposes."""
    name = getattr(request, "param", None) or "wordpiece"
    return json.load(open(_get_tokenizers_path(name), encoding="utf-8"))


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


@fixture(scope="function")
def template_post_processor() -> TemplatePostProcessor:
    """Get a template post processor."""
    template_json = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [50281], "tokens": ["[CLS]"]},
            "[MASK]": {"id": "[MASK]", "ids": [50284], "tokens": ["[MASK]"]},
            "[PAD]": {"id": "[PAD]", "ids": [50283], "tokens": ["[PAD]"]},
            "[SEP]": {"id": "[SEP]", "ids": [50282], "tokens": ["[SEP]"]},
            "[UNK]": {"id": "[UNK]", "ids": [50280], "tokens": ["[UNK]"]},
        },
    }

    return TemplatePostProcessor.model_validate(template_json)

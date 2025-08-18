from math import log
from typing import Any, Literal, overload

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.models import BPE, Model, ModelType, Unigram, WordLevel, WordPiece


@overload
def _get_default_model(model_type: Literal[ModelType.UNIGRAM]) -> Unigram: ...


@overload
def _get_default_model(model_type: Literal[ModelType.BPE]) -> BPE: ...


@overload
def _get_default_model(model_type: Literal[ModelType.WORDLEVEL]) -> WordLevel: ...


@overload
def _get_default_model(model_type: Literal[ModelType.WORDPIECE]) -> WordPiece: ...


@overload
def _get_default_model(model_type: ModelType) -> Model: ...


def _get_default_model(model_type: ModelType) -> Model:
    """Helper function to get the default instantiation of a model."""
    vocab = {
        "[PAD]": 0,
        "[SEP]": 1,
        "[UNK]": 2,
        "[CLS]": 3,
        "[MASK]": 4,
        "a": 5,
        "b": 6,
        "c": 7,
        "d": 8,
        "e": 9,
        " ": 10,
    }
    if model_type == ModelType.BPE:
        return BPE(
            vocab=vocab,
            merges=[],
            dropout=0.1,
            unk_token="[UNK]",
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=False,
        )
    elif model_type == ModelType.WORDPIECE:
        return WordPiece(vocab=vocab, unk_token="[UNK]", continuing_subword_prefix="", max_input_chars_per_word=100)
    elif model_type == ModelType.UNIGRAM:
        p = log(1.0 / len(vocab))
        u_vocab = [(x, p) for x, _ in sorted(vocab.items(), key=lambda item: item[1], reverse=True)]
        return Unigram(vocab=u_vocab, unk_id=2, byte_fallback=False)
    elif model_type == ModelType.WORDLEVEL:
        return WordLevel(vocab=vocab, unk_token="[UNK]")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@pytest.mark.parametrize("model_type", [ModelType.BPE, ModelType.WORDPIECE, ModelType.UNIGRAM, ModelType.WORDLEVEL])
def test_model(small_tokenizer_json: dict[str, Any], model_type: ModelType) -> None:
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    model = _get_default_model(model_type)
    model_dict = model.model_dump()
    small_tokenizer_json["model"] = model_dict
    tokenizer = TokenizerModel.model_validate(small_tokenizer_json)

    assert tokenizer.model is not None
    assert tokenizer.model.type == model_type
    assert isinstance(tokenizer.model, model.__class__)


def _get_none_unigram() -> Unigram:
    """A unigram model with UNK_id set to None."""
    model = _get_default_model(ModelType.UNIGRAM)
    model.unk_id = None

    return model


@pytest.mark.parametrize("model", [*[_get_default_model(x) for x in ModelType], _get_none_unigram()])
def test_greedy(model: Model) -> None:
    """Tests the greedy behavior."""
    model = model.to_greedy()
    assert model.type == ModelType.WORDPIECE

    tokenizer_model = TokenizerModel(model=model)
    tokenizer = tokenizer_model.to_tokenizer()

    assert tokenizer.encode("a b c d e").tokens == ["a", " ", "b", " ", "c", " ", "d", " ", "e"]

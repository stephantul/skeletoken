from math import log
from typing import Any

import pytest

from tokenizerdatamodels.base import TokenizerModel
from tokenizerdatamodels.models import BPE, Model, ModelType, Unigram, WordLevel, WordPiece


def _get_default_model(model_type: ModelType) -> Model:
    """Helper function to get the default instantiation of a model."""
    vocab = {"[PAD]": 0, "[SEP]": 1, "[UNK]": 2, "[CLS]": 3, "[MASK]": 4, "a": 5, "b": 6, "c": 7, "d": 8, "e": 9}
    if model_type == ModelType.BPE:
        return BPE(
            vocab=vocab,
            merges=[],
            dropout=0.1,
            unk_token="[UNK]",
            continuing_subword_prefix="##",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=False,
        )
    elif model_type == ModelType.WORDPIECE:
        return WordPiece(vocab=vocab, unk_token="[UNK]", continuing_subword_prefix="##", max_input_chars_per_word=100)
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

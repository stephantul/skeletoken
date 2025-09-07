from math import log
from typing import Any, Literal, overload

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.merges import Merges
from skeletoken.models import BPE, Model, ModelType, Unigram, WordLevel, WordPiece, get_subword_prefix_token
from skeletoken.vocabulary import UnigramVocabulary, Vocabulary


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
            vocab=Vocabulary(vocab),
            merges=Merges([]),
            dropout=0.1,
            unk_token="[UNK]",
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=False,
        )
    elif model_type == ModelType.WORDPIECE:
        return WordPiece(
            vocab=Vocabulary(vocab), unk_token="[UNK]", continuing_subword_prefix="", max_input_chars_per_word=100
        )
    elif model_type == ModelType.UNIGRAM:
        p = log(1.0 / len(vocab))
        u_vocab = [(x, p) for x, _ in sorted(vocab.items(), key=lambda item: item[1])]
        return Unigram(vocab=UnigramVocabulary(u_vocab), unk_id=2, byte_fallback=False)
    elif model_type == ModelType.WORDLEVEL:
        return WordLevel(vocab=Vocabulary(vocab), unk_token="[UNK]")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@pytest.mark.parametrize("model_type", [ModelType.BPE, ModelType.WORDPIECE, ModelType.UNIGRAM, ModelType.WORDLEVEL])
def test_model(small_tokenizer_json: dict[str, Any], model_type: ModelType) -> None:
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    internal_model = _get_default_model(model_type)
    model_dict = internal_model.model_dump()
    small_tokenizer_json["model"] = model_dict
    model = TokenizerModel.model_validate(small_tokenizer_json)

    assert model.model is not None
    assert model.model.type == model_type
    assert isinstance(model.model, internal_model.__class__)

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def _get_none_unigram() -> Unigram:
    """A unigram model with UNK_id set to None."""
    model = _get_default_model(ModelType.UNIGRAM)
    model.unk_id = None

    return model


def _get_none_bpe() -> BPE:
    """A BPE model with UNK_token set to None."""
    model = _get_default_model(ModelType.BPE)
    model.unk_token = None

    return model


@pytest.mark.parametrize("model", [*[_get_default_model(x) for x in ModelType], _get_none_unigram(), _get_none_bpe()])
def test_greedy(model: Model) -> None:
    """Tests the greedy behavior."""
    model = model.to_greedy()
    assert model.type == ModelType.WORDPIECE

    tokenizer_model = TokenizerModel(model=model)
    tokenizer = tokenizer_model.to_tokenizer()

    assert tokenizer.encode("a b c d e").tokens == ["a", " ", "b", " ", "c", " ", "d", " ", "e"]


def test_get_subword_prefix_token() -> None:
    """Tests the continuing subword prefix behavior."""
    wordpiece = _get_default_model(ModelType.WORDPIECE)
    wordpiece.continuing_subword_prefix = "##"

    assert get_subword_prefix_token(wordpiece) == "##"

    wordpiece.continuing_subword_prefix = ""
    assert get_subword_prefix_token(wordpiece) == ""

    bpe = _get_default_model(ModelType.BPE)
    bpe.continuing_subword_prefix = "##"

    assert get_subword_prefix_token(bpe) == "##"

    bpe.continuing_subword_prefix = ""
    assert get_subword_prefix_token(bpe) == ""

    unigram = _get_default_model(ModelType.UNIGRAM)
    assert get_subword_prefix_token(unigram) is None

    wordlevel = _get_default_model(ModelType.WORDLEVEL)
    assert get_subword_prefix_token(wordlevel) is None


def test_unk_token_unigram() -> None:
    """Test the unk token in unigram model."""
    model = _get_default_model(ModelType.UNIGRAM)
    assert model.unk_id == 2
    assert model.unk_token == model.vocab.sorted_vocabulary[2]
    model.unk_token = "a"
    assert model.unk_token == "a"
    assert model.unk_id == model.vocab["a"]

    model.unk_token = None
    assert model.unk_id is None
    assert model.unk_token is None


@pytest.mark.parametrize("model", [*[_get_default_model(x) for x in ModelType]])
def test_add_token(model: Model) -> None:
    """Test the add token functionality."""
    model.add_token("new_token", is_added_token=False)
    assert model.vocab["new_token"] == 11
    if isinstance(model, BPE):
        assert model.vocab.sorted_vocabulary == [
            "[PAD]",
            "[SEP]",
            "[UNK]",
            "[CLS]",
            "[MASK]",
            "a",
            "b",
            "c",
            "d",
            "e",
            " ",
            "new_token",
            "_",
            "k",
            "ke",
            "n",
            "ne",
            "new_",
            "new_toke",
            "o",
            "t",
            "to",
            "toke",
            "w",
            "w_",
        ]
        assert model.merges.root == [
            ("n", "e"),
            ("w", "_"),
            ("t", "o"),
            ("k", "e"),
            ("ne", "w_"),
            ("to", "ke"),
            ("new_", "toke"),
            ("new_toke", "n"),
        ]
    else:
        assert model.vocab.sorted_vocabulary == [
            "[PAD]",
            "[SEP]",
            "[UNK]",
            "[CLS]",
            "[MASK]",
            "a",
            "b",
            "c",
            "d",
            "e",
            " ",
            "new_token",
        ]
    with pytest.raises(ValueError):
        model.add_token("new_token")


@pytest.mark.parametrize("model", [*[_get_default_model(x) for x in ModelType]])
def test_replace_token(model: Model) -> None:
    """Test the replace token functionality."""
    model.replace_token("a", "new_token", is_added_token=False)
    if isinstance(model, BPE):
        assert model.vocab.sorted_vocabulary == [
            "[PAD]",
            "[SEP]",
            "[UNK]",
            "[CLS]",
            "[MASK]",
            "new_token",
            "b",
            "c",
            "d",
            "e",
            " ",
            "_",
            "k",
            "ke",
            "n",
            "ne",
            "new_",
            "new_toke",
            "o",
            "t",
            "to",
            "toke",
            "w",
            "w_",
        ]
        assert model.merges.root == [
            ("n", "e"),
            ("w", "_"),
            ("t", "o"),
            ("k", "e"),
            ("ne", "w_"),
            ("to", "ke"),
            ("new_", "toke"),
            ("new_toke", "n"),
        ]
    else:
        assert model.vocab.sorted_vocabulary == [
            "[PAD]",
            "[SEP]",
            "[UNK]",
            "[CLS]",
            "[MASK]",
            "new_token",
            "b",
            "c",
            "d",
            "e",
            " ",
        ]
    with pytest.raises(ValueError):
        model.add_token("new_token")


@pytest.mark.parametrize("model", [*[_get_default_model(x) for x in ModelType]])
def test_remove_token(model: Model) -> None:
    """Test the remove token functionality."""
    model.remove_token("a")
    assert "a" not in model.vocab
    assert model.vocab.sorted_vocabulary == [
        "[PAD]",
        "[SEP]",
        "[UNK]",
        "[CLS]",
        "[MASK]",
        "b",
        "c",
        "d",
        "e",
        " ",
    ]
    with pytest.raises(ValueError):
        model.remove_token("a")

    model.replace_token("b", "x", is_added_token=True)
    # X gets added to the vocabulary, but not as a merge.
    if isinstance(model, BPE):
        assert "x" not in model.merges._all_merge_tokens

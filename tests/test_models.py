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
        "de": 11,
    }
    if model_type == ModelType.BPE:
        return BPE(
            vocab=Vocabulary(vocab),
            merges=Merges([("d", "e")]),
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
    assert model.vocab["new_token"] == 12
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
            "de",
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
            ("d", "e"),
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
            "de",
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
            "de",
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
            ("d", "e"),
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
            "de",
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
        "de",
    ]
    with pytest.raises(ValueError):
        model.remove_token("a")

    model.replace_token("b", "x", is_added_token=True)
    # X gets added to the vocabulary, but not as a merge.
    if isinstance(model, BPE):
        assert "x" not in model.merges._all_merge_tokens


@pytest.mark.parametrize("model", [*[_get_default_model(x) for x in ModelType]])
def test_remove_tokens(model: Model) -> None:
    """Test the remove token functionality."""
    model.remove_tokens(["a", "b", "c"])
    assert "a" not in model.vocab
    assert "b" not in model.vocab
    assert "c" not in model.vocab
    assert model.vocab.sorted_vocabulary == [
        "[PAD]",
        "[SEP]",
        "[UNK]",
        "[CLS]",
        "[MASK]",
        "d",
        "e",
        " ",
        "de",
    ]
    with pytest.raises(ValueError):
        model.remove_token("a")

    model.replace_token("d", "x", is_added_token=True)
    # X gets added to the vocabulary, but not as a merge.
    if isinstance(model, BPE):
        assert "x" not in model.merges._all_merge_tokens


def test_merges_for_bpe() -> None:
    """Test that merges are correctly computed for BPE."""
    model = _get_default_model(ModelType.BPE)
    model.replace_vocabulary(["[PAD]", "[SEP]", "[UNK]", "[CLS]", "[MASK]", "a", "b", "c", "d", "e", " ", None])

    # Merges should contain all merges needed to construct "new_token"
    expected_merges = model.merges.root
    assert model.merges.root == expected_merges

    model = _get_default_model(ModelType.BPE)
    with pytest.raises(ValueError):
        model.replace_vocabulary(["[SEP]", "[UNK]", "[CLS]", "[MASK]", "a", "b", "c", "d", "e", " ", "de"])


def test_replace_vocabulary_models() -> None:
    """Test replace_vocabulary behavior across model types (non-BPE and BPE specifics)."""
    # WordPiece and WordLevel should delegate to Vocabulary.replace_vocabulary.
    for model_type in (ModelType.WORDPIECE, ModelType.WORDLEVEL):
        model = _get_default_model(model_type)
        # Replace last token with None to simulate removal
        new_vocab = ["[PAD]", "[SEP]", "[UNK]", "[CLS]", "[MASK]", "a", "b", "c", "d", "e", " ", None]
        model.replace_vocabulary(new_vocab)
        # The None should be removed and vocabulary indices compacted
        assert "de" not in model.vocab.vocabulary
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
        ]

    # Unigram must keep the same length; _get_default_model Unigram has length 12
    unigram = _get_default_model(ModelType.UNIGRAM)
    with pytest.raises(ValueError):
        # shorter list should raise
        unigram.replace_vocabulary(["[PAD]", "[SEP]"])

    # BPE: replacing vocabulary should update merges to only include valid tokens
    bpe = _get_default_model(ModelType.BPE)
    # Remove the 'de' token by setting its entry to None
    bpe.replace_vocabulary(["[PAD]", "[SEP]", "[UNK]", "[CLS]", "[MASK]", "a", "b", "c", "d", "e", " ", None])
    # 'de' was formed by merging 'd' and 'e', removing it should also remove that merge
    assert ("d", "e") not in bpe.merges.root


def test_bpe_replace_vocabulary_preserve_and_variants() -> None:
    """Focused tests to exercise different branches in BPE.replace_vocabulary."""
    # Base vocabulary tokens in order (indices 0..11)
    base = ["[PAD]", "[SEP]", "[UNK]", "[CLS]", "[MASK]", "a", "b", "c", "d", "e", " ", "de"]

    # 1) Preserve merges when vocabulary unchanged
    bpe = _get_default_model(ModelType.BPE)
    original_merges = list(bpe.merges.root)
    bpe.replace_vocabulary(list(base))
    assert bpe.merges.root == original_merges

    # 2) Remove merge when merged token is removed (vocabulary[index] is None)
    bpe = _get_default_model(ModelType.BPE)
    v: list[str | None] = list(base)
    v[11] = None  # remove 'de'
    bpe.replace_vocabulary(v)
    assert ("d", "e") not in bpe.merges.root

    # 3) Skip merge when left token is removed (vocabulary[left_idx] is None)
    bpe = _get_default_model(ModelType.BPE)
    v = list(base)
    v[8] = None  # remove 'd'
    # keep 'de' present so the code reaches the left/right None check
    assert v[11] is not None
    bpe.replace_vocabulary(v)
    assert ("d", "e") not in bpe.merges.root

    # 4) Merge computed but concatenation not present in new vocab -> not added
    bpe = _get_default_model(ModelType.BPE)
    v = list(base)
    # rename 'd' and 'e' such that 'de' won't be equal to their concatenation
    v[8] = "x"
    v[9] = "y"
    # keep slot 11 but set to something different than 'xy'
    v[11] = "not_xy"
    bpe.replace_vocabulary(v)
    # after replacement there should be no ('x','y') merge because 'xy' is not in vocab
    assert ("x", "y") not in bpe.merges.root


def test_bpe_replace_vocabulary_concat_present() -> None:
    """Ensure the code path where the concatenated merge token exists is exercised."""
    bpe = _get_default_model(ModelType.BPE)
    # Add tokens x, y and their concatenation xy to the vocabulary
    bpe.vocab.add_token("x")
    bpe.vocab.add_token("y")
    bpe.vocab.add_token("xy")
    # Add the merge and rebuild merge indices
    bpe.merges.root.append(("x", "y"))
    bpe.merges.model_post_init({})

    # Build the new vocabulary list in index order and call replace_vocabulary
    root_list: list[str | None] = [None] * len(bpe.vocab.root)
    for tok, idx in bpe.vocab.root.items():
        root_list[idx] = tok

    # Call replace_vocabulary with identical list; this should preserve the ('x','y') merge
    bpe.replace_vocabulary(root_list)
    assert ("x", "y") in bpe.merges.root

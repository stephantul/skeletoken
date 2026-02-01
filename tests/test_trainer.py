from collections.abc import Iterator
from typing import Any

import pytest
from tokenizers import Tokenizer

from skeletoken.base import TokenizerModel
from skeletoken.merges import Merges
from skeletoken.models import BPE, ModelType, Unigram, WordPiece
from skeletoken.trainer import train_tokenizer
from skeletoken.vocabulary import UnigramVocabulary, Vocabulary


def _dummy_data() -> Iterator[str]:
    """
    Small iterator of strings used as fake training data.

    Returns
    -------
    Iterator[str]

    """
    for s in ["hello world", "another sentence"]:
        yield s


@pytest.mark.parametrize("model_type", [ModelType.BPE, ModelType.UNIGRAM, ModelType.WORDPIECE])
def test_train_calls_tokenizer_train_from_iterator(model_type: ModelType, monkeypatch: Any) -> None:
    """
    Ensure train_tokenizer calls Tokenizer.train_from_iterator with a trainer for the given model type.

    We mock Tokenizer.train_from_iterator to be a no-op and assert it was called with a trainer instance.
    """
    # Construct a tiny vocabulary usable for all models
    vocab = {"[PAD]": 0, "[UNK]": 1, "a": 2, " ": 3}

    model: BPE | Unigram | WordPiece
    if model_type == ModelType.BPE:
        model = BPE(
            merges=Merges([]),
            vocab=Vocabulary(vocab),
            dropout=0.0,
            unk_token="[UNK]",
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
            ignore_merges=False,
        )
    elif model_type == ModelType.UNIGRAM:
        u_root = [(t, -1.0) for t in ["[PAD]", "[UNK]", "a", " "]]
        model = Unigram(vocab=UnigramVocabulary(u_root), unk_id=1, byte_fallback=False)
    else:
        model = WordPiece(vocab=Vocabulary(vocab), unk_token="[UNK]", continuing_subword_prefix="")

    tokenizer_model = TokenizerModel(model=model)
    tokenizer = tokenizer_model.to_tokenizer()
    called: dict[str, Any] = {}

    def fake_train_from_iterator(data: Iterator[str], trainer: Any) -> None:
        called["data"] = list(data)
        called["trainer_type"] = type(trainer).__name__

    monkeypatch.setattr(tokenizer, "train_from_iterator", fake_train_from_iterator)

    # Monkeypatch TokenizerModel.to_tokenizer to return our tokenizer
    monkeypatch.setattr(TokenizerModel, "to_tokenizer", lambda self: tokenizer)

    out = train_tokenizer(tokenizer_model, _dummy_data(), vocab_size=100)

    assert "trainer_type" in called
    assert isinstance(out, TokenizerModel)


def test_train_unknown_model_raises() -> None:
    """If model type is unknown, train_tokenizer should raise ValueError."""

    # Create a fake object with the minimal interface used by train_tokenizer
    class Fake:
        def __init__(self) -> None:
            class M:
                type = "Unknown"

            self.model = M()

            class AT:
                def get_special_tokens(self) -> list[Any]:
                    return []

            self.added_tokens = AT()

        def to_tokenizer(self) -> Tokenizer:
            return Tokenizer.from_file("tests/data/wordpiece/tokenizer.json")

    fake = Fake()

    with pytest.raises(ValueError):
        train_tokenizer(fake, _dummy_data(), vocab_size=10)  # type: ignore

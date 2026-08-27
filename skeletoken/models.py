from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from skeletoken.merges import Merges
from skeletoken.vocabulary import UnigramVocabulary, Vocabulary, tokens_ordered_by_id

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    WORDPIECE = "WordPiece"
    BPE = "BPE"
    UNIGRAM = "Unigram"
    WORDLEVEL = "WordLevel"


VocabType = TypeVar("VocabType", Vocabulary, UnigramVocabulary)


class VocabMixinMethod(Generic[VocabType]):
    """Mixin to override token addition, removal etc."""

    vocab: VocabType

    def add_token(self, token: str, is_added_token: bool = False) -> None:
        """Add a token to the vocabulary."""
        self.vocab.add_token(token)

    def replace_token(self, old_token: str, new_token: str, is_added_token: bool = False) -> None:
        """Replace a token in the vocabulary."""
        self.vocab.replace_token(old_token, new_token)

    def remove_token(self, token: str) -> None:
        """Remove a token from the vocabulary."""
        self.vocab.remove_token(token)

    def remove_tokens(self, tokens: Sequence[str]) -> None:
        """Remove multiple tokens from the vocabulary."""
        self.vocab.remove_tokens(tokens)

    def replace_vocabulary(self, vocabulary: list[str | None]) -> None:
        """Completely replaces the vocabulary by a vocabulary of the same length."""
        self.vocab.replace_vocabulary(vocabulary)


class WordPiece(BaseModel, VocabMixinMethod[Vocabulary]):
    """Data model representing a WordPiece vocabulary."""

    type: Literal[ModelType.WORDPIECE] = ModelType.WORDPIECE
    vocab: Vocabulary
    unk_token: str
    continuing_subword_prefix: str
    max_input_chars_per_word: int = 100


class BPE(BaseModel, VocabMixinMethod[Vocabulary]):
    """Data model representing a BPE vocabulary."""

    type: Literal[ModelType.BPE] = ModelType.BPE
    merges: Merges
    vocab: Vocabulary
    dropout: float | None
    unk_token: str | None
    continuing_subword_prefix: str | None
    end_of_word_suffix: str | None
    fuse_unk: bool
    byte_fallback: bool
    ignore_merges: bool

    def add_token(self, token: str, is_added_token: bool = False) -> None:
        """Add a token to the vocabulary."""
        self.vocab.add_token(token)
        if is_added_token:
            return
        self.merges._add_merges_for_token(token)
        new_tokens = sorted(self.merges._all_merge_tokens - set(self.vocab.vocabulary))
        for new_token in new_tokens:
            self.vocab.add_token(new_token)

    def replace_token(self, old_token: str, new_token: str, is_added_token: bool = False) -> None:
        """Replace a token in the vocabulary."""
        self.vocab.replace_token(old_token, new_token)
        # Added tokens do not require merge updates.
        if is_added_token:
            return
        new_tokens = self.merges._add_merges_for_token(new_token)
        for token in new_tokens:
            if token not in self.vocab.vocabulary:
                self.vocab.add_token(token)

    def remove_token(self, token: str) -> None:
        """Remove a token from the vocabulary."""
        self.vocab.remove_token(token)
        self.merges._remove_merges_for_token(token)

    def remove_tokens(self, tokens: Sequence[str]) -> None:
        """Remove multiple tokens from the vocabulary."""
        self.vocab.remove_tokens(tokens)
        self.merges._remove_merges_for_tokens(tokens)

    def replace_vocabulary(self, vocabulary: list[str | None]) -> None:
        """Completely replaces the vocabulary with a vocabulary of the same length."""
        vocab = self.vocab.root
        if len(vocabulary) != len(vocab):
            raise ValueError("New vocabulary must be of the same length as the existing vocabulary.")
        self.vocab.replace_vocabulary(vocabulary)
        self.merges.root = []
        self.merges.model_post_init({})
        tokens = tokens_ordered_by_id(self.vocab.inverse_vocabulary)
        v = set(tokens)

        for token in tokens:
            self.merges._add_merges_for_token(token, vocab=v)


class Unigram(BaseModel, VocabMixinMethod[UnigramVocabulary]):
    """Data model representing a Unigram vocabulary."""

    type: Literal[ModelType.UNIGRAM] = ModelType.UNIGRAM
    vocab: UnigramVocabulary
    unk_id: int | None
    byte_fallback: bool

    def model_post_init(self, __context: dict[Any, Any]) -> None:
        """Check if the unk_id is valid."""
        if self.unk_id is not None and self.unk_id > len(self.vocab.root):
            logger.warning("Unk token ID in model has id larger than vocab size, setting it to None.")
            self.unk_id = None

    @property
    def unk_token(self) -> str | None:
        """Return the unknown token, if any."""
        if self.unk_id is None:
            return None
        return self.vocab.root[self.unk_id][0]

    @unk_token.setter
    def unk_token(self, token: str | None) -> None:
        """Set the unknown token."""
        if token is None:
            self.unk_id = None
        else:
            self.unk_id = self.vocab.vocabulary[token]


class WordLevel(BaseModel, VocabMixinMethod[Vocabulary]):
    """Data model representing a WordLevel vocabulary."""

    type: Literal[ModelType.WORDLEVEL] = ModelType.WORDLEVEL
    vocab: Vocabulary
    unk_token: str


Model = WordPiece | BPE | Unigram | WordLevel
ModelDiscriminator = Annotated[Model, Field(discriminator="type")]


def convert_to_greedy(model: Model) -> WordPiece:
    """Convert a model to a greedy WordPiece model."""
    match model:
        case WordPiece():
            return model
        case BPE():
            if model.unk_token is None:
                logger.warning("BPE model has no unk_token, using the first token in the vocab.")
                unk_token = tokens_ordered_by_id(model.vocab.inverse_vocabulary)[0]
            else:
                unk_token = model.unk_token
            return WordPiece(
                vocab=model.vocab,
                unk_token=unk_token,
                continuing_subword_prefix=model.continuing_subword_prefix or "",
                max_input_chars_per_word=100,
            )
        case Unigram():
            if model.unk_id is None:
                logger.warning("Unigram model has no `unk_id`, using the first token in the vocab.")
                unk_token = model.vocab.root[0][0]
            else:
                unk_token = model.vocab.root[model.unk_id][0]
            return WordPiece(
                vocab=Vocabulary({token: idx for idx, (token, _) in enumerate(model.vocab.root)}),
                unk_token=unk_token,
                continuing_subword_prefix="",
                max_input_chars_per_word=100,
            )
        case WordLevel():
            return WordPiece(
                vocab=model.vocab,
                unk_token=model.unk_token,
                continuing_subword_prefix="",
                max_input_chars_per_word=100,
            )


def convert_to_flota(model: Model) -> Unigram:
    """Convert a model to a flota model."""
    vocabulary = model.vocab.sorted_vocabulary
    length_scores: list[float] = [-1 for x in vocabulary]
    with_length = list(zip(vocabulary, length_scores, strict=True))

    match model:
        case Unigram():
            unk_id = model.unk_id
            byte_fallback = model.byte_fallback
        case BPE():
            if model.unk_token:
                unk_id = model.vocab[model.unk_token]
            else:
                unk_id = None
            byte_fallback = model.byte_fallback
        case WordPiece() | WordLevel():
            unk_id = model.vocab[model.unk_token]
            byte_fallback = False

    return Unigram(vocab=UnigramVocabulary(with_length), unk_id=unk_id, byte_fallback=byte_fallback)


def get_continuing_subword_prefix_token(model: Model) -> str | None:
    """Get the prefix token from the model, if any."""
    # Only WordPiece and BPE models have these.
    if isinstance(model, (WordPiece, BPE)):
        return model.continuing_subword_prefix
    return None


def set_continuing_subword_prefix_token(model: Model, prefix: str) -> None:
    """Set the subword prefix. This will raise a ValueError if the model does not support one.

    Only WordPiece is supported here, even though BPE also has a `continuing_subword_prefix` field.
    For BPE, that field is only meaningful if the vocabulary and merges were built around it from the
    start (every continuation merge component prefixed, with a matching de-prefixed word-form also in
    the vocabulary): skeletoken's merge-rebuilding logic doesn't construct that shape, and the
    underlying `tokenizers` Rust BPE model doesn't validate it either, so setting this on an
    otherwise-ordinary BPE model reliably corrupts it (and can panic when it's reloaded).
    """
    if isinstance(model, WordPiece):
        model.continuing_subword_prefix = prefix
    else:
        raise ValueError("Setting a subword prefix token is only supported for WordPiece models.")


MODELS_THAT_NEED_UNK = (WordPiece, WordLevel)

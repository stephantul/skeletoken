from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from skeletoken.merges import Merges
from skeletoken.vocabulary import UnigramVocabulary, Vocabulary

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    WORDPIECE = "WordPiece"
    BPE = "BPE"
    UNIGRAM = "Unigram"
    WORDLEVEL = "WordLevel"


VocabTypeVar = TypeVar("VocabTypeVar", Vocabulary, UnigramVocabulary)


class VocabMixinMethod(Generic[VocabTypeVar]):
    """Mixin to override token addition, removal etc."""

    vocab: VocabTypeVar

    def add_token(self, token: str, is_added_token: bool = False) -> None:
        """Add a token to the vocabulary."""
        self.vocab.add_token(token)

    def replace_token(self, old_token: str, new_token: str, is_added_token: bool = False) -> None:
        """Replace a token in the vocabulary."""
        self.vocab.replace_token(old_token, new_token)

    def remove_token(self, token: str) -> None:
        """Remove a token from the vocabulary."""
        self.vocab.remove_token(token)

    def remove_tokens(self, tokens: list[str]) -> None:
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

    def to_greedy(self) -> WordPiece:
        """Convert the WordPiece model to a greedy version."""
        return self


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

    def to_greedy(self) -> WordPiece:
        """Convert the BPE model to a greedy WordPiece model."""
        if not self.unk_token:
            logger.warning("BPE model has no unk_token, using the first token in the vocab.")
            unk_token = self.vocab.sorted_vocabulary[0]
        else:
            unk_token = self.unk_token
        return WordPiece(
            vocab=self.vocab,
            unk_token=unk_token,
            continuing_subword_prefix=self.continuing_subword_prefix or "",
            max_input_chars_per_word=100,
        )

    def add_token(self, token: str, is_added_token: bool = False) -> None:
        """Add a token to the vocabulary."""
        self.vocab.add_token(token)
        if is_added_token:
            return
        self.merges._add_merges_for_token(token)
        new_tokens = sorted(self.merges._all_merge_tokens - set(self.vocab.vocabulary))
        for token in new_tokens:
            self.vocab.add_token(token)

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

    def remove_tokens(self, tokens: list[str]) -> None:
        """Remove multiple tokens from the vocabulary."""
        self.vocab.remove_tokens(tokens)
        for token in tokens:
            self.merges._remove_merges_for_token(token)

    def replace_vocabulary(self, vocabulary: list[str | None]) -> None:
        """Completely replaces the vocabulary with a vocabulary of the same length."""
        vocab = self.vocab.root
        if len(vocabulary) != len(vocab):
            raise ValueError("New vocabulary must be of the same length as the existing vocabulary.")
        current_vocab = self.vocab.vocabulary
        merge_index: list[tuple[int, int]] = []
        for left, right in self.merges.root:
            # We know that this merge leads somewhere
            merge_token = left + right
            index = current_vocab[merge_token]
            # No need to merge things that are removed
            if vocabulary[index] is None:
                continue
            left_idx = current_vocab[left]
            right_idx = current_vocab[right]
            if vocabulary[left_idx] is None or vocabulary[right_idx] is None:
                continue
            merge_index.append((left_idx, right_idx))
        self.vocab.replace_vocabulary(vocabulary)
        new_merges = []
        for left_idx, right_idx in merge_index:
            left_token, right_token = vocabulary[left_idx], vocabulary[right_idx]
            if left_token is None or right_token is None:
                continue
            token = left_token + right_token
            if token in self.vocab.vocabulary:
                new_merges.append((left_token, right_token))
        self.merges.root = new_merges
        self.merges.model_post_init({})


class Unigram(BaseModel, VocabMixinMethod[UnigramVocabulary]):
    """Data model representing a Unigram vocabulary."""

    type: Literal[ModelType.UNIGRAM] = ModelType.UNIGRAM
    vocab: UnigramVocabulary
    unk_id: int | None
    byte_fallback: bool

    def to_greedy(self) -> WordPiece:
        """Convert the Unigram model to a greedy WordPiece model."""
        if self.unk_id is None:
            unk_token = self.vocab.root[0][0]  # Use the first token as unk_token
            logger.warning("Unigram model has no `unk_id`, using the first token in the vocab.")
        else:
            unk_token = self.vocab.root[self.unk_id][0]
        return WordPiece(
            vocab=Vocabulary({token: idx for idx, (token, _) in enumerate(self.vocab.root)}),
            unk_token=unk_token,
            continuing_subword_prefix="",
            max_input_chars_per_word=100,
        )

    @property
    def unk_token(self) -> str | None:
        """Return the unknown token, if any."""
        if self.unk_id is None:
            return None
        return self.vocab.sorted_vocabulary[self.unk_id]

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

    def to_greedy(self) -> WordPiece:
        """Convert the WordLevel model to a greedy WordPiece model."""
        return WordPiece(
            vocab=self.vocab,
            unk_token=self.unk_token,
            continuing_subword_prefix="",
            max_input_chars_per_word=100,
        )


Model = WordPiece | BPE | Unigram | WordLevel
ModelDiscriminator = Annotated[Model, Field(discriminator="type")]


def get_subword_prefix_token(model: Model) -> str | None:
    """Get the prefix token from the model, if any."""
    # Only WordPiece and BPE models have these.
    if isinstance(model, WordPiece):
        return model.continuing_subword_prefix
    elif isinstance(model, BPE):
        return model.continuing_subword_prefix
    return None


MODELS_THAT_NEED_UNK = (WordPiece, WordLevel)

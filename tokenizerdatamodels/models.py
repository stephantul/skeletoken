from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    WORDPIECE = "WordPiece"
    BPE = "BPE"
    UNIGRAM = "Unigram"
    WORDLEVEL = "WordLevel"


class WordPiece(BaseModel):
    """Data model representing a WordPiece vocabulary."""

    type: Literal[ModelType.WORDPIECE] = ModelType.WORDPIECE
    vocab: dict[str, int]
    unk_token: str
    continuing_subword_prefix: str
    max_input_chars_per_word: int = 100

    def to_greedy(self) -> WordPiece:
        """Convert the WordPiece model to a greedy version."""
        return self


class BPE(BaseModel):
    """Data model representing a BPE vocabulary."""

    type: Literal[ModelType.BPE] = ModelType.BPE
    merges: list[tuple[str, str]]
    vocab: dict[str, int]
    dropout: float | None
    unk_token: str | None
    continuing_subword_prefix: str | None
    end_of_word_suffix: str | None
    fuse_unk: bool
    byte_fallback: bool
    ignore_merges: bool

    def to_greedy(self) -> WordPiece:
        """Convert the BPE model to a greedy WordPiece model."""
        first_token = next(iter(self.vocab))
        return WordPiece(
            vocab=self.vocab,
            unk_token=first_token,
            continuing_subword_prefix=self.continuing_subword_prefix or "",
            max_input_chars_per_word=100,
        )


class Unigram(BaseModel):
    """Data model representing a Unigram vocabulary."""

    type: Literal[ModelType.UNIGRAM] = ModelType.UNIGRAM
    vocab: list[tuple[str, float]]
    unk_id: int | None
    byte_fallback: bool

    def to_greedy(self) -> WordPiece:
        """Convert the Unigram model to a greedy WordPiece model."""
        if self.unk_id is None:
            unk_token = self.vocab[0][0]  # Use the first token as unk_token
        else:
            unk_token = self.vocab[self.unk_id][0]
        return WordPiece(
            vocab={token: idx for idx, (token, _) in enumerate(self.vocab)},
            unk_token=unk_token,
            continuing_subword_prefix="",
            max_input_chars_per_word=100,
        )


class WordLevel(BaseModel):
    """Data model representing a WordLevel vocabulary."""

    type: Literal[ModelType.WORDLEVEL] = ModelType.WORDLEVEL
    vocab: dict[str, int]
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

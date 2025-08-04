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


class BPE(BaseModel):
    """Data model representing a BPE vocabulary."""

    type: Literal[ModelType.BPE] = ModelType.BPE
    merges: list[tuple[str, str]]
    vocab: dict[str, int]
    dropout: float | None
    unk_token: str | None
    continuing_subword_prefix: str
    end_of_word_suffix: str
    fuse_unk: bool
    byte_fallback: bool
    ignore_merges: bool


class Unigram(BaseModel):
    """Data model representing a Unigram vocabulary."""

    type: Literal[ModelType.UNIGRAM] = ModelType.UNIGRAM
    vocab: list[tuple[str, float]]
    unk_id: int | None
    byte_fallback: bool


class WordLevel(BaseModel):
    """Data model representing a WordLevel vocabulary."""

    type: Literal[ModelType.WORDLEVEL] = ModelType.WORDLEVEL
    vocab: dict[str, int]
    unk_token: str


Model = WordPiece | BPE | Unigram | WordLevel
ModelDiscriminator = Annotated[Model, Field(discriminator="type")]

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from tokenizerdatamodels.common import PrependScheme, RegexPattern, StringPattern


class DecoderType(str, Enum):
    BPEDECODER = "BPEDecoder"
    BYTEFALLBACK = "ByteFallback"
    BYTELEVEL = "ByteLevel"
    CTC = "CTC"
    FUSE = "Fuse"
    METASPACE = "Metaspace"
    REPLACE = "Replace"
    STRIP = "Strip"
    WORDPIECE = "WordPiece"


class BPEDecoder(BaseModel):
    """
    A legacy BPE decoder.

    The `suffix` is used to determine the end of a BPE token.
    For example, if the suffix is "a", then "baba" will be decoded as "bb".

    Attributes
    ----------
        suffix: The suffix to use for BPE tokenization.

    """

    type: Literal[DecoderType.BPEDECODER] = DecoderType.BPEDECODER
    suffix: str


class ByteFallbackDecoder(BaseModel):
    """
    A ByteFallback decoder is used to handle byte-level tokens.

    It acts as a fallback for weird bytes, and replaces them with �, much like
    'utf-8' decoding does.
    """

    type: Literal[DecoderType.BYTEFALLBACK] = DecoderType.BYTEFALLBACK


class ByteLevelDecoder(BaseModel):
    """A ByteLevel decoder is used for byte-level tokenization."""

    type: Literal[DecoderType.BYTELEVEL] = DecoderType.BYTELEVEL
    add_prefix_space: bool
    trim_offsets: bool
    use_regex: bool


class CTCDecoder(BaseModel):
    """
    A CTC decoder is used for connectionist temporal classification.

    It removes any contiguous duplicates, e.g., "hh_ee_ll_ll_oo" -> "hello",
    and removes the padding token, e.g., "h h _ e e l l l _ o o" -> "hello".
    The word delimiter is replaced by a space between words, e.g., "hello|world" -> "hello | world".

    Attributes
    ----------
        pad_token: The padding token to remove.
        word_delimiter_token: The token used to separate words.
        cleanup: If True, it will clean up the output by removing some stuff.

    """

    type: Literal[DecoderType.CTC] = DecoderType.CTC
    pad_token: str
    word_delimiter_token: str
    cleanup: bool


class FuseDecoder(BaseModel):
    """
    A Fuse decoder just merges tokens.

    e.g., "un" + "known" -> "unknown"
    """

    type: Literal[DecoderType.FUSE] = DecoderType.FUSE


class MetaspaceDecoder(BaseModel):
    type: Literal[DecoderType.METASPACE] = DecoderType.METASPACE
    replacement: str
    prepend_scheme: PrependScheme
    split: bool


class ReplaceDecoder(BaseModel):
    type: Literal[DecoderType.REPLACE] = DecoderType.REPLACE
    pattern: StringPattern | RegexPattern
    content: str


class StripDecoder(BaseModel):
    type: Literal[DecoderType.STRIP] = DecoderType.STRIP
    content: str
    start: int
    stop: int


class WordPieceDecoder(BaseModel):
    type: Literal[DecoderType.WORDPIECE] = DecoderType.WORDPIECE
    prefix: str
    cleanup: bool


Decoder = (
    BPEDecoder
    | ByteFallbackDecoder
    | ByteLevelDecoder
    | CTCDecoder
    | FuseDecoder
    | MetaspaceDecoder
    | ReplaceDecoder
    | StripDecoder
    | WordPieceDecoder
)
DecoderDiscriminator = Annotated[Decoder, Field(discriminator="type")]

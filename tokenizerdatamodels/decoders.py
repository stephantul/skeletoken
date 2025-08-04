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
    type: Literal[DecoderType.BPEDECODER] = DecoderType.BPEDECODER
    suffix: str


class ByteFallbackDecoder(BaseModel):
    type: Literal[DecoderType.BYTEFALLBACK] = DecoderType.BYTEFALLBACK


class ByteLevelDecoder(BaseModel):
    type: Literal[DecoderType.BYTELEVEL] = DecoderType.BYTELEVEL
    add_prefix_space: bool
    trim_offsets: bool
    use_regex: bool


class CTCDecoder(BaseModel):
    type: Literal[DecoderType.CTC] = DecoderType.CTC
    pad_token: str
    word_delimiter_token: str
    cleanup: bool


class FuseDecoder(BaseModel):
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

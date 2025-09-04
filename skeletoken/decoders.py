from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from skeletoken.common import PrependScheme, RegexPattern, StringPattern


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
    SEQUENCE = "Sequence"


class DecoderSequence(BaseModel):
    """A sequence of decoders."""

    type: Literal[DecoderType.SEQUENCE] = DecoderType.SEQUENCE
    decoders: list[Decoder]


class BPEDecoder(BaseModel):
    """
    A legacy BPE decoder.

    The `suffix` is used to determine the end of a BPE token.
    For example, if the suffix is "a", then "baba" will be decoded as "bb".

    Attributes
    ----------
    suffix : str
        The suffix to use for BPE tokenization.

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
    """
    A ByteLevel decoder is used for byte-level tokenization.

    This decoder implements the inverse of the ByteLevel pretokenizer.

    Attributes
    ----------
    add_prefix_space : bool
        Whether to add a space before the first token. This leads to more consistent
        behavior for sentence-initial tokens, and is recommended to be set to True.
    trim_offsets : bool
        Whether to trim the offsets of the tokens.

    """

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
    pad_token : str
        The padding token to remove.
    word_delimiter_token : str
        The token used to separate words.
    cleanup : bool
        If set, it will clean up the output by removing some artifacts.

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
    """
    A Metaspace decoder is used for metaspace tokenization.

    This decoder inverts the metaspace tokenization process.

    Attributes
    ----------
    replacement : str
        The string to replace the metaspace tokens with.
    prepend_scheme : PrependScheme
        The scheme to use for prepending the replacement string.
    split : bool
        Should be set if the original Metaspace tokenizer performed splitting.

    """

    type: Literal[DecoderType.METASPACE] = DecoderType.METASPACE
    replacement: str
    prepend_scheme: PrependScheme
    split: bool


class ReplaceDecoder(BaseModel):
    """
    A Replace decoder replaces a pattern in the input text with a given content.

    It can be seen as the inverse of the Replace pretokenizer, but doesn't need to be the exact inverse.

    """

    type: Literal[DecoderType.REPLACE] = DecoderType.REPLACE
    pattern: StringPattern | RegexPattern
    content: str


class StripDecoder(BaseModel):
    """
    A Strip decoder strips characters from the input text.

    This decoder removes a specific content from the start and/or end of the input text.
    The start and stop indices are used to determine where to strip the content.
    If the start index is higher than the stop index, it will strip from the end of the content.

    e.g., start 0 and end 1 will strip the first character,
    while start 1 and end 0 will strip the last character.

    Attributes
    ----------
    content : str
        The content to strip.
    start : int
        The start index to strip from.
    stop : int
        The stop index to strip to.

    """

    type: Literal[DecoderType.STRIP] = DecoderType.STRIP
    content: str
    start: int
    stop: int


class WordPieceDecoder(BaseModel):
    """
    A WordPiece decoder is used for WordPiece tokenization.

    This decoder implements the inverse of the WordPiece pretokenizer.

    Attributes
    ----------
    prefix : str
        The subword prefix to use for WordPiece tokenization. This is usually
        '##'.
    cleanup : bool
        If True, it will clean up the output by removing some artifacts.

    """

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
    | DecoderSequence
)
DecoderDiscriminator = Annotated[Decoder, Field(discriminator="type")]

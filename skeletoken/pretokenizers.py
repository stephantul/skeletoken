from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from skeletoken.common import Behavior, PrependScheme, RegexPattern, StringPattern


class PreTokenizerType(str, Enum):
    BERT_PRETOKENIZER = "BertPreTokenizer"
    BYTELEVEL = "ByteLevel"
    CHARDELIMITERSPLIT = "CharDelimiterSplit"
    DIGITS = "Digits"
    FIXEDLENGTH = "FixedLength"
    METASPACE = "Metaspace"
    PUNCTUATION = "Punctuation"
    SPLIT = "Split"
    WHITESPACE = "Whitespace"
    WHITESPACESPLIT = "WhitespaceSplit"
    UNICODESCRIPTS = "UnicodeScripts"
    SEQUENCE = "Sequence"


class PreTokenizerSequence(BaseModel):
    """A sequence of pretokenizers to be applied in order."""

    type: Literal[PreTokenizerType.SEQUENCE] = PreTokenizerType.SEQUENCE
    pretokenizers: list[PreTokenizer]


class BertPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.BERT_PRETOKENIZER] = PreTokenizerType.BERT_PRETOKENIZER


class ByteLevelPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.BYTELEVEL] = PreTokenizerType.BYTELEVEL
    add_prefix_space: bool
    use_regex: bool
    trim_offsets: bool


class CharDelimiterSplitPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.CHARDELIMITERSPLIT] = PreTokenizerType.CHARDELIMITERSPLIT
    delimiter: str


class DigitsPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.DIGITS] = PreTokenizerType.DIGITS
    individual_digits: bool


class FixedLengthPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.FIXEDLENGTH] = PreTokenizerType.FIXEDLENGTH
    length: int


class MetaspacePreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.METASPACE] = PreTokenizerType.METASPACE
    replacement: str
    prepend_scheme: PrependScheme


class PunctuationPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.PUNCTUATION] = PreTokenizerType.PUNCTUATION
    behavior: Behavior


class SplitPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.SPLIT] = PreTokenizerType.SPLIT
    pattern: StringPattern | RegexPattern
    behavior: Behavior
    invert: bool


class WhitespacePreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.WHITESPACE] = PreTokenizerType.WHITESPACE


class WhitespaceSplitPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.WHITESPACESPLIT] = PreTokenizerType.WHITESPACESPLIT


class UnicodeScriptsPreTokenizer(BaseModel):
    type: Literal[PreTokenizerType.UNICODESCRIPTS] = PreTokenizerType.UNICODESCRIPTS


PreTokenizer = (
    BertPreTokenizer
    | ByteLevelPreTokenizer
    | CharDelimiterSplitPreTokenizer
    | DigitsPreTokenizer
    | FixedLengthPreTokenizer
    | MetaspacePreTokenizer
    | PunctuationPreTokenizer
    | SplitPreTokenizer
    | WhitespacePreTokenizer
    | WhitespaceSplitPreTokenizer
    | UnicodeScriptsPreTokenizer
    | PreTokenizerSequence
)
PreTokenizerDiscriminator = Annotated[PreTokenizer, Field(discriminator="type")]


def byte_tokenizes(pretokenizer: PreTokenizerDiscriminator | None) -> bool:
    """Check if a pretokenizer transforms the input into bytes."""
    if pretokenizer is None:
        return False
    # If it is a sequence, apply the function recursively
    # This is necessary, because it is possible to nest sequences of pretokenizers.
    if isinstance(pretokenizer, PreTokenizerSequence):
        return any(byte_tokenizes(x) for x in pretokenizer.pretokenizers)

    return isinstance(pretokenizer, ByteLevelPreTokenizer)

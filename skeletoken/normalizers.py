from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from skeletoken.common import RegexPattern, StringPattern


class NormalizerType(str, Enum):
    SEQUENCE = "Sequence"
    NFC = "NFC"
    NFD = "NFD"
    NFKC = "NFKC"
    NFKD = "NFKD"
    BERTNORMALIZER = "BertNormalizer"
    BYTELEVEL = "ByteLevel"
    LOWERCASE = "Lowercase"
    NMT = "Nmt"
    PREPEND = "Prepend"
    STRIP = "Strip"
    STRIPACCENTS = "StripAccents"
    REPLACE = "Replace"
    PRECOMPILED = "Precompiled"


class NormalizerSequence(BaseModel):
    """A sequence of normalizers to be applied in order."""

    sequence_field: str = "normalizers"
    type: Literal[NormalizerType.SEQUENCE] = NormalizerType.SEQUENCE
    normalizers: list[NormalizerDiscriminator]


class NFCNormalizer(BaseModel):
    """
    Applies NFC normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFC] = NormalizerType.NFC


class NFDNormalizer(BaseModel):
    """
    Applies NFD normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFD] = NormalizerType.NFD


class NFKCNormalizer(BaseModel):
    """
    Applies NFKC normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFKC] = NormalizerType.NFKC


class NFKDNormalizer(BaseModel):
    """
    Applies NFKD normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFKD] = NormalizerType.NFKD


class BertNormalizer(BaseModel):
    type: Literal[NormalizerType.BERTNORMALIZER] = NormalizerType.BERTNORMALIZER
    clean_text: bool
    handle_chinese_chars: bool
    strip_accents: bool | None
    lowercase: bool


class ByteLevelNormalizer(BaseModel):
    r"""
    Applies byte-level normalization to the input text.

    This normalizer applies the same transformations as the ByteLevel pretokenizer.
    Using this normalizer and adding a regex split pretokenizer is equivalent to using the ByteLevel pretokenizer.
    """

    type: Literal[NormalizerType.BYTELEVEL] = NormalizerType.BYTELEVEL


class LowercaseNormalizer(BaseModel):
    """Lowercases the input text."""

    type: Literal[NormalizerType.LOWERCASE] = NormalizerType.LOWERCASE


class NmtNormalizer(BaseModel):
    """
    A normalizer that removes specific codepoints.

    The codepoints:
        0x0001..=0x0008 -> Control characters SOH to BS
        0x000B -> Vertical tab
        0x000E..=0x001F -> More control characters
        0x007F -> DEL (delete)
        0x008F, 0x009F -> Control characters from C1 set

    are removed

    The codepoints:
        0x0009 => Tab (Horizontal Tab)
        0x000A => Line Feed (LF / Newline)
        0x000C => Form Feed (FF)
        0x000D => Carriage Return (CR)
        0x1680 => Ogham Space Mark
        0x200B..=0x200F => Zero Width Space and related (ZWSP, ZWNJ, ZWJ, LRM, RLM, etc.)
        0x2028 => Line Separator
        0x2029 => Paragraph Separator
        0x2581 => Lower One Eighth Block (▁) – used as visible space in some tokenizers
        0xFEFF => Zero Width No-Break Space / Byte Order Mark (BOM)
        0xFFFD => Replacement Character (�)

    are replaced with a space character (U+0020).
    """

    type: Literal[NormalizerType.NMT] = NormalizerType.NMT


class PrependedNormalizer(BaseModel):
    """Prepends a string to the input text."""

    type: Literal[NormalizerType.PREPEND] = NormalizerType.PREPEND
    prepend: str


class StripNormalizer(BaseModel):
    """Strips whitespace from the left and/or right side of the input text."""

    type: Literal[NormalizerType.STRIP] = NormalizerType.STRIP
    strip_left: bool
    strip_right: bool


class StripAccentsNormalizer(BaseModel):
    """Strips accents from the input text."""

    type: Literal[NormalizerType.STRIPACCENTS] = NormalizerType.STRIPACCENTS


class ReplaceNormalizer(BaseModel):
    """Replaces a pattern in the input text with a given content."""

    type: Literal[NormalizerType.REPLACE] = NormalizerType.REPLACE
    pattern: StringPattern | RegexPattern
    content: str


class PrecompiledNormalizer(BaseModel):
    """
    A precompiled normalizer that uses a precompiled characters map.

    NOTE: It is unclear how this is constructed, and is mainly here for compatibility with sentencepiece
    """

    type: Literal[NormalizerType.PRECOMPILED] = NormalizerType.PRECOMPILED
    precompiled_charsmap: str


Normalizer = (
    NFCNormalizer
    | NFDNormalizer
    | NFKCNormalizer
    | NFKDNormalizer
    | BertNormalizer
    | ByteLevelNormalizer
    | LowercaseNormalizer
    | NmtNormalizer
    | PrependedNormalizer
    | StripNormalizer
    | StripAccentsNormalizer
    | ReplaceNormalizer
    | PrecompiledNormalizer
    | NormalizerSequence
)
NormalizerDiscriminator = Annotated[Normalizer, Field(discriminator="type")]


def lower_cases(normalizer: NormalizerDiscriminator | None) -> bool:
    """
    Check if a normalizer performs lowercasing.

    Note that it is possible to still lowercase if this returns True, because
    of a replacenormalizer. I think this is a super edge case, but it could still be done.
    """
    if normalizer is None:
        return False
    # If it is a sequence, apply the function recursively
    # This is necessary, because it is possible to nest sequences of normalizers.
    if isinstance(normalizer, NormalizerSequence):
        return any(lower_cases(x) for x in normalizer.normalizers)

    if isinstance(normalizer, LowercaseNormalizer):
        return True
    elif isinstance(normalizer, BertNormalizer):
        if normalizer.lowercase:
            return True

    return False


def byte_normalizes(normalizer: NormalizerDiscriminator | None) -> bool:
    """Check if a normalizer transforms the input into bytes."""
    if normalizer is None:
        return False
    # If it is a sequence, apply the function recursively
    # This is necessary, because it is possible to nest sequences of normalizers.
    if isinstance(normalizer, NormalizerSequence):
        return any(byte_normalizes(x) for x in normalizer.normalizers)

    return isinstance(normalizer, ByteLevelNormalizer)

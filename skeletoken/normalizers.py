from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, PrivateAttr

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


class BaseNormalizer(BaseModel):
    """Base class for all normalizers."""

    _lowercases: bool = PrivateAttr(default=False)
    _byte_normalizes: bool = PrivateAttr(default=False)

    @property
    def lowercases(self) -> bool:
        """Check if the normalizer lowercases."""
        return self._lowercases

    @property
    def byte_normalizes(self) -> bool:
        """Check if the normalizer byte normalizes."""
        return self._byte_normalizes


class NormalizerSequence(BaseModel):
    """A sequence of normalizers to be applied in order."""

    sequence_field: str = "normalizers"
    type: Literal[NormalizerType.SEQUENCE] = NormalizerType.SEQUENCE
    normalizers: list[NormalizerDiscriminator]

    @property
    def lowercases(self) -> bool:
        """Check if the sequence contains a lowercase normalizer."""
        return any(normalizer.lowercases for normalizer in self.normalizers)

    @property
    def byte_normalizes(self) -> bool:
        """Check if the sequence contains a byte normalizer."""
        return any(normalizer.byte_normalizes for normalizer in self.normalizers)


class NFCNormalizer(BaseNormalizer):
    """
    Applies NFC normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFC] = NormalizerType.NFC


class NFDNormalizer(BaseNormalizer):
    """
    Applies NFD normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFD] = NormalizerType.NFD


class NFKCNormalizer(BaseNormalizer):
    """
    Applies NFKC normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFKC] = NormalizerType.NFKC


class NFKDNormalizer(BaseNormalizer):
    """
    Applies NFKD normalization to the input text.

    See here for more details:
    https://unicode.org/reports/tr15/#Normalization_Forms

    """

    type: Literal[NormalizerType.NFKD] = NormalizerType.NFKD


class BertNormalizer(BaseNormalizer):
    """
    The normalization used by the original BERT.

    Parameters
    ----------
    clean_text : bool
        If set, this normalizes whitespace and removes any control characters.
    handle_chinese_chars : bool
        If set, it surrounds any Chinese characters in the text with spaces so that they are pretokenized correctly.
    strip_accents : bool | None
        If set, this removes all accents from the text.

    """

    type: Literal[NormalizerType.BERTNORMALIZER] = NormalizerType.BERTNORMALIZER
    clean_text: bool
    handle_chinese_chars: bool
    strip_accents: bool | None
    lowercase: bool

    @property
    def lowercases(self) -> bool:
        """Check if the normalizer lowercases."""
        return self.lowercase


class ByteLevelNormalizer(BaseNormalizer):
    """
    Applies byte-level normalization to the input text.

    This normalizer applies the same transformations as the ByteLevel pretokenizer.
    Using this normalizer and adding a regex split pretokenizer is equivalent to using the ByteLevel pretokenizer.

    """

    type: Literal[NormalizerType.BYTELEVEL] = NormalizerType.BYTELEVEL
    _byte_normalizes: bool = PrivateAttr(default=True)


class LowercaseNormalizer(BaseNormalizer):
    """Lowercases the input text."""

    type: Literal[NormalizerType.LOWERCASE] = NormalizerType.LOWERCASE
    _lowercases: bool = True


class NmtNormalizer(BaseNormalizer):
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


class PrependNormalizer(BaseNormalizer):
    """
    Prepends a string to the input text.

    Parameters
    ----------
    prepend : str
        The string to prepend to a text.

    """

    type: Literal[NormalizerType.PREPEND] = NormalizerType.PREPEND
    prepend: str


class StripNormalizer(BaseNormalizer):
    """
    Strips whitespace from the left and/or right side of the input text.

    Parameters
    ----------
    strip_left : bool
        If set, this removes whitespace from the left side.
    strip_right : bool
        If set, this removes whitespace from the right side.

    """

    type: Literal[NormalizerType.STRIP] = NormalizerType.STRIP
    strip_left: bool
    strip_right: bool


class StripAccentsNormalizer(BaseNormalizer):
    """Strips accents from the input text."""

    type: Literal[NormalizerType.STRIPACCENTS] = NormalizerType.STRIPACCENTS


class ReplaceNormalizer(BaseNormalizer):
    """Replaces a pattern in the input text with a given content."""

    type: Literal[NormalizerType.REPLACE] = NormalizerType.REPLACE
    pattern: StringPattern | RegexPattern
    content: str


class PrecompiledNormalizer(BaseNormalizer):
    """
    A precompiled normalizer that uses a precompiled characters map.

    NOTE: It is unclear how this is constructed, and is mainly here for compatibility with sentencepiece

    Attributes
    ----------
    precompiled_charsmap : str
        The precompiled characters map.

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
    | PrependNormalizer
    | StripNormalizer
    | StripAccentsNormalizer
    | ReplaceNormalizer
    | PrecompiledNormalizer
    | NormalizerSequence
)
NormalizerDiscriminator = Annotated[Normalizer, Field(discriminator="type")]

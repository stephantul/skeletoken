from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from tokenizerdatamodels.common import RegexPattern, StringPattern


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
    type: Literal[NormalizerType.SEQUENCE] = NormalizerType.SEQUENCE
    normalizers: list[Normalizer]


class NFCNormalizer(BaseModel):
    type: Literal[NormalizerType.NFC] = NormalizerType.NFC


class NFDNormalizer(BaseModel):
    type: Literal[NormalizerType.NFD] = NormalizerType.NFD


class NFKCNormalizer(BaseModel):
    type: Literal[NormalizerType.NFKC] = NormalizerType.NFKC


class NFKDNormalizer(BaseModel):
    type: Literal[NormalizerType.NFKD] = NormalizerType.NFKD


class BertNormalizer(BaseModel):
    type: Literal[NormalizerType.BERTNORMALIZER] = NormalizerType.BERTNORMALIZER
    clean_text: bool
    handle_chinese_chars: bool
    strip_accents: bool | None
    lowercase: bool


class ByteLevelNormalizer(BaseModel):
    type: Literal[NormalizerType.BYTELEVEL] = NormalizerType.BYTELEVEL


class LowercaseNormalizer(BaseModel):
    type: Literal[NormalizerType.LOWERCASE] = NormalizerType.LOWERCASE


class NmtNormalizer(BaseModel):
    type: Literal[NormalizerType.NMT] = NormalizerType.NMT


class PrependedNormalizer(BaseModel):
    type: Literal[NormalizerType.PREPEND] = NormalizerType.PREPEND
    prepend: str


class StripNormalizer(BaseModel):
    type: Literal[NormalizerType.STRIP] = NormalizerType.STRIP
    strip_left: bool
    strip_right: bool


class StripAccentsNormalizer(BaseModel):
    type: Literal[NormalizerType.STRIPACCENTS] = NormalizerType.STRIPACCENTS


class ReplaceNormalizer(BaseModel):
    type: Literal[NormalizerType.REPLACE] = NormalizerType.REPLACE
    pattern: StringPattern | RegexPattern
    content: str


class PrecompiledNormalizer(BaseModel):
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

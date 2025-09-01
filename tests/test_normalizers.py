from typing import Any

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.common import PrependScheme, StringPattern
from skeletoken.normalizers import (
    BertNormalizer,
    ByteLevelNormalizer,
    LowercaseNormalizer,
    NFCNormalizer,
    NFDNormalizer,
    NFKCNormalizer,
    NFKDNormalizer,
    NmtNormalizer,
    Normalizer,
    NormalizerSequence,
    NormalizerType,
    PrecompiledNormalizer,
    PrependNormalizer,
    ReplaceNormalizer,
    StripAccentsNormalizer,
    StripNormalizer,
)


def _get_default_normalizer(normalizer_type: NormalizerType) -> Normalizer:  # noqa: C901
    """Helper function to get the default instantiation of a normalizer."""
    if normalizer_type == NormalizerType.BYTELEVEL:
        return ByteLevelNormalizer()
    elif normalizer_type == NormalizerType.BERTNORMALIZER:
        return BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)
    elif normalizer_type == NormalizerType.LOWERCASE:
        return LowercaseNormalizer()
    elif normalizer_type == NormalizerType.NFC:
        return NFCNormalizer()
    elif normalizer_type == NormalizerType.NFD:
        return NFDNormalizer()
    elif normalizer_type == NormalizerType.NFKC:
        return NFKCNormalizer()
    elif normalizer_type == NormalizerType.NFKD:
        return NFKDNormalizer()
    elif normalizer_type == NormalizerType.NMT:
        return NmtNormalizer()
    elif normalizer_type == NormalizerType.PREPEND:
        return PrependNormalizer(prepend=PrependScheme.FIRST)
    elif normalizer_type == NormalizerType.STRIP:
        return StripNormalizer(strip_left=True, strip_right=True)
    elif normalizer_type == NormalizerType.REPLACE:
        return ReplaceNormalizer(pattern=StringPattern(String="a"), content="replacement")
    elif normalizer_type == NormalizerType.STRIPACCENTS:
        return StripAccentsNormalizer()
    elif normalizer_type == NormalizerType.PRECOMPILED:
        return PrecompiledNormalizer(precompiled_charsmap="precompiled_charsmap")
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}")


@pytest.mark.parametrize(
    "normalizer_type",
    [
        NormalizerType.BYTELEVEL,
        NormalizerType.BERTNORMALIZER,
        NormalizerType.LOWERCASE,
        NormalizerType.NFC,
        NormalizerType.NFD,
        NormalizerType.NFKC,
        NormalizerType.NFKD,
        NormalizerType.NMT,
        NormalizerType.PREPEND,
        NormalizerType.STRIP,
        NormalizerType.REPLACE,
    ],
)
def test_normalizer(small_tokenizer_json: dict[str, Any], normalizer_type: NormalizerType) -> None:
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    normalizer = _get_default_normalizer(normalizer_type)
    normalizer_dict = normalizer.model_dump()
    small_tokenizer_json["normalizer"] = normalizer_dict
    tokenizer = TokenizerModel.model_validate(small_tokenizer_json)

    assert tokenizer.normalizer is not None
    assert tokenizer.normalizer.type == normalizer_type

    # Implicit test. If this fails, the model is incorrect.
    tokenizer.to_tokenizer()


@pytest.mark.parametrize(
    "normalizer,should_normalize",
    [
        [_get_default_normalizer(NormalizerType.BERTNORMALIZER), True],
        [BertNormalizer(clean_text=False, handle_chinese_chars=True, strip_accents=False, lowercase=False), False],
        [_get_default_normalizer(NormalizerType.LOWERCASE), True],
        [_get_default_normalizer(NormalizerType.STRIP), False],
        [NormalizerSequence(normalizers=[_get_default_normalizer(NormalizerType.LOWERCASE)]), True],
        [
            NormalizerSequence(
                normalizers=[NormalizerSequence(normalizers=[_get_default_normalizer(NormalizerType.LOWERCASE)])]
            ),
            True,
        ],
    ],
)
def test_lowercases(normalizer: Normalizer, should_normalize: bool) -> None:
    """Test whether the lowercases detection works."""
    assert normalizer.lowercases == should_normalize


@pytest.mark.parametrize(
    "normalizer,should_normalize",
    [
        [_get_default_normalizer(NormalizerType.BYTELEVEL), True],
        [_get_default_normalizer(NormalizerType.STRIP), False],
        [NormalizerSequence(normalizers=[_get_default_normalizer(NormalizerType.LOWERCASE)]), False],
        [
            NormalizerSequence(
                normalizers=[NormalizerSequence(normalizers=[_get_default_normalizer(NormalizerType.LOWERCASE)])]
            ),
            False,
        ],
    ],
)
def test_byte_normalizes(normalizer: Normalizer, should_normalize: bool) -> None:
    """Test whether the lowercases detection works."""
    assert normalizer.byte_normalizes == should_normalize

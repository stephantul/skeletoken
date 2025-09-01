from typing import Any

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.common import PrependScheme, StringPattern
from skeletoken.pretokenizers import (
    Behavior,
    BertPreTokenizer,
    ByteLevelPreTokenizer,
    CharDelimiterSplitPreTokenizer,
    DigitsPreTokenizer,
    FixedLengthPreTokenizer,
    MetaspacePreTokenizer,
    PreTokenizer,
    PreTokenizerSequence,
    PreTokenizerType,
    PunctuationPreTokenizer,
    SplitPreTokenizer,
    UnicodeScriptsPreTokenizer,
    WhitespacePreTokenizer,
    WhitespaceSplitPreTokenizer,
    get_metaspace,
)


def _get_default_pretokenizer(pretokenizer_type: PreTokenizerType) -> PreTokenizer:  # noqa: C901
    """Helper function to get the default instantiation of a pretokenizer."""
    if pretokenizer_type == PreTokenizerType.BERT_PRETOKENIZER:
        return BertPreTokenizer()
    elif pretokenizer_type == PreTokenizerType.BYTELEVEL:
        return ByteLevelPreTokenizer(add_prefix_space=False, use_regex=False, trim_offsets=False)
    elif pretokenizer_type == PreTokenizerType.CHARDELIMITERSPLIT:
        return CharDelimiterSplitPreTokenizer(delimiter="|")
    elif pretokenizer_type == PreTokenizerType.DIGITS:
        return DigitsPreTokenizer(individual_digits=True)
    elif pretokenizer_type == PreTokenizerType.FIXEDLENGTH:
        return FixedLengthPreTokenizer(length=5)
    elif pretokenizer_type == PreTokenizerType.METASPACE:
        return MetaspacePreTokenizer(replacement=" ", prepend_scheme=PrependScheme.FIRST, split=True)
    elif pretokenizer_type == PreTokenizerType.PUNCTUATION:
        return PunctuationPreTokenizer(behavior=Behavior.CONTIGUOUS)
    elif pretokenizer_type == PreTokenizerType.SPLIT:
        return SplitPreTokenizer(pattern=StringPattern(String="a"), behavior=Behavior.ISOLATED, invert=False)
    elif pretokenizer_type == PreTokenizerType.WHITESPACE:
        return WhitespacePreTokenizer()
    elif pretokenizer_type == PreTokenizerType.WHITESPACESPLIT:
        return WhitespaceSplitPreTokenizer()
    elif pretokenizer_type == PreTokenizerType.UNICODESCRIPTS:
        return UnicodeScriptsPreTokenizer()
    else:
        raise ValueError(f"Unknown pretokenizer type: {pretokenizer_type}")


@pytest.mark.parametrize(
    "pretokenizer_type",
    [
        PreTokenizerType.BYTELEVEL,
        PreTokenizerType.BERT_PRETOKENIZER,
        PreTokenizerType.CHARDELIMITERSPLIT,
        PreTokenizerType.DIGITS,
        PreTokenizerType.FIXEDLENGTH,
        PreTokenizerType.METASPACE,
        PreTokenizerType.PUNCTUATION,
        PreTokenizerType.SPLIT,
        PreTokenizerType.WHITESPACE,
        PreTokenizerType.WHITESPACESPLIT,
        PreTokenizerType.UNICODESCRIPTS,
    ],
)
def test_pretokenizer(small_tokenizer_json: dict[str, Any], pretokenizer_type: PreTokenizerType) -> None:
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    normalizer = _get_default_pretokenizer(pretokenizer_type)
    normalizer_dict = normalizer.model_dump()
    small_tokenizer_json["pre_tokenizer"] = normalizer_dict
    tokenizer = TokenizerModel.model_validate(small_tokenizer_json)

    assert tokenizer.pre_tokenizer is not None
    assert tokenizer.pre_tokenizer.type == pretokenizer_type

    # Implicit test. If this fails, the model is incorrect.
    tokenizer.to_tokenizer()


@pytest.mark.parametrize(
    "pretokenizer,should_byte_transform",
    [
        [_get_default_pretokenizer(PreTokenizerType.BERT_PRETOKENIZER), False],
        [_get_default_pretokenizer(PreTokenizerType.BYTELEVEL), True],
        [PreTokenizerSequence(pretokenizers=[_get_default_pretokenizer(PreTokenizerType.BYTELEVEL)]), True],
        [
            PreTokenizerSequence(
                pretokenizers=[
                    PreTokenizerSequence(pretokenizers=[_get_default_pretokenizer(PreTokenizerType.BYTELEVEL)])
                ]
            ),
            True,
        ],
    ],
)
def test_byte_transform(pretokenizer: PreTokenizer, should_byte_transform: bool) -> None:
    """Test whether the byte transform detection works."""
    assert pretokenizer._byte_pretokenizes == should_byte_transform


@pytest.mark.parametrize(
    "pretokenizer,splits",
    [
        [_get_default_pretokenizer(PreTokenizerType.BERT_PRETOKENIZER), True],
        [_get_default_pretokenizer(PreTokenizerType.BYTELEVEL), False],
        [PreTokenizerSequence(pretokenizers=[_get_default_pretokenizer(PreTokenizerType.BYTELEVEL)]), False],
        [
            PreTokenizerSequence(
                pretokenizers=[
                    PreTokenizerSequence(pretokenizers=[_get_default_pretokenizer(PreTokenizerType.BYTELEVEL)])
                ]
            ),
            False,
        ],
        [_get_default_pretokenizer(PreTokenizerType.METASPACE), True],
    ],
)
def test_splits(pretokenizer: PreTokenizer, splits: bool) -> None:
    """Test whether a pretokenizer splits."""
    assert pretokenizer._splits == splits


def test_get_metaspace() -> None:
    """Test detection of a metaspace."""
    default = _get_default_pretokenizer(PreTokenizerType.METASPACE)
    assert get_metaspace(default) == " "
    assert get_metaspace(PreTokenizerSequence(pretokenizers=[default])) == " "

    other_default = _get_default_pretokenizer(PreTokenizerType.BERT_PRETOKENIZER)
    assert get_metaspace(other_default) is None
    assert get_metaspace(PreTokenizerSequence(pretokenizers=[other_default])) is None

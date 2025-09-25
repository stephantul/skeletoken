from tokenizers.pre_tokenizers import (
    BertPreTokenizer,
    ByteLevel,
    CharDelimiterSplit,
    Digits,
    FixedLength,
    Metaspace,
    Punctuation,
    Sequence,
    Split,
    UnicodeScripts,
    Whitespace,
    WhitespaceSplit,
)
from tokenizers.pre_tokenizers import PreTokenizer as BasePretokenizer

from skeletoken.preprocessor.utils import replace_pattern
from skeletoken.pretokenizers import PreTokenizer, PreTokenizerSequence, PreTokenizerType

_class_maping: dict[PreTokenizerType, type[BasePretokenizer]] = {
    PreTokenizerType.SEQUENCE: Sequence,
    PreTokenizerType.BYTELEVEL: ByteLevel,
    PreTokenizerType.DIGITS: Digits,
    PreTokenizerType.METASPACE: Metaspace,
    PreTokenizerType.PUNCTUATION: Punctuation,
    PreTokenizerType.SPLIT: Split,
    PreTokenizerType.BYTELEVEL: ByteLevel,
    PreTokenizerType.WHITESPACE: Whitespace,
    PreTokenizerType.WHITESPACESPLIT: WhitespaceSplit,
    PreTokenizerType.BERT_PRETOKENIZER: BertPreTokenizer,
    PreTokenizerType.CHARDELIMITERSPLIT: CharDelimiterSplit,
    PreTokenizerType.FIXEDLENGTH: FixedLength,
    PreTokenizerType.UNICODESCRIPTS: UnicodeScripts,
}


def create_pretokenizer(pretokenizer: PreTokenizer) -> BasePretokenizer:
    """Create a pretokenizer from a Pretokenizer object."""
    t = pretokenizer.type
    if isinstance(pretokenizer, PreTokenizerSequence):
        return Sequence(pre_tokenizers=[create_pretokenizer(child) for child in pretokenizer.pretokenizers])  # type: ignore[arg-type]
    selected_class = _class_maping[t]
    dict_obj = pretokenizer.model_dump()
    dict_obj.pop("type")
    dict_obj = replace_pattern(dict_obj)
    return selected_class(**dict_obj)

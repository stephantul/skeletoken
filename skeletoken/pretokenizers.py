from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, PrivateAttr

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

    @property
    def _byte_pretokenizes(self) -> bool:
        return any(x._byte_pretokenizes for x in self.pretokenizers)

    @property
    def _splits(self) -> bool:
        return any(x._splits for x in self.pretokenizers)


class BasePretokenizer(BaseModel):
    """A base class for pretokenizers."""

    _byte_pretokenizes: bool = PrivateAttr(default=False)

    @property
    def _splits(self) -> bool:
        """Most pretokenizers split."""
        return True


class BertPreTokenizer(BasePretokenizer):
    """
    The BERT pretokenizer.

    This pretokenizer splits on spaces and punctuation.
    Each occurrence of a punctuation character counts separately,

    so: ",,," will be pretokenized as [",", ",", ","], despite these
    not being separated by spaces.

    """

    type: Literal[PreTokenizerType.BERT_PRETOKENIZER] = PreTokenizerType.BERT_PRETOKENIZER


class ByteLevelPreTokenizer(BasePretokenizer):
    """
    The pretokenizer used for BPE.

    This pretokenizer converts your input text into a sequence of byte-level tokens.
    These byte level tokens are not exactly the standard byte ranges, but a remapped
    version.

    This pretokenizer only splits, i.e., creates pre-tokens, if use_regex is set. Otherwise,
    it creates a single token, and should be accompanied by another pretokenizer.

    Attributes
    ----------
    add_prefix_space : bool
        Whether to add a space before the first token. This is desirable because it
        creates more correspondence between tokens.
    use_regex: bool
        Whether to use regex for splitting. If this is not set, no splitting is performed.
    trim_offsets: bool
        Doesn't do anything

    """

    type: Literal[PreTokenizerType.BYTELEVEL] = PreTokenizerType.BYTELEVEL
    add_prefix_space: bool
    use_regex: bool
    trim_offsets: bool
    _byte_pretokenizes: bool = PrivateAttr(default=True)

    @property
    def _splits(self) -> bool:
        """Only splits if use_regex is True."""
        return self.use_regex


class CharDelimiterSplitPreTokenizer(BasePretokenizer):
    """
    Pretokenizes on a specific delimiter.

    The delimiter should be a single character, and can't be a regular expression.

    Attributes
    ----------
    delimiter : str
        The character to split on.

    """

    type: Literal[PreTokenizerType.CHARDELIMITERSPLIT] = PreTokenizerType.CHARDELIMITERSPLIT
    delimiter: str


class DigitsPreTokenizer(BasePretokenizer):
    """
    Split on digits.

    This tokenizer splits digits, and possibly sequences of digits, into tokens.

    Attributes
    ----------
    individual_digits : bool
        If set, this splits on individual digits. If not, this splits on groups of digits.
        e.g. "111" -> ["1", "1", "1"] if this is set.

    """

    type: Literal[PreTokenizerType.DIGITS] = PreTokenizerType.DIGITS
    individual_digits: bool


class FixedLengthPreTokenizer(BasePretokenizer):
    """
    Pretokenizes into fixed length sequences.

    Attributes
    ----------
    length : int
        The length of each sequence.

    """

    type: Literal[PreTokenizerType.FIXEDLENGTH] = PreTokenizerType.FIXEDLENGTH
    length: int


class MetaspacePreTokenizer(BasePretokenizer):
    """
    The metaspace pretokenizer.

    This tokenizer replaces the space character by a single character. This is usually
    `▁` (U+2581).

    Attributes
    ----------
    replacement : str
        The character to replace spaces with.
    prepend_scheme : PrependScheme
        The scheme to use for prepending the sequence.
        If this is set to `FIRST`, it will prepend the replacement character to the
        first token in the sequence.
    split : bool
        Whether to split the input text into tokens.
        The split is done on the specified replacement character, but the split character
        is kept during splitting, and prepended to the token.

    """

    type: Literal[PreTokenizerType.METASPACE] = PreTokenizerType.METASPACE
    replacement: str
    prepend_scheme: PrependScheme
    split: bool

    @property
    def _splits(self) -> bool:
        """Splits if split is set to True."""
        return self.split


class PunctuationPreTokenizer(BasePretokenizer):
    """
    Splits on punctuation, e.g., most punctuation characters.

    Note that the behavior for this is weird for unicode punctuation.

    Attributes
    ----------
    behavior : Behavior
        See the Behavior docstring for how this functions.

    """

    type: Literal[PreTokenizerType.PUNCTUATION] = PreTokenizerType.PUNCTUATION
    behavior: Behavior


class SplitPreTokenizer(BasePretokenizer):
    """
    Split a sequence on a specified pattern.

    The specified pattern can either be a regex or a string. The string and regex
    should be specified as follows:

    {"String": my_string}
    or
    {"Regex": my_regex}

    Attributes
    ----------
    pattern : StringPattern | RegexPattern
        The pattern to split on.
        If you use a regex, it should be a valid regex pattern.
    behavior : Behavior
        The behavior to use when splitting. See the docstring of Behavior
        for the different possibilities.
    invert : bool
        Whether to invert the pattern. The invert is ignored when a non-Regex, i.e.,
        a string pattern, is used.

    """

    type: Literal[PreTokenizerType.SPLIT] = PreTokenizerType.SPLIT
    pattern: StringPattern | RegexPattern
    behavior: Behavior
    invert: bool


class WhitespacePreTokenizer(BasePretokenizer):
    r"""
    Splits using the following regex: "\w+|[^\w\s]+".

    This can be seen as a standard word-based splitter, very similar to the
    one used in older tokenizers.

    """

    type: Literal[PreTokenizerType.WHITESPACE] = PreTokenizerType.WHITESPACE


class WhitespaceSplitPreTokenizer(BasePretokenizer):
    """
    Literally only splits on whitespace.

    It's not really desirable, and can be considered a variant of CharDelimiterSplit.

    """

    type: Literal[PreTokenizerType.WHITESPACESPLIT] = PreTokenizerType.WHITESPACESPLIT


class UnicodeScriptsPreTokenizer(BasePretokenizer):
    """
    Splits when encountering a new Unicode script.

    This is interesting when training multilingual tokenizers, because this guarantees
    that the tokenizer will split on script and language boundaries.

    Examples
    --------
    >>> s = "hello, ごきげんよう?"
    >>> tokenizer = UnicodeScriptsPreTokenizer()
    >>> tokenizer.pre_tokenize_str(s)
    [('hello', (0, 5)), (', ', (5, 7)), ('ごきげんよう', (7, 13)), ('?', (13, 14))]

    """

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


def get_metaspace(pre_tokenizer: PreTokenizerDiscriminator) -> str | None:
    """Get the metaspace token from a pre-tokenizer."""
    if isinstance(pre_tokenizer, PreTokenizerSequence):
        for pt in pre_tokenizer.pretokenizers:
            if result := get_metaspace(pt):
                return result
    elif isinstance(pre_tokenizer, MetaspacePreTokenizer):
        return pre_tokenizer.replacement
    return None

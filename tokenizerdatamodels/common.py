from enum import Enum

from pydantic import BaseModel


class StringPattern(BaseModel):
    """A string pattern for use in a replace."""

    String: str


class RegexPattern(BaseModel):
    """A regex pattern for use in a replace."""

    Regex: str


class PrependScheme(str, Enum):
    """
    The prepend scheme used in metaspace tokenizers.

    This enum governs the behavior of the metaspace prepend.

    ALWAYS: prepend all sequences with a metaspace.
    NEVER: never prepend. The behavior of this depends on other tokenizers.
        If Metaspace is the only tokenizer, this only governs the behavior of
        the first token in the sequence, otherwise it also affects other tokens.
    FIRST: Only prepend the first token in a sequence with a metaspace.
    """

    ALWAYS = "always"
    NEVER = "never"
    FIRST = "first"


class Behavior(str, Enum):
    """
    Behavior for split.

    For pattern "a", the behavior is as follows:

    REMOVED: removes the character from the sequence. "bab" -> "b", "b"
    ISOLATED: splits the character as a separate token. "bab" -> "b", "a", "b"
    MERGED_WITH_PREVIOUS: merges with the token before it. "bab" -> "ba", "b"
    MERGED_WITH_NEXT: merges with the following token. "bab" -> "b", "ab"
    CONTIGUOUS: merges contiguous blocks of patterns: "baaab" -> "b", "aaa", "b"
    """

    REMOVED = "Removed"
    ISOLATED = "Isolated"
    MERGED_WITH_PREVIOUS = "MergedWithPrevious"
    MERGED_WITH_NEXT = "MergedWithNext"
    CONTIGUOUS = "Contiguous"

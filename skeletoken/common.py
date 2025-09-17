import re
from enum import Enum

import regex
from pydantic import BaseModel


class StringPattern(BaseModel):
    """A string pattern for use in a replace."""

    String: str


class RegexPattern(BaseModel):
    """A regex pattern for use in a replace."""

    Regex: str


def coerce_string_regex_pattern(
    v: str | regex.Pattern | re.Pattern | dict | StringPattern | RegexPattern,
) -> dict | StringPattern | RegexPattern:
    """Helper function that turns a string or regex pattern into the appropriate dict."""
    # Users can pass: str, compiled regex, or the tagged dict forms
    if isinstance(v, (StringPattern, RegexPattern)):
        return v
    if isinstance(v, str):
        return {"String": v}
    if isinstance(v, (regex.Pattern, re.Pattern)):
        return {"Regex": v.pattern}
    if isinstance(v, dict) and (("String" in v) or ("Regex" in v)):
        return v
    raise TypeError("pattern must be a string, a compiled regex, or a dict like {'String': ...} / {'Regex': ...}")


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

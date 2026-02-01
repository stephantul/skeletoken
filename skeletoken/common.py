import re
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import regex
from pydantic import BaseModel, TypeAdapter, ValidationError

PathLike = str | Path
RegexType = TypeVar("RegexType", re.Pattern[str], regex.Pattern[str])


class StringPattern(BaseModel):
    """A string pattern for use in a replace."""

    String: str


class RegexPattern(BaseModel):
    """A regex pattern for use in a replace."""

    Regex: str


Pattern = StringPattern | RegexPattern
PATTERN_ADAPTOR: TypeAdapter[Pattern] = TypeAdapter(Pattern)


def coerce_string_regex_pattern(
    v: str | RegexType | dict[str, Any] | StringPattern | RegexPattern,
) -> StringPattern | RegexPattern:
    """Helper function that turns a string or regex pattern into the appropriate dict."""
    # Users can pass: str, compiled regex, or the tagged dict forms
    if isinstance(v, (StringPattern, RegexPattern)):
        return v
    if isinstance(v, str):
        return StringPattern(String=v)
    if isinstance(v, (regex.Pattern, re.Pattern)):
        return RegexPattern(Regex=v.pattern)
    try:
        # If this is a dict (implicit)
        return PATTERN_ADAPTOR.validate_python(v)
    except ValidationError as e:
        raise TypeError(
            "pattern must be a string, a compiled regex, or a dict like {'String': ...} / {'Regex': ...}"
        ) from e


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

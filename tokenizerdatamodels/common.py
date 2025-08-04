from enum import Enum
from typing import Literal

from pydantic import BaseModel


class StringPattern(BaseModel):
    String: str


class RegexPattern(BaseModel):
    Regex: str


class PrependScheme(str, Enum):
    ALWAYS = "always"
    NEVER = "never"
    FIRST = "first"


class Behavior(str, Enum):
    REMOVED = "Removed"
    ISOLATED = "Isolated"
    MERGED_WITH_PREVIOUS = "MergedWithPrevious"
    MERGED_WITH_NEXT = "MergedWithNext"
    CONTIGUOUS = "Contiguous"

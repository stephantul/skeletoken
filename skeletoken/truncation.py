from enum import Enum

from pydantic import BaseModel


class TruncationDirection(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


class TruncationStrategy(str, Enum):
    LONGEST_FIRST = "LongestFirst"
    ONLY_FIRST = "OnlyFirst"
    ONLY_SECOND = "OnlySecond"


class Truncation(BaseModel):
    """Defines a truncation strategy to use."""

    direction: TruncationDirection
    max_length: int
    strategy: TruncationStrategy
    stride: int

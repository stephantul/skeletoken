from enum import Enum

from pydantic import BaseModel


class TruncationDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class TruncationStrategy(str, Enum):
    LONGEST_FIRST = "longest_first"
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"


class Truncation(BaseModel):
    """Defines a truncation strategy to use."""

    direction: TruncationDirection
    max_length: int
    strategy: TruncationStrategy
    stride: int

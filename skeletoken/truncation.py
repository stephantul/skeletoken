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
    """
    Defines a truncation strategy to use.

    Attributes
    ----------
    direction : TruncationDirection
        The direction from which to truncate. This can be left or right.
    max_length : int
        The maximum length of the sequence after truncation.
    strategy : TruncationStrategy
        The strategy to use for truncation. OnlyFirst will only truncate from the
        first sequence, OnlySecond will only truncate from the second one, while
        LongestFirst will first truncate from the longest sequence, and then truncate
        the remainder from the shorter one.
    stride : int
        The stride to use. Stride here means how much overlap to leave in between
        sequential chunks. For example, if stride is 5, then each chunk contains 5
        chunks of the previous one at the start.

    """

    direction: TruncationDirection
    max_length: int
    strategy: TruncationStrategy
    stride: int

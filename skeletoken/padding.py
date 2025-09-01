from enum import Enum
from typing import Literal

from pydantic import BaseModel, RootModel


class BatchLongestStrategy(RootModel[Literal["BatchLongest"]]):
    """Pad to the longest string in the sequence."""

    ...


class FixedStrategy(BaseModel):
    """Pad to a fixed amount."""

    Fixed: int


# Union of the two shapes
PaddingStrategy = BatchLongestStrategy | FixedStrategy


class PaddingDirection(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


class Padding(BaseModel):
    """A padding configuration."""

    strategy: PaddingStrategy = FixedStrategy(Fixed=0)
    direction: PaddingDirection = PaddingDirection.RIGHT
    pad_to_multiple_of: int | None = None
    pad_id: int
    pad_type_id: int
    pad_token: str

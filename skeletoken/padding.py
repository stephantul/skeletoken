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
    """
    The padding configuration.

    Attributes
    ----------
    strategy : PaddingStrategy
        The padding strategy to use.
        If this is BatchLongest, the input will be padded to the longest sequence in the batch.
        If this is Fixed, the padding will be applied to a fixed amount. This allows for a
        clever hack. By setting the strategy to Fixed, and the amount to 0, you can set a padding
        token without actually padding.
    direction : PaddingDirection
        The direction to pad in. This can be either left or right.
    pad_to_multiple_of : int | None
        If set, the input will be padded to a multiple of this value.
    pad_id : int
        The ID of the padding token in the vocabulary.
    pad_type_id : int
        The type ID to insert when padding. This is usually 0.
    pad_token : str
        The form of the padding token in the vocabulary.

    """

    strategy: PaddingStrategy = FixedStrategy(Fixed=0)
    direction: PaddingDirection = PaddingDirection.RIGHT
    pad_to_multiple_of: int | None = None
    pad_id: int
    pad_type_id: int
    pad_token: str

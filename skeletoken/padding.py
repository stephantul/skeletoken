from enum import Enum

from pydantic import BaseModel


class PaddingStrategy(str, Enum):
    BATCH_LONGEST = "BatchLongest"
    FIXED = "Fixed"


class PaddingDirection(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


class Padding(BaseModel):
    """A padding configuration."""

    strategy: PaddingStrategy
    direction: PaddingDirection
    pad_to_multiple_of: int | None = None
    pad_id: int
    pad_type_id: int
    pad_token: str

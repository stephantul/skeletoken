from enum import Enum

from pydantic import BaseModel


class PaddingStrategy(str, Enum):
    BATCH_LONGEST = "batch_longest"
    FIXED = "fixed"


class PaddingDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Padding(BaseModel):
    """A padding configuration."""

    strategy: PaddingStrategy
    direction: PaddingDirection
    pad_to_multiple_of: int | None = None
    pad_id: int
    pad_type_id: int
    pad_token: str

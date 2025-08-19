from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel


class PostProcessorType(str, Enum):
    SEQUENCE = "Sequence"
    BERT_PROCESSING = "BertProcessing"
    BYTE_LEVEL = "ByteLevel"
    ROBERTA_PROCESSING = "RobertaProcessing"
    TEMPLATE_PROCESSING = "TemplateProcessing"


class PostProcessorSequence(BaseModel):
    """A sequence of postprocessors."""

    type: Literal[PostProcessorType.SEQUENCE] = PostProcessorType.SEQUENCE
    post_processors: list[PostProcessor]


class BertPostProcessor(BaseModel):
    """
    The BERT postprocessor.

    This adds the SEP and CLS tokens to the sequence.
    Note that this processor is actually never used, even BERT uses
    the TemplatePostProcessor, see below.

    Attributes
    ----------
    sep : tuple[str, int]
        The SEP token and its token id.
    cls : tuple[str, int]
        The CLS token and its token id.

    """

    type: Literal[PostProcessorType.BERT_PROCESSING] = PostProcessorType.BERT_PROCESSING
    sep: tuple[str, int]
    cls: tuple[str, int]


class ByteLevelPostProcessor(BaseModel):
    """The ByteLevelPostProcessor. This adds the prefix space, and trims the offsets."""

    type: Literal[PostProcessorType.BYTE_LEVEL] = PostProcessorType.BYTE_LEVEL
    add_prefix_space: bool
    trim_offsets: bool
    use_regex: bool


class RobertaPostProcessor(BaseModel):
    type: Literal[PostProcessorType.ROBERTA_PROCESSING] = PostProcessorType.ROBERTA_PROCESSING
    sep: tuple[str, int]
    cls: tuple[str, int]
    trim_offsets: bool
    add_prefix_space: bool


class TokenContent(BaseModel):
    id: str
    type_id: int


class SpecialToken(BaseModel):
    SpecialToken: TokenContent


class SequenceToken(BaseModel):
    Sequence: TokenContent


class TokenInfo(BaseModel):
    id: str
    ids: list[int]
    tokens: list[str]


class SpecialTokens(RootModel[dict[str, TokenInfo]]): ...


# Simple type alias for a sequence of tokens for a template post-processor.
TokenSequence = tuple[SpecialToken | SequenceToken, ...]


class TemplatePostProcessor(BaseModel):
    type: Literal[PostProcessorType.TEMPLATE_PROCESSING] = PostProcessorType.TEMPLATE_PROCESSING
    single: TokenSequence | None
    pair: TokenSequence | None
    special_tokens: SpecialTokens | None


PostProcessor = (
    BertPostProcessor | ByteLevelPostProcessor | RobertaPostProcessor | TemplatePostProcessor | PostProcessorSequence
)
PostProcessorDiscriminator = Annotated[PostProcessor, Field(discriminator="type")]

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
    processors: list[PostProcessor]


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


class SpecialTokens(RootModel[dict[str, TokenInfo]]):
    def __getitem__(self, key: str) -> TokenInfo:
        """Gets a token."""
        return self.root[key]


# Simple type alias for a sequence of tokens for a template post-processor.
TokenSequence = tuple[SpecialToken | SequenceToken, ...]


class TemplatePostProcessor(BaseModel):
    type: Literal[PostProcessorType.TEMPLATE_PROCESSING] = PostProcessorType.TEMPLATE_PROCESSING
    single: TokenSequence
    pair: TokenSequence
    special_tokens: SpecialTokens


PostProcessor = (
    BertPostProcessor | ByteLevelPostProcessor | RobertaPostProcessor | TemplatePostProcessor | PostProcessorSequence
)
PostProcessorDiscriminator = Annotated[PostProcessor, Field(discriminator="type")]


def get_bos_token_from_post_processor(post_processor: PostProcessor) -> list[str] | None:
    """Get the beginning-of-sequence token from a post-processor."""
    if isinstance(post_processor, PostProcessorSequence):
        return None
    if isinstance(post_processor, ByteLevelPostProcessor):
        return None
    if isinstance(post_processor, (RobertaPostProcessor, BertPostProcessor)):
        return [post_processor.cls[0]]
    if isinstance(post_processor, TemplatePostProcessor):
        single_encoding = post_processor.single
        if not single_encoding:
            return None
        if not isinstance(single_encoding[0], SpecialToken):
            return None
        identifier = single_encoding[0].SpecialToken.id
        return post_processor.special_tokens.root[identifier].tokens


def get_eos_token_from_post_processor(post_processor: PostProcessor) -> list[str] | None:
    """Get the end-of-sequence token from a post-processor."""
    if isinstance(post_processor, PostProcessorSequence):
        return None
    if isinstance(post_processor, ByteLevelPostProcessor):
        return None
    if isinstance(post_processor, (RobertaPostProcessor, BertPostProcessor)):
        return [post_processor.sep[0]]
    if isinstance(post_processor, TemplatePostProcessor):
        single_encoding = post_processor.single
        if not single_encoding:
            return None
        if not isinstance(single_encoding[-1], SpecialToken):
            return None
        identifier = single_encoding[-1].SpecialToken.id
        return post_processor.special_tokens[identifier].tokens


def maybe_replace_token_in_post_processor(
    old_token: str, new_token: str, index: int, post_processor: PostProcessorDiscriminator
) -> PostProcessor:
    """
    Replace a token in a post-processor, if it exists.

    Parameters
    ----------
    old_token : str
        The token to replace.
    new_token : str
        The new token to insert.
    index : int
        The new index to insert.
    post_processor : PostProcessor
        The post-processor to replace the token in.

    """
    post_processor = post_processor.model_copy(deep=True)
    if isinstance(post_processor, PostProcessorSequence):
        return PostProcessorSequence(
            processors=[
                maybe_replace_token_in_post_processor(old_token, new_token, index, p) for p in post_processor.processors
            ]
        )
    if isinstance(post_processor, ByteLevelPostProcessor):
        # Has no tokens.
        return post_processor
    if isinstance(post_processor, (RobertaPostProcessor, BertPostProcessor)):
        # cls and sep can be the same token.
        if old_token == post_processor.cls[0]:
            post_processor.cls = (new_token, index)
        if old_token == post_processor.sep[0]:
            post_processor.sep = (new_token, index)
    if isinstance(post_processor, TemplatePostProcessor):
        for special_token in post_processor.special_tokens.root.values():
            new_tokens = []
            new_ids = []
            for token, id in zip(special_token.tokens, special_token.ids):
                if token == old_token:
                    new_tokens.append(new_token)
                    new_ids.append(index)
                else:
                    new_tokens.append(token)
                    new_ids.append(id)
            special_token.tokens = new_tokens
            special_token.ids = new_ids

    return post_processor

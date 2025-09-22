from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, SerializationInfo, model_serializer, model_validator

logger = logging.getLogger(__name__)


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
    type : Literal[PostProcessorType.BERT_PROCESSING]
        The type of the postprocessor. This is always "BertProcessing".
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
    """
    The RoBERTa postprocessor.

    This adds the SEP and CLS tokens to the sequence.
    Note that this processor is actually never used, even RoBERTa uses
    the TemplatePostProcessor, see below.

    Attributes
    ----------
    type : Literal[PostProcessorType.ROBERTA_PROCESSING]
        The type of the postprocessor. This is always "RobertaProcessing".
    sep : tuple[str, int]
        The SEP token and its token id.
    cls : tuple[str, int]
        The CLS token and its token id.
    trim_offsets : bool
        Whether to trim the offsets of the tokens.
    add_prefix_space : bool
        Whether to add a space before the first token. This leads to more consistent
        behavior for sentence-initial tokens, and is recommended to be set to True.

    """

    type: Literal[PostProcessorType.ROBERTA_PROCESSING] = PostProcessorType.ROBERTA_PROCESSING
    sep: tuple[str, int]
    cls: tuple[str, int]
    trim_offsets: bool
    add_prefix_space: bool


class TokenType(str, Enum):
    SPECIAL = "SpecialToken"
    SEQUENCE = "Sequence"


class Token(BaseModel):
    id: str
    type_id: int
    type: TokenType

    @model_validator(mode="before")
    @classmethod
    def parse_format(cls, v) -> dict:
        """Parse either {'SpecialToken': {...}} or {'Sequence': {...}}."""
        if isinstance(v, dict):
            # Two types:
            # 1. {"Sequence": {... }} or {"SpecialToken": {...}}
            # 2. {"id": ..., "type_id": ..., "type": ...}
            if len(v) == 1:
                t = next(iter(v))
                v = v[t]
            else:
                t = v["type"]
            inner = {**v}
            inner["type"] = TokenType(t)
            return inner
        raise TypeError("Token must be a dict or a Token instance.")

    @model_serializer
    def serializer(self, info: SerializationInfo) -> dict[str, Any]:
        """Serialization to either {'SpecialToken': {...}} or {'Sequence': {...}}."""
        data = {"id": self.id, "type_id": self.type_id}
        return {self.type: data}


class SpecialTokenInfo(BaseModel):
    """Information about a special token used in a template post-processor."""

    id: str
    ids: list[int]
    tokens: list[str]

    def model_post_init(self, __context: dict) -> None:
        """Validates that ids and tokens have the same length."""
        if len(self.ids) != len(self.tokens):
            logger.warning("ids and tokens must have the same length. Ids: %s, tokens: %s", self.ids, self.tokens)


# Simple type alias for a sequence of tokens for a template post-processor.
TokenSequence = tuple[Token, ...]


class TemplatePostProcessor(BaseModel):
    type: Literal[PostProcessorType.TEMPLATE_PROCESSING] = PostProcessorType.TEMPLATE_PROCESSING
    single: TokenSequence
    pair: TokenSequence
    special_tokens: dict[str, SpecialTokenInfo]

    def model_post_init(self, __context: dict) -> None:
        """Validates that all special tokens in single and pair are defined in special_tokens."""
        for token_name, t in self.special_tokens.items():
            if token_name != t.id:
                raise ValueError(f"Special token name {token_name} does not match its id {t.id}.")
        num_types = _count_and_check_types(self.single, self.special_tokens)
        if num_types != 1:
            raise ValueError("Single sequence template can only have one sequence type.")
        num_types = _count_and_check_types(self.pair, self.special_tokens)
        if num_types != 2:
            raise ValueError("Pair sequence template can only have two sequence types.")


def _count_and_check_types(tokens: TokenSequence, special_tokens: dict[str, SpecialTokenInfo]) -> int:
    """Count the number of unique type_ids in a sequence of tokens."""
    sequence_tokens = 0
    unique_types = set()
    for token in tokens:
        if token.type == TokenType.SEQUENCE:
            sequence_tokens += 1
            unique_types.add(token.id)
        if token.type == TokenType.SPECIAL and token.id not in special_tokens:
            raise ValueError(f"Special token {token.id} is not defined in special_tokens.")
    if sequence_tokens != len(unique_types):
        raise ValueError("All sequence tokens must have a unique id.")
    return sequence_tokens


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
        if single_encoding[0].type != TokenType.SPECIAL:
            return None
        identifier = single_encoding[0].id
        return post_processor.special_tokens[identifier].tokens


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
        if single_encoding[-1].type != TokenType.SPECIAL:
            return None
        identifier = single_encoding[-1].id
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
    post_processor : PostProcessorDiscriminator
        The post-processor to replace the token in.

    Returns
    -------
    PostProcessor
        The post-processor with the token replaced, if it existed.

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
        for special_token in post_processor.special_tokens.values():
            new_tokens = []
            new_ids = []
            for token, id in zip(special_token.tokens, special_token.ids, strict=True):
                if token == old_token:
                    new_tokens.append(new_token)
                    new_ids.append(index)
                else:
                    new_tokens.append(token)
                    new_ids.append(id)
            special_token.tokens = new_tokens
            special_token.ids = new_ids

    return post_processor

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any, Literal, Type

from pydantic import BaseModel, Field, SerializationInfo, model_serializer, model_validator

logger = logging.getLogger(__name__)

_PROMPT_SPECIAL_TOKEN_ID = "PROMPT"


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
    """The BERT postprocessor.

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
    """The RoBERTa postprocessor.

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
    def parse_format(cls, v: Any) -> dict[Any, Any]:
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
        """Serialize to either {'SpecialToken': {...}} or {'Sequence': {...}}."""
        data = {"id": self.id, "type_id": self.type_id}
        return {self.type: data}


class SpecialTokenInfo(BaseModel):
    """Information about a special token used in a template post-processor."""

    id: str
    ids: list[int]
    tokens: list[str]

    def model_post_init(self, __context: dict[Any, Any]) -> None:
        """Validate that ids and tokens have the same length."""
        if len(self.ids) != len(self.tokens):
            logger.warning("ids and tokens must have the same length. Ids: %s, tokens: %s", self.ids, self.tokens)


# Simple type alias for a sequence of tokens for a template post-processor.
TokenSequence = tuple[Token, ...]

# Special tokens can be sequences or single strings
SpecialTokenTuple = tuple[list[str], list[int]] | tuple[str, int]


def _special_token_info(fallback_id: str, t: SpecialTokenTuple) -> SpecialTokenInfo:
    """Helper to create a SpecialTokenInfo object."""
    match t:
        case str() as token, int() as token_id:
            return SpecialTokenInfo(id=token, ids=[token_id], tokens=[token])
        case tokens, ids:
            return SpecialTokenInfo(id=fallback_id, ids=list(ids), tokens=list(tokens))


class TemplatePostProcessor(BaseModel):
    type: Literal[PostProcessorType.TEMPLATE_PROCESSING] = PostProcessorType.TEMPLATE_PROCESSING
    single: TokenSequence
    pair: TokenSequence
    special_tokens: dict[str, SpecialTokenInfo]

    @classmethod
    def from_tokens(
        cls: Type[TemplatePostProcessor],
        start_of_sequence_token: SpecialTokenTuple | None,
        continuation_token: SpecialTokenTuple | None,
        start_name: str = "START",
        continuation_name: str = "CONTINUATION",
    ) -> TemplatePostProcessor:
        """Constructs a standard TemplatePostProcessor."""
        # Implicit overwriting: if both tokens have the same form, we still end up with one.
        if start_of_sequence_token is None and continuation_token is None:
            raise ValueError("At least one of start_of_sequence_token or continuation_token must be provided")

        start_info = _special_token_info(start_name, start_of_sequence_token) if start_of_sequence_token else None
        continuation_info = _special_token_info(continuation_name, continuation_token) if continuation_token else None
        special_tokens = {info.id: info for info in (start_info, continuation_info) if info is not None}

        start_token_object = Token(id=start_info.id, type_id=0, type=TokenType.SPECIAL) if start_info else None
        continuation_after_a = (
            Token(id=continuation_info.id, type_id=0, type=TokenType.SPECIAL) if continuation_info else None
        )
        continuation_after_b = (
            Token(id=continuation_info.id, type_id=1, type=TokenType.SPECIAL) if continuation_info else None
        )
        a_sequence = Token(id="A", type_id=0, type=TokenType.SEQUENCE)
        b_sequence = Token(id="B", type_id=1, type=TokenType.SEQUENCE)
        single_sequence = [x for x in [start_token_object, a_sequence, continuation_after_a] if x is not None]
        pair_sequence = [
            x
            for x in [start_token_object, a_sequence, continuation_after_a, b_sequence, continuation_after_b]
            if x is not None
        ]
        return cls(single=tuple(single_sequence), pair=tuple(pair_sequence), special_tokens=special_tokens)

    def model_post_init(self, __context: dict[Any, Any]) -> None:
        """Validate that all special tokens in single and pair are defined in special_tokens."""
        for token_name, t in self.special_tokens.items():
            if token_name != t.id:
                logger.warning(
                    f"Special token name '{token_name}' does not match its id '{t.id}'. "
                    "This leads to mismatches when inspecting the string and ids."
                )
        num_types = _count_and_check_types(self.single, self.special_tokens)
        if num_types != 1:
            logger.warning(f"Single sequence template must have exactly one sequence type, found {num_types}")
        num_types = _count_and_check_types(self.pair, self.special_tokens)
        if num_types != 2:
            logger.warning(f"Pair sequence template must have exactly two sequence types, found {num_types}")


def _count_and_check_types(tokens: TokenSequence, special_tokens: dict[str, SpecialTokenInfo]) -> int:
    """Count the number of unique type_ids in a sequence of tokens."""
    sequence_tokens = 0
    unique_types = set()
    for token in tokens:
        if token.type == TokenType.SEQUENCE:
            sequence_tokens += 1
            unique_types.add(token.id)
        if token.type == TokenType.SPECIAL and token.id not in special_tokens:
            logger.warning(f"Special token {token.id} is not defined in special_tokens.")
    if sequence_tokens != len(unique_types):
        logger.warning("All sequence tokens must have a unique id.")
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
    if isinstance(post_processor, RobertaPostProcessor | BertPostProcessor):
        return [post_processor.cls[0]]
    if isinstance(post_processor, TemplatePostProcessor):  # type: ignore
        single_encoding = post_processor.single
        if not single_encoding or single_encoding[0].type != TokenType.SPECIAL:
            return None
        identifier = single_encoding[0].id
        info = post_processor.special_tokens.get(identifier)
        return info.tokens if info is not None else None


def get_eos_token_from_post_processor(post_processor: PostProcessor) -> list[str] | None:
    """Get the end-of-sequence token from a post-processor."""
    if isinstance(post_processor, PostProcessorSequence):
        return None
    if isinstance(post_processor, ByteLevelPostProcessor):
        return None
    if isinstance(post_processor, RobertaPostProcessor | BertPostProcessor):
        return [post_processor.sep[0]]
    if isinstance(post_processor, TemplatePostProcessor):  # type: ignore
        single_encoding = post_processor.single
        if not single_encoding or single_encoding[-1].type != TokenType.SPECIAL:
            return None
        identifier = single_encoding[-1].id
        info = post_processor.special_tokens.get(identifier)
        return info.tokens if info is not None else None


def get_prompt_from_post_processor(post_processor: PostProcessor | None) -> list[str] | None:
    """Get the prompt tokens inserted right before every sequence, if any."""
    if not isinstance(post_processor, TemplatePostProcessor):
        return None
    info = post_processor.special_tokens.get(_PROMPT_SPECIAL_TOKEN_ID)
    if info is None or not info.tokens:
        return None
    return info.tokens


def _prompt_insertion_index(tokens: TokenSequence) -> int:
    """Index of a prompt: right after an initial special token, else the front."""
    if not tokens:
        return 0
    return tokens[0].type == TokenType.SPECIAL


def set_prompt_in_post_processor(
    post_processor: PostProcessor | None, tokens: list[str], ids: list[int]
) -> PostProcessor:
    """Set (or replace) the prompt inserted right before every sequence.

    The prompt gets a special-token slot, inserted right after a real beginning-of-
    sequence token if the post-processor has one (e.g. `[CLS]`), or at the front otherwise.

    Parameters
    ----------
    post_processor : PostProcessor | None
        The post-processor to set the prompt on, or `None` if the tokenizer has none yet.
    tokens : list[str]
        The prompt's tokens.
    ids : list[int]
        The prompt tokens' vocabulary ids, matched by position to `tokens`.

    Returns
    -------
    PostProcessor
        The post-processor with the prompt set.

    Raises
    ------
    ValueError
        If `post_processor` is a type that cannot hold a prompt (anything other than `None` or a
        `TemplatePostProcessor`).

    """
    if post_processor is not None and not isinstance(post_processor, TemplatePostProcessor):
        raise ValueError(
            f"Cannot set a prompt: the tokenizer has a {type(post_processor).__name__} post-processor, "
            "which cannot hold a prompt. Convert it to a TemplatePostProcessor first."
        )

    prompt_info = SpecialTokenInfo(id=_PROMPT_SPECIAL_TOKEN_ID, tokens=list(tokens), ids=list(ids))
    prompt_token = Token(id=_PROMPT_SPECIAL_TOKEN_ID, type_id=0, type=TokenType.SPECIAL)

    if post_processor is None:
        return TemplatePostProcessor.from_tokens((tokens, ids), None, _PROMPT_SPECIAL_TOKEN_ID)

    # The postprocessor already has a prompt. Overwrite it.
    if _PROMPT_SPECIAL_TOKEN_ID in post_processor.special_tokens:
        post_processor.special_tokens[_PROMPT_SPECIAL_TOKEN_ID] = prompt_info
        return post_processor

    # It doesn't have one: insert it after BOS.
    single_index = _prompt_insertion_index(post_processor.single)
    pair_index = _prompt_insertion_index(post_processor.pair)
    post_processor.special_tokens[_PROMPT_SPECIAL_TOKEN_ID] = prompt_info
    post_processor.single = (*post_processor.single[:single_index], prompt_token, *post_processor.single[single_index:])
    post_processor.pair = (*post_processor.pair[:pair_index], prompt_token, *post_processor.pair[pair_index:])
    return post_processor


def unset_prompt_in_post_processor(post_processor: PostProcessor | None) -> PostProcessor | None:
    """Remove a previously set prompt, if any, leaving any other special tokens."""
    if not isinstance(post_processor, TemplatePostProcessor):
        return post_processor
    if _PROMPT_SPECIAL_TOKEN_ID not in post_processor.special_tokens:
        return post_processor

    def _is_prompt_slot(t: Token) -> bool:
        return t.type == TokenType.SPECIAL and t.id == _PROMPT_SPECIAL_TOKEN_ID

    post_processor.single = tuple(t for t in post_processor.single if not _is_prompt_slot(t))
    post_processor.pair = tuple(t for t in post_processor.pair if not _is_prompt_slot(t))
    post_processor.special_tokens.pop(_PROMPT_SPECIAL_TOKEN_ID, None)
    if not post_processor.special_tokens:
        return None
    return post_processor


def get_tokens_from_post_processor(post_processor: PostProcessor) -> set[str]:
    """Get all token strings referenced by a post-processor."""
    if isinstance(post_processor, PostProcessorSequence):
        tokens: set[str] = set()
        for p in post_processor.processors:
            tokens |= get_tokens_from_post_processor(p)
        return tokens
    if isinstance(post_processor, ByteLevelPostProcessor):
        return set()
    if isinstance(post_processor, RobertaPostProcessor | BertPostProcessor):
        return {post_processor.cls[0], post_processor.sep[0]}
    if isinstance(post_processor, TemplatePostProcessor):  # type: ignore
        identifiers = {
            token.id
            for sequence in (post_processor.single, post_processor.pair)
            for token in sequence
            if token.type == TokenType.SPECIAL
        }
        return {
            token
            for identifier in identifiers
            if (info := post_processor.special_tokens.get(identifier)) is not None
            for token in info.tokens
        }


def maybe_replace_token_in_post_processor(
    old_token: str, new_token: str, index: int, post_processor: PostProcessorDiscriminator
) -> PostProcessor:
    """Replace a token in a post-processor, if it exists.

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
    if isinstance(post_processor, PostProcessorSequence):
        return PostProcessorSequence(
            processors=[
                maybe_replace_token_in_post_processor(old_token, new_token, index, p) for p in post_processor.processors
            ]
        )
    post_processor = post_processor.model_copy(deep=True)
    if isinstance(post_processor, ByteLevelPostProcessor):
        # Has no tokens.
        return post_processor
    if isinstance(post_processor, RobertaPostProcessor | BertPostProcessor):
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


def resync_post_processor_ids(post_processor: PostProcessorDiscriminator, vocabulary: dict[str, int]) -> PostProcessor:
    """Rewrite every id in a post-processor's special tokens to match the current vocabulary.

    Unlike `maybe_replace_token_in_post_processor`, this does not rename any tokens: for each
    token string already stored in the post-processor, it looks up that token's current id in
    `vocabulary` and overwrites the stored id with it. This repairs ids left stale by vocabulary
    compaction (removal or consolidation), regardless of whether the token is a registered added
    token.

    Parameters
    ----------
    post_processor : PostProcessorDiscriminator
        The post-processor to resync.
    vocabulary : dict[str, int]
        The current token -> id mapping to resync against.

    Returns
    -------
    PostProcessor
        The post-processor with all special token ids matching `vocabulary`.

    Raises
    ------
    ValueError
        If a token the post-processor references is not in `vocabulary`.

    """
    if isinstance(post_processor, PostProcessorSequence):
        return PostProcessorSequence(
            processors=[resync_post_processor_ids(p, vocabulary) for p in post_processor.processors]
        )
    post_processor = post_processor.model_copy(deep=True)
    if isinstance(post_processor, RobertaPostProcessor | BertPostProcessor):
        for token in (post_processor.cls[0], post_processor.sep[0]):
            if token not in vocabulary:
                raise ValueError(f"Post-processor token '{token}' is no longer in the vocabulary.")
        post_processor.cls = (post_processor.cls[0], vocabulary[post_processor.cls[0]])
        post_processor.sep = (post_processor.sep[0], vocabulary[post_processor.sep[0]])
    if isinstance(post_processor, TemplatePostProcessor):
        identifiers = {
            token.id
            for sequence in (post_processor.single, post_processor.pair)
            for token in sequence
            if token.type == TokenType.SPECIAL
        }
        for identifier in identifiers:
            special_token = post_processor.special_tokens.get(identifier)
            if special_token is None:
                continue
            for token in special_token.tokens:
                if token not in vocabulary:
                    raise ValueError(f"Post-processor token '{token}' is no longer in the vocabulary.")
            special_token.ids = [vocabulary[token] for token in special_token.tokens]

    return post_processor

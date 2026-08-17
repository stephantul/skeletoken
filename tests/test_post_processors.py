import logging
from typing import Any

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.post_processors import (
    BertPostProcessor,
    ByteLevelPostProcessor,
    PostProcessor,
    PostProcessorSequence,
    PostProcessorType,
    RobertaPostProcessor,
    SpecialTokenInfo,
    TemplatePostProcessor,
    Token,
    TokenSequence,
    TokenType,
    get_bos_token_from_post_processor,
    get_eos_token_from_post_processor,
    get_tokens_from_post_processor,
    maybe_replace_token_in_post_processor,
)
from tests.conftest import call_tokenizer


def _get_default_postprocessor(post_processor_type: PostProcessorType) -> PostProcessor:  # noqa: C901
    """Get the default instantiation of a normalizer."""
    if post_processor_type == PostProcessorType.BYTE_LEVEL:
        return ByteLevelPostProcessor(trim_offsets=True, add_prefix_space=False, use_regex=False)
    elif post_processor_type == PostProcessorType.BERT_PROCESSING:
        return BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0))
    elif post_processor_type == PostProcessorType.ROBERTA_PROCESSING:
        return RobertaPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False)
    elif post_processor_type == PostProcessorType.TEMPLATE_PROCESSING:
        return TemplatePostProcessor(
            single=(
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
            ),
            pair=(
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
                Token(id="sequence2", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
            ),
            special_tokens={
                "special_begin": SpecialTokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
                "special_end": SpecialTokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
                "sequence": SpecialTokenInfo(id="sequence", ids=[2], tokens=["[SEQ]"]),
            },
        )
    elif post_processor_type == PostProcessorType.SEQUENCE:
        return PostProcessorSequence(
            processors=[ByteLevelPostProcessor(trim_offsets=True, add_prefix_space=False, use_regex=False)]
        )
    else:
        raise ValueError(f"Unknown normalizer type: {post_processor_type}")


@pytest.mark.parametrize("post_processor_type", [PostProcessorType.BYTE_LEVEL, PostProcessorType.BERT_PROCESSING])
def test_post_processor(small_tokenizer_json: dict[str, Any], post_processor_type: PostProcessorType) -> None:
    """Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    normalizer = _get_default_postprocessor(post_processor_type)
    normalizer_dict = normalizer.model_dump()
    small_tokenizer_json["post_processor"] = normalizer_dict
    tokenizer = TokenizerModel.model_validate(small_tokenizer_json)

    assert tokenizer.post_processor is not None
    assert tokenizer.post_processor.type == post_processor_type

    call_tokenizer(tokenizer)


def _get_no_eos_template() -> TemplatePostProcessor:
    """Get a template processor without eos."""
    return TemplatePostProcessor(
        single=(
            Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
            Token(id="sequence", type_id=0, type=TokenType.SEQUENCE),
        ),
        pair=(
            Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
            Token(id="sequence1", type_id=0, type=TokenType.SEQUENCE),
            Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
            Token(id="sequence2", type_id=1, type=TokenType.SEQUENCE),
        ),
        special_tokens={
            "special_begin": SpecialTokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
        },
    )


def _get_no_bos_template() -> TemplatePostProcessor:
    """Get a template processor without bos."""
    return TemplatePostProcessor(
        single=(
            Token(id="sequence", type_id=0, type=TokenType.SEQUENCE),
            Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
        ),
        pair=(
            Token(id="sequence", type_id=0, type=TokenType.SEQUENCE),
            Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
            Token(id="sequence2", type_id=1, type=TokenType.SEQUENCE),
            Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
        ),
        special_tokens={
            "special_end": SpecialTokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
        },
    )


@pytest.mark.parametrize(
    "post_processor,result",
    [
        (_get_default_postprocessor(PostProcessorType.SEQUENCE), None),
        (_get_default_postprocessor(PostProcessorType.BYTE_LEVEL), None),
        (_get_default_postprocessor(PostProcessorType.BERT_PROCESSING), ["[CLS]"]),
        (_get_default_postprocessor(PostProcessorType.ROBERTA_PROCESSING), ["[CLS]"]),
        (_get_default_postprocessor(PostProcessorType.TEMPLATE_PROCESSING), ["[BEGIN]"]),
        (_get_no_eos_template(), ["[BEGIN]"]),
        (_get_no_bos_template(), None),
    ],
)
def test_get_bos_token_from_post_processor(post_processor: PostProcessor, result: str | None) -> None:
    """Tests getting the bos token from the post processor."""
    bos_token = get_bos_token_from_post_processor(post_processor)
    assert bos_token == result


@pytest.mark.parametrize(
    "post_processor,result",
    [
        (_get_default_postprocessor(PostProcessorType.SEQUENCE), None),
        (_get_default_postprocessor(PostProcessorType.BYTE_LEVEL), None),
        (_get_default_postprocessor(PostProcessorType.BERT_PROCESSING), ["[SEP]"]),
        (_get_default_postprocessor(PostProcessorType.ROBERTA_PROCESSING), ["[SEP]"]),
        (_get_default_postprocessor(PostProcessorType.TEMPLATE_PROCESSING), ["[END]"]),
        (_get_no_eos_template(), None),
        (_get_no_bos_template(), ["[END]"]),
    ],
)
def test_get_eos_token_from_post_processor(post_processor: PostProcessor, result: str | None) -> None:
    """Tests getting the eos token from the post processor."""
    eos_token = get_eos_token_from_post_processor(post_processor)
    assert eos_token == result


@pytest.mark.parametrize(
    "post_processor,old_token,new_token",
    [
        (_get_default_postprocessor(PostProcessorType.SEQUENCE), "a", "b"),
        (_get_default_postprocessor(PostProcessorType.BYTE_LEVEL), "a", "b"),
        (_get_default_postprocessor(PostProcessorType.BERT_PROCESSING), "[CLS]", "[AB]"),
        (_get_default_postprocessor(PostProcessorType.ROBERTA_PROCESSING), "[CLS]", "[AB]"),
        (_get_default_postprocessor(PostProcessorType.BERT_PROCESSING), "[SEP]", "[AB]"),
        (_get_default_postprocessor(PostProcessorType.ROBERTA_PROCESSING), "[SEP]", "[AB]"),
        (_get_default_postprocessor(PostProcessorType.TEMPLATE_PROCESSING), "[END]", "[END]"),
    ],
)
def test_maybe_replace_token_in_post_processor(post_processor: PostProcessor, old_token: str, new_token: str) -> None:
    """Tests maybe replace the token in post processor."""
    result = maybe_replace_token_in_post_processor(old_token, new_token, 11, post_processor)
    if isinstance(result, PostProcessorSequence):
        # No change, because it has a single post-processor.
        assert result == post_processor
    if isinstance(result, RobertaPostProcessor | BertPostProcessor) and old_token == "[CLS]":
        assert result.cls == (new_token, 11)
        assert result.sep == ("[SEP]", 1)
    if isinstance(result, RobertaPostProcessor | BertPostProcessor) and old_token == "[SEP]":
        assert result.cls == ("[CLS]", 0)
        assert result.sep == (new_token, 11)
    if isinstance(result, TemplatePostProcessor):
        assert result.special_tokens["special_end"].tokens == [new_token]


@pytest.mark.parametrize(
    "post_processor,result",
    [
        (_get_default_postprocessor(PostProcessorType.SEQUENCE), set()),
        (_get_default_postprocessor(PostProcessorType.BYTE_LEVEL), set()),
        (_get_default_postprocessor(PostProcessorType.BERT_PROCESSING), {"[CLS]", "[SEP]"}),
        (_get_default_postprocessor(PostProcessorType.ROBERTA_PROCESSING), {"[CLS]", "[SEP]"}),
        (_get_default_postprocessor(PostProcessorType.TEMPLATE_PROCESSING), {"[BEGIN]", "[END]"}),
        (
            PostProcessorSequence(
                processors=[
                    ByteLevelPostProcessor(trim_offsets=True, add_prefix_space=False, use_regex=False),
                    BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0)),
                ]
            ),
            {"[CLS]", "[SEP]"},
        ),
    ],
)
def test_get_tokens_from_post_processor(post_processor: PostProcessor, result: set[str]) -> None:
    """Tests getting all token strings referenced by a post-processor."""
    tokens = get_tokens_from_post_processor(post_processor)
    assert tokens == result


def test_parse_token() -> None:
    """Test the Token parsing."""
    t = Token(id="A", type_id=0, type=TokenType.SPECIAL)
    assert t.id == "A"
    assert t.type_id == 0
    assert t.type == TokenType.SPECIAL

    t_dict = {"id": "B", "type_id": 1, "type": "Sequence"}
    t2 = Token.model_validate(t_dict)
    assert t2.id == "B"
    assert t2.type_id == 1
    assert t2.type == TokenType.SEQUENCE

    t_dict = {"Sequence": {"id": "B", "type_id": 1}}
    t2 = Token.model_validate(t_dict)
    assert t2.id == "B"
    assert t2.type_id == 1
    assert t2.type == TokenType.SEQUENCE

    t3 = Token.model_validate(t2)
    assert t3 == t2

    with pytest.raises(TypeError):
        Token.model_validate(11)  # type: ignore[arg-type]


def test_unequal_special_token(caplog: Any) -> None:
    """Test that unequal special tokens are not equal."""
    with caplog.at_level(logging.WARNING):
        SpecialTokenInfo(id="A", ids=[0], tokens=["a", "b"])

    assert "ids and tokens must have the same length." in caplog.text
    assert caplog.records[0].levelname == "WARNING"


def test_template_missing_special_token(caplog: Any) -> None:
    """Test that incorrect tokens in template raise errors."""
    with caplog.at_level(logging.WARNING):
        TemplatePostProcessor(
            single=(
                Token(id="missing_special", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
            ),
            pair=(
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
            ),
            special_tokens={
                "special_begin": SpecialTokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
                "special_end": SpecialTokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
            },
        )

    assert "Special token missing_special is not defined in special_tokens" in caplog.text
    assert caplog.records[0].levelname == "WARNING"


def test_template_incorrectly_named_special_token(caplog: Any) -> None:
    """Test that incorrect tokens in template raise errors."""
    with caplog.at_level(logging.WARNING):
        TemplatePostProcessor(
            single=(
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
            ),
            pair=(
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
            ),
            special_tokens={
                "special_begin": SpecialTokenInfo(id="special_missing", ids=[0], tokens=["[BEGIN]"]),
                "special_end": SpecialTokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
            },
        )

    assert (
        "Special token name 'special_begin' does not match its id 'special_missing'. "
        "This leads to mismatches when inspecting the string and ids" in caplog.text
    )
    assert caplog.records[0].levelname == "WARNING"


def test_template_too_many_tokens(caplog: Any) -> None:
    """Test whether there are too many tokens in a template."""
    with caplog.at_level(logging.WARNING):
        TemplatePostProcessor(
            single=(Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),),
            pair=(Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),),
            special_tokens={
                "special_begin": SpecialTokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
                "special_end": SpecialTokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
            },
        )

    assert "Single sequence template must have exactly one sequence type" in caplog.text
    assert caplog.records[0].levelname == "WARNING"


@pytest.mark.parametrize(
    "single,pair,failure",
    [
        [(), (), "Single sequence template must have exactly one sequence type,"],
        [
            (
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=0, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
            ),
            (
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=0, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
            ),
            "Pair sequence template must have exactly two sequence types,",
        ],
        [
            (
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=0, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
            ),
            (
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence1", type_id=0, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence1", type_id=0, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=0, type=TokenType.SPECIAL),
            ),
            "All sequence tokens must have a unique id.",
        ],
        [
            (
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
            ),
            (
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence1", type_id=2, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SPECIAL),
                Token(id="sequence2", type_id=1, type=TokenType.SEQUENCE),
                Token(id="special_begin", type_id=0, type=TokenType.SPECIAL),
                Token(id="sequence3", type_id=1, type=TokenType.SEQUENCE),
                Token(id="special_end", type_id=1, type=TokenType.SEQUENCE),
            ),
            "Pair sequence template must have exactly two sequence types,",
        ],
    ],
)
def test_template_creation_failure(caplog: Any, single: TokenSequence, pair: TokenSequence, failure: str) -> None:
    """Tests incorrect template creation."""
    with caplog.at_level(logging.WARNING):
        TemplatePostProcessor(
            single=single,
            pair=pair,
            special_tokens={
                "special_begin": SpecialTokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
                "special_end": SpecialTokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
            },
        )

    assert failure in caplog.text
    assert caplog.records[0].levelname == "WARNING"


def test_template_post_processor_from_tokens() -> None:
    """Test that from_tokens builds the standard BERT-style template."""
    post_processor = TemplatePostProcessor.from_tokens(("[CLS]", 0), ("[SEP]", 1))

    assert [(t.id, t.type_id, t.type) for t in post_processor.single] == [
        ("[CLS]", 0, TokenType.SPECIAL),
        ("A", 0, TokenType.SEQUENCE),
        ("[SEP]", 0, TokenType.SPECIAL),
    ]
    assert [(t.id, t.type_id, t.type) for t in post_processor.pair] == [
        ("[CLS]", 0, TokenType.SPECIAL),
        ("A", 0, TokenType.SEQUENCE),
        ("[SEP]", 0, TokenType.SPECIAL),
        ("B", 1, TokenType.SEQUENCE),
        ("[SEP]", 1, TokenType.SPECIAL),
    ]
    assert post_processor.special_tokens == {
        "[CLS]": SpecialTokenInfo(id="[CLS]", ids=[0], tokens=["[CLS]"]),
        "[SEP]": SpecialTokenInfo(id="[SEP]", ids=[1], tokens=["[SEP]"]),
    }
    assert get_bos_token_from_post_processor(post_processor) == ["[CLS]"]
    assert get_eos_token_from_post_processor(post_processor) == ["[SEP]"]


def test_template_post_processor_from_tokens_only_start() -> None:
    """Test from_tokens with only a start-of-sequence token."""
    post_processor = TemplatePostProcessor.from_tokens(("<s>", 0), None)

    assert [(t.id, t.type_id, t.type) for t in post_processor.single] == [
        ("<s>", 0, TokenType.SPECIAL),
        ("A", 0, TokenType.SEQUENCE),
    ]
    assert [(t.id, t.type_id, t.type) for t in post_processor.pair] == [
        ("<s>", 0, TokenType.SPECIAL),
        ("A", 0, TokenType.SEQUENCE),
        ("B", 1, TokenType.SEQUENCE),
    ]
    assert get_bos_token_from_post_processor(post_processor) == ["<s>"]
    assert get_eos_token_from_post_processor(post_processor) is None


def test_template_post_processor_from_tokens_only_continuation() -> None:
    """Test from_tokens with only a continuation token."""
    post_processor = TemplatePostProcessor.from_tokens(None, ("</s>", 1))

    assert [(t.id, t.type_id, t.type) for t in post_processor.single] == [
        ("A", 0, TokenType.SEQUENCE),
        ("</s>", 0, TokenType.SPECIAL),
    ]
    assert [(t.id, t.type_id, t.type) for t in post_processor.pair] == [
        ("A", 0, TokenType.SEQUENCE),
        ("</s>", 0, TokenType.SPECIAL),
        ("B", 1, TokenType.SEQUENCE),
        ("</s>", 1, TokenType.SPECIAL),
    ]
    assert get_bos_token_from_post_processor(post_processor) is None
    assert get_eos_token_from_post_processor(post_processor) == ["</s>"]


def test_template_post_processor_from_tokens_requires_a_token() -> None:
    """Test that from_tokens raises if neither token is provided."""
    with pytest.raises(ValueError, match="At least one of"):
        TemplatePostProcessor.from_tokens(None, None)


def test_template_post_processor_from_tokens_tokenizer(small_tokenizer_json: dict[str, Any]) -> None:
    """Test that a from_tokens template can be loaded and used by a real tokenizer."""
    post_processor = TemplatePostProcessor.from_tokens(("[CLS]", 0), ("[SEP]", 1))
    small_tokenizer_json["post_processor"] = post_processor.model_dump()
    tokenizer = TokenizerModel.model_validate(small_tokenizer_json)

    call_tokenizer(tokenizer)

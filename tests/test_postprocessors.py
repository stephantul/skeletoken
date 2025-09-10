from typing import Any

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.postprocessors import (
    BertPostProcessor,
    ByteLevelPostProcessor,
    PostProcessor,
    PostProcessorSequence,
    PostProcessorType,
    RobertaPostProcessor,
    SequenceToken,
    SpecialToken,
    SpecialTokens,
    TemplatePostProcessor,
    TokenContent,
    TokenInfo,
    get_bos_token_from_post_processor,
    get_eos_token_from_post_processor,
    maybe_replace_token_in_post_processor,
)


def _get_default_postprocessor(post_processor_type: PostProcessorType) -> PostProcessor:  # noqa: C901
    """Helper function to get the default instantiation of a normalizer."""
    if post_processor_type == PostProcessorType.BYTE_LEVEL:
        return ByteLevelPostProcessor(trim_offsets=True, add_prefix_space=False, use_regex=False)
    elif post_processor_type == PostProcessorType.BERT_PROCESSING:
        return BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0))
    elif post_processor_type == PostProcessorType.ROBERTA_PROCESSING:
        return RobertaPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False)
    elif post_processor_type == PostProcessorType.TEMPLATE_PROCESSING:
        return TemplatePostProcessor(
            single=(
                SpecialToken(SpecialToken=TokenContent(id="special_begin", type_id=0)),
                SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
                SpecialToken(SpecialToken=TokenContent(id="special_end", type_id=1)),
            ),
            pair=(
                SpecialToken(SpecialToken=TokenContent(id="special_begin", type_id=0)),
                SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
                SpecialToken(SpecialToken=TokenContent(id="special_end", type_id=1)),
                SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
                SpecialToken(SpecialToken=TokenContent(id="special_end", type_id=1)),
            ),
            special_tokens=SpecialTokens(
                {
                    "special_begin": TokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
                    "special_end": TokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
                    "sequence": TokenInfo(id="sequence", ids=[2], tokens=["[SEQ]"]),
                }
            ),
        )
    elif post_processor_type == PostProcessorType.SEQUENCE:
        return PostProcessorSequence(
            processors=[ByteLevelPostProcessor(trim_offsets=True, add_prefix_space=False, use_regex=False)]
        )
    else:
        raise ValueError(f"Unknown normalizer type: {post_processor_type}")


@pytest.mark.parametrize("post_processor_type", [PostProcessorType.BYTE_LEVEL, PostProcessorType.BERT_PROCESSING])
def test_post_processor(small_tokenizer_json: dict[str, Any], post_processor_type: PostProcessorType) -> None:
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    normalizer = _get_default_postprocessor(post_processor_type)
    normalizer_dict = normalizer.model_dump()
    small_tokenizer_json["post_processor"] = normalizer_dict
    tokenizer = TokenizerModel.model_validate(small_tokenizer_json)

    assert tokenizer.post_processor is not None
    assert tokenizer.post_processor.type == post_processor_type

    # Implicit test. If this fails, the model is incorrect.
    tokenizer.to_tokenizer()


def _get_no_single_template() -> TemplatePostProcessor:
    """Gets a template with missing things."""
    return TemplatePostProcessor(
        single=(),
        pair=(
            SpecialToken(SpecialToken=TokenContent(id="special_begin", type_id=0)),
            SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
            SpecialToken(SpecialToken=TokenContent(id="special_end", type_id=1)),
        ),
        special_tokens=SpecialTokens(
            {
                "special_begin": TokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
                "special_end": TokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
                "sequence": TokenInfo(id="sequence", ids=[2], tokens=["[SEQ]"]),
            }
        ),
    )


def _get_no_eos_template() -> TemplatePostProcessor:
    """Gets a template processor without eos."""
    return TemplatePostProcessor(
        single=(
            SpecialToken(SpecialToken=TokenContent(id="special_begin", type_id=0)),
            SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
        ),
        pair=(
            SpecialToken(SpecialToken=TokenContent(id="special_begin", type_id=0)),
            SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
        ),
        special_tokens=SpecialTokens(
            {
                "special_begin": TokenInfo(id="special_begin", ids=[0], tokens=["[BEGIN]"]),
                "sequence": TokenInfo(id="sequence", ids=[2], tokens=["[SEQ]"]),
            }
        ),
    )


def _get_no_bos_template() -> TemplatePostProcessor:
    """Gets a template processor without bos."""
    return TemplatePostProcessor(
        single=(
            SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
            SpecialToken(SpecialToken=TokenContent(id="special_end", type_id=1)),
        ),
        pair=(
            SequenceToken(Sequence=TokenContent(id="sequence", type_id=2)),
            SpecialToken(SpecialToken=TokenContent(id="special_end", type_id=1)),
        ),
        special_tokens=SpecialTokens(
            {
                "special_end": TokenInfo(id="special_end", ids=[1], tokens=["[END]"]),
                "sequence": TokenInfo(id="sequence", ids=[2], tokens=["[SEQ]"]),
            }
        ),
    )


@pytest.mark.parametrize(
    "post_processor,result",
    [
        (_get_default_postprocessor(PostProcessorType.SEQUENCE), None),
        (_get_default_postprocessor(PostProcessorType.BYTE_LEVEL), None),
        (_get_default_postprocessor(PostProcessorType.BERT_PROCESSING), ["[CLS]"]),
        (_get_default_postprocessor(PostProcessorType.ROBERTA_PROCESSING), ["[CLS]"]),
        (_get_default_postprocessor(PostProcessorType.TEMPLATE_PROCESSING), ["[BEGIN]"]),
        (_get_no_single_template(), None),
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
        (_get_no_single_template(), None),
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
    if isinstance(result, (RobertaPostProcessor, BertPostProcessor)) and old_token == "[CLS]":
        assert result.cls == (new_token, 11)
        assert result.sep == ("[SEP]", 1)
    if isinstance(result, (RobertaPostProcessor, BertPostProcessor)) and old_token == "[SEP]":
        assert result.cls == ("[CLS]", 0)
        assert result.sep == (new_token, 11)
    if isinstance(result, TemplatePostProcessor):
        assert result.special_tokens["special_end"].tokens == [new_token]

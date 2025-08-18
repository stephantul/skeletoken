from typing import Any

import pytest
from tokenizers import Tokenizer

from skeletoken.base import TokenizerModel
from skeletoken.postprocessors import (
    BertPostProcessor,
    ByteLevelPostProcessor,
    PostProcessor,
    PostProcessorType,
    RobertaPostProcessor,
    SequenceToken,
    SpecialToken,
    SpecialTokens,
    TemplatePostProcessor,
    TokenContent,
    TokenInfo,
)


def _get_default_postprocessor(normalizer_type: PostProcessorType) -> PostProcessor:  # noqa: C901
    """Helper function to get the default instantiation of a normalizer."""
    if normalizer_type == PostProcessorType.BYTE_LEVEL:
        return ByteLevelPostProcessor(trim_offsets=True, add_prefix_space=False, use_regex=False)
    elif normalizer_type == PostProcessorType.BERT_PROCESSING:
        return BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0))
    elif normalizer_type == PostProcessorType.ROBERTA_PROCESSING:
        return RobertaPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False)
    elif normalizer_type == PostProcessorType.TEMPLATE_PROCESSING:
        return TemplatePostProcessor(
            single=(
                SpecialToken(SpecialToken=TokenContent(id="single_special", type_id=0)),
                SequenceToken(Sequence=TokenContent(id="sequence_special", type_id=1)),
                SpecialToken(SpecialToken=TokenContent(id="single_special", type_id=0)),
            ),
            pair=(
                SpecialToken(SpecialToken=TokenContent(id="pair_special_1", type_id=0)),
                SequenceToken(Sequence=TokenContent(id="pair_sequence", type_id=1)),
                SpecialToken(SpecialToken=TokenContent(id="pair_special_2", type_id=2)),
                SequenceToken(Sequence=TokenContent(id="pair_sequence_2", type_id=3)),
                SpecialToken(SpecialToken=TokenContent(id="pair_special_3", type_id=4)),
            ),
            special_tokens=SpecialTokens(
                {
                    "single_special": TokenInfo(id="single_special", ids=[0], tokens=["[SINGLE]"]),
                    "pair_special_1": TokenInfo(id="pair_special_1", ids=[1], tokens=["[PAIR1]"]),
                    "pair_sequence": TokenInfo(id="pair_sequence", ids=[2], tokens=["[PAIR_SEQ]"]),
                    "pair_special_2": TokenInfo(id="pair_special_2", ids=[3], tokens=["[PAIR2]"]),
                    "pair_sequence_2": TokenInfo(id="pair_sequence_2", ids=[4], tokens=["[PAIR_SEQ2]"]),
                    "pair_special_3": TokenInfo(id="pair_special_3", ids=[5], tokens=["[PAIR3]"]),
                }
            ),
        )
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}")


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

    Tokenizer.from_str(tokenizer.model_dump_json())

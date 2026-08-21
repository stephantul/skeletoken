from typing import Any

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.common import PrependScheme, StringPattern
from skeletoken.decoders import (
    BPEDecoder,
    ByteFallbackDecoder,
    ByteLevelDecoder,
    CTCDecoder,
    Decoder,
    DecoderSequence,
    DecoderType,
    FuseDecoder,
    MetaspaceDecoder,
    ReplaceDecoder,
    StripDecoder,
    WordPieceDecoder,
    add_clean_up_tokenization_spaces,
    strip_clean_up_tokenization_spaces,
)
from tests.conftest import call_tokenizer


def _get_default_decoder(decoder_type: DecoderType) -> Decoder:
    """Helper function to get the default instantiation of a decoder."""
    if decoder_type == DecoderType.BPEDECODER:
        return BPEDecoder(suffix=r"\w")
    elif decoder_type == DecoderType.BYTEFALLBACK:
        return ByteFallbackDecoder()
    elif decoder_type == DecoderType.BYTELEVEL:
        return ByteLevelDecoder(add_prefix_space=False, trim_offsets=False, use_regex=False)
    elif decoder_type == DecoderType.CTC:
        return CTCDecoder(pad_token="[PAD]", word_delimiter_token="", cleanup=False)
    elif decoder_type == DecoderType.FUSE:
        return FuseDecoder()
    elif decoder_type == DecoderType.METASPACE:
        return MetaspaceDecoder(replacement=" ", prepend_scheme=PrependScheme.FIRST, split=False)
    elif decoder_type == DecoderType.REPLACE:
        return ReplaceDecoder(pattern=StringPattern(String="a"), content="replacement")
    elif decoder_type == DecoderType.STRIP:
        return StripDecoder(content=" ", start=0, stop=1)
    elif decoder_type == DecoderType.WORDPIECE:
        return WordPieceDecoder(prefix="##", cleanup=True)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


@pytest.mark.parametrize(
    "decoder_type",
    [
        DecoderType.BPEDECODER,
        DecoderType.BYTEFALLBACK,
        DecoderType.BYTELEVEL,
        DecoderType.CTC,
        DecoderType.FUSE,
        DecoderType.METASPACE,
        DecoderType.REPLACE,
        DecoderType.STRIP,
        DecoderType.WORDPIECE,
    ],
)
def test_decoder(small_tokenizer_json: dict[str, Any], decoder_type: DecoderType) -> None:
    """Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    decoder = _get_default_decoder(decoder_type)
    decoder_dict = decoder.model_dump()
    small_tokenizer_json["decoder"] = decoder_dict
    model = TokenizerModel.model_validate(small_tokenizer_json)

    assert model.decoder is not None
    assert model.decoder.type == decoder_type

    call_tokenizer(model)


def test_add_clean_up_tokenization_spaces_none() -> None:
    """Adding to no decoder produces a bare Fuse + Replace sequence."""
    result = add_clean_up_tokenization_spaces(None)
    assert isinstance(result, DecoderSequence)
    assert result.decoders[0] == FuseDecoder()
    assert len(result.decoders) == 11  # Fuse + 10 Replace steps
    assert all(isinstance(step, (FuseDecoder, ReplaceDecoder)) for step in result.decoders)


def test_add_clean_up_tokenization_spaces_single_decoder() -> None:
    """Adding to a single, non-Sequence decoder wraps it in a new Sequence with the original first."""
    original = WordPieceDecoder(prefix="##", cleanup=True)
    result = add_clean_up_tokenization_spaces(original)
    assert isinstance(result, DecoderSequence)
    assert result.decoders[0] == original
    assert len(result.decoders) == 12  # original + Fuse + 10 Replace steps


def test_add_clean_up_tokenization_spaces_existing_sequence() -> None:
    """Adding to an existing Sequence appends the steps rather than nesting a new Sequence."""
    original = DecoderSequence(decoders=[WordPieceDecoder(prefix="##", cleanup=True), FuseDecoder()])
    result = add_clean_up_tokenization_spaces(original)
    assert isinstance(result, DecoderSequence)
    assert result.decoders[:2] == original.decoders
    assert len(result.decoders) == 13  # 2 original + Fuse + 10 Replace steps


def test_strip_clean_up_tokenization_spaces_not_sequence() -> None:
    """A single, non-Sequence decoder is returned unchanged."""
    original = WordPieceDecoder(prefix="##", cleanup=True)
    assert strip_clean_up_tokenization_spaces(original) == original


def test_strip_clean_up_tokenization_spaces_sequence_without_suffix() -> None:
    """A Sequence that doesn't end in the exact suffix is returned unchanged."""
    original = DecoderSequence(decoders=[WordPieceDecoder(prefix="##", cleanup=True), FuseDecoder()])
    assert strip_clean_up_tokenization_spaces(original) == original


def test_strip_clean_up_tokenization_spaces_exact_suffix() -> None:
    """Stripping a decoder made of nothing but the suffix leaves no decoder behind."""
    added = add_clean_up_tokenization_spaces(None)
    assert strip_clean_up_tokenization_spaces(added) is None


def test_strip_clean_up_tokenization_spaces_single_remaining() -> None:
    """Stripping a wrapped single decoder unwraps it back to that decoder alone."""
    original = WordPieceDecoder(prefix="##", cleanup=True)
    added = add_clean_up_tokenization_spaces(original)
    assert strip_clean_up_tokenization_spaces(added) == original


def test_strip_clean_up_tokenization_spaces_multiple_remaining() -> None:
    """Stripping a Sequence with more than one leading step leaves the remainder as a Sequence."""
    original = DecoderSequence(decoders=[WordPieceDecoder(prefix="##", cleanup=True), FuseDecoder()])
    added = add_clean_up_tokenization_spaces(original)
    assert strip_clean_up_tokenization_spaces(added) == original

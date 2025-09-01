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
    DecoderType,
    FuseDecoder,
    MetaspaceDecoder,
    ReplaceDecoder,
    StripDecoder,
    WordPieceDecoder,
)


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
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    decoder = _get_default_decoder(decoder_type)
    decoder_dict = decoder.model_dump()
    small_tokenizer_json["decoder"] = decoder_dict
    model = TokenizerModel.model_validate(small_tokenizer_json)

    assert model.decoder is not None
    assert model.decoder.type == decoder_type

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()

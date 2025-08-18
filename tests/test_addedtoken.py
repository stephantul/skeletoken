from typing import Any

from tokenizers import Tokenizer

from skeletoken.addedtoken import AddedToken
from skeletoken.base import TokenizerModel


def test_addedtokens(small_tokenizer_json: dict[str, Any]) -> None:
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    token_a = AddedToken(
        content="a", single_word=True, lstrip=False, rstrip=False, normalized=False, special=False, id=11
    )
    token_b = AddedToken(
        content="b", single_word=True, lstrip=False, rstrip=False, normalized=False, special=False, id=10
    )
    toks = [token_a, token_b]
    for token in toks:
        small_tokenizer_json["added_tokens"].append(token.model_dump())

    tokenizer = TokenizerModel.model_validate(small_tokenizer_json)
    assert tokenizer.version == "1.0"
    assert len(tokenizer.added_tokens) == 2
    for a, b in zip(tokenizer.added_tokens, toks, strict=True):
        assert a.content == b.content
        assert a.single_word == b.single_word
        assert a.rstrip == b.rstrip
        assert a.lstrip == b.lstrip
        assert a.normalized == b.normalized
        assert a.special == b.special
        assert a.id == b.id

    # Implicit test. If this fails, the model is incorrect.
    Tokenizer.from_str(tokenizer.model_dump_json())

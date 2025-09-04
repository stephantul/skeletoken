from typing import Any

from skeletoken.addedtoken import AddedToken, AddedTokens
from skeletoken.base import TokenizerModel
from skeletoken.models import WordPiece


def test_addedtoken(small_tokenizer_json: dict[str, Any]) -> None:
    """
    Test that the small tokenizer JSON can be loaded and contains the expected structure.

    This test checks that the tokenizer JSON has the correct keys and types for its fields.
    """
    small_tokenizer_json["added_tokens"] = []
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
    assert isinstance(tokenizer.model, WordPiece)
    tokenizer.model.continuing_subword_prefix = ""
    assert len(tokenizer.added_tokens) == 3
    for a, b in zip(tokenizer.added_tokens.root[:-1], toks, strict=True):
        assert a.content == b.content
        assert a.single_word == b.single_word
        assert a.rstrip == b.rstrip
        assert a.lstrip == b.lstrip
        assert a.normalized == b.normalized
        assert a.special == b.special
        assert a.id == tokenizer.model.vocab[b.content]

    # Implicit test. If this fails, the model is incorrect.
    tok = tokenizer.to_tokenizer()

    assert tok.encode("a b c").tokens == ["a", " ", "b", " ", "c"]


def test_addedtokens_object() -> None:
    """Test the AddedTokens object."""
    tokens = AddedTokens([])
    tokens.maybe_add_token("a", 11)
    assert len(tokens) == 1
    assert tokens.root[0].id == 11
    tokens.maybe_add_token("a", 12)
    assert len(tokens) == 1
    assert tokens.root[0].id == 12

    tokens.maybe_remove_token("a")
    assert len(tokens) == 0
    assert tokens.root == []
    # Should not raise an error
    tokens.maybe_remove_token("a")
    tokens.maybe_replace_token("a", "b")

    tokens.maybe_add_token("a", 10)
    tokens.maybe_replace_token("a", "b")
    assert len(tokens) == 1
    assert tokens.root[0].content == "b"

    tokens.maybe_add_token("b", 10, normalized=True)
    assert len(tokens) == 1
    assert tokens.root[0].id == 10
    assert tokens.root[0].normalized is True

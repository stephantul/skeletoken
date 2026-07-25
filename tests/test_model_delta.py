from typing import Any

import pytest

from skeletoken.base import TokenizerModel
from skeletoken.model_delta import compute_model_delta


@pytest.mark.parametrize("param", ["wordpiece"])
def test_compute_model_delta_with_id_remapping(small_tokenizer_json: dict[str, Any], param: str) -> None:
    """When the modified tokenizer has an _id_remapping set we should use it to map ids back to the original."""
    # Build original and modified TokenizerModel from the same base JSON
    orig = TokenizerModel.model_validate(small_tokenizer_json)

    # Make a deep copy and simulate a remapping: swap ids 5 and 6 (tokens 'a' and 'b')
    mod = TokenizerModel.model_validate(small_tokenizer_json)
    sorted_vocab = mod.sorted_vocabulary
    # Swapping.
    mod.vocabulary[sorted_vocab[5]] = 6
    mod.vocabulary[sorted_vocab[6]] = 5

    new_sorted_vocab = mod.sorted_vocabulary
    assert sorted_vocab[5] == new_sorted_vocab[6]
    assert sorted_vocab[6] == new_sorted_vocab[5]
    assert sorted_vocab[:5] == new_sorted_vocab[:5]
    assert sorted_vocab[7:] == new_sorted_vocab[7:]

    delta = compute_model_delta(orig, mod)

    # Expect that token_mapping contains entries for the indices that were remapped
    # Because inv_id_remapping is inverted, indices 5 and 6 in the new vocab should map back
    assert 5 in delta.token_mapping
    assert 6 in delta.token_mapping
    # And they should point to the original indices
    assert delta.token_mapping[5] == 6
    assert delta.token_mapping[6] == 5


def test_compute_model_delta_with_moved_token(small_tokenizer_json: dict[str, Any]) -> None:
    """If a token exists in the old vocabulary but at a different index, it should be mapped using old_vocab lookup."""
    orig = TokenizerModel.model_validate(small_tokenizer_json)

    # Create modified but replace vocabulary so that token 'a' moves to a different index
    mod_json = dict(small_tokenizer_json)
    # Create a new vocab mapping where 'a' is at the end. IDs must stay contiguous, so the
    # remaining tokens are shifted down to fill the gap left by 'a'.
    old_vocab = dict(mod_json["model"]["vocab"])
    remaining = sorted((token for token in old_vocab if token != "a"), key=lambda t: old_vocab[t])
    new_vocab = {token: idx for idx, token in enumerate(remaining)}
    new_vocab["a"] = len(new_vocab)
    mod_json = dict(mod_json)
    mod_json["model"] = dict(mod_json["model"])  # shallow copy
    mod_json["model"]["vocab"] = new_vocab

    mod = TokenizerModel.model_validate(mod_json)

    delta = compute_model_delta(orig, mod)

    # The delta should map 'a's new (real) ID back to its original ID.
    new_id = mod.vocabulary["a"]
    assert new_id in delta.token_mapping
    assert delta.token_mapping[new_id] == orig.vocabulary["a"]


def test_compute_model_delta_detects_new_tokens(small_tokenizer_json: dict[str, Any]) -> None:
    """New tokens that didn't exist in the original vocabulary should be listed in new_tokens."""
    orig = TokenizerModel.model_validate(small_tokenizer_json)

    mod_json = dict(small_tokenizer_json)
    mod_json = dict(mod_json)
    mod_json["model"] = dict(mod_json["model"])  # shallow copy
    # Add a new token 'NEW_TOKEN' with a fresh index
    new_vocab = dict(mod_json["model"]["vocab"]).copy()
    new_vocab["NEW_TOKEN"] = max(new_vocab.values()) + 1
    mod_json["model"]["vocab"] = new_vocab

    mod = TokenizerModel.model_validate(mod_json)

    delta = compute_model_delta(orig, mod)

    assert "NEW_TOKEN" in delta.new_tokens
    assert delta.new_tokens["NEW_TOKEN"] == mod.vocabulary["NEW_TOKEN"]

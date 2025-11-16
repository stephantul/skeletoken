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
    # Simulate that during some transformation token at new index 5 was originally at 6 and vice versa
    # _id_remapping maps old->new; compute_model_delta inverts it, so set values accordingly
    mod._id_remapping = {6: 5, 5: 6}

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
    # Create a new vocab mapping where 'a' is at the end
    new_vocab = dict(mod_json["model"]["vocab"]).copy()
    new_vocab.pop("a")
    max_index = max(new_vocab.values())
    new_vocab["a"] = max_index + 1
    mod_json = dict(mod_json)
    mod_json["model"] = dict(mod_json["model"])  # shallow copy
    mod_json["model"]["vocab"] = new_vocab

    mod = TokenizerModel.model_validate(mod_json)

    delta = compute_model_delta(orig, mod)

    # Find the new index for 'a' in the modified sorted vocabulary
    new_sorted = mod.sorted_vocabulary
    new_index = new_sorted.index("a")
    # The delta should map the new index back to the original index
    assert new_index in delta.token_mapping
    assert delta.token_mapping[new_index] == orig.vocabulary["a"]


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

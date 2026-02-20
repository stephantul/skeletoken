from pydantic import BaseModel

from skeletoken.base import TokenizerModel


class ModelDelta(BaseModel):
    """Represents the differences between two tokenizer models.

    This is used to remap indices in the embedding table when loading.

    Attributes
    ----------
    token_mapping : dict[int, int]
        A mapping from old token IDs to new token IDs.
    new_tokens : dict[str, int]
        A mapping from new token strings to their IDs. This is useful for inferring contents of the new tokens.
    new_vocabulary_size : int
        The size of the new vocabulary in the modified model.

    """

    # Used to remap from old token ids to new token ids.
    token_mapping: dict[int, int]
    # Used to identify which tokens go to which new ids.
    new_tokens: dict[str, int]
    # The new vocabulary size. This is used to resize the embedding table.
    new_vocabulary_size: int


def compute_model_delta(original: TokenizerModel, modified: TokenizerModel) -> ModelDelta:
    """Compute the delta between two tokenizer models."""
    # Compute the mapping
    token_mapping = {}
    new_tokens = {}
    new_vocab = modified.sorted_vocabulary
    old_vocab = original.vocabulary
    for i, token in enumerate(new_vocab):
        if token in old_vocab:
            # This is an old token that got a new index
            old_index = old_vocab[token]
            token_mapping[i] = old_index
        else:
            # This is a new token
            new_tokens[token] = i
    # Compute new tokens
    new_vocabulary_size = len(modified.vocabulary)

    return ModelDelta(token_mapping=token_mapping, new_tokens=new_tokens, new_vocabulary_size=new_vocabulary_size)

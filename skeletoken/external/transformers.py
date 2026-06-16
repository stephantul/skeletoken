from typing import TypeVar

import torch
from transformers import PreTrainedModel

from skeletoken import TokenizerModel

T = TypeVar("T", bound=PreTrainedModel)


def _remap_embeddings(embeddings: torch.Tensor, shift_mapping: dict[int, int]) -> torch.Tensor:
    """Remap the embeddings according to the provided shift mapping.

    Parameters
    ----------
    embeddings : torch.Tensor
        The original embeddings to be remapped.
    shift_mapping : dict[int, int]
        A mapping from new indices to old indices.

    Returns
    -------
    torch.Tensor
        The remapped embeddings.

    """
    embeddings = embeddings.clone()
    if not shift_mapping:
        return embeddings

    to_map, from_map = zip(*shift_mapping.items(), strict=True)
    to_map_tensor = torch.tensor(to_map, dtype=torch.long, device=embeddings.device)
    from_map_tensor = torch.tensor(from_map, dtype=torch.long, device=embeddings.device)
    embeddings[to_map_tensor] = embeddings[from_map_tensor]

    return embeddings


def reshape_embeddings(model: T, tokenizer_model: TokenizerModel) -> T:
    """Reshape the embeddings of a given model to match the vocabulary size of a tokenizer model.

    Parameters
    ----------
    model : T
        The model whose embeddings are to be reshaped.
    tokenizer_model : TokenizerModel
        The tokenizer model whose vocabulary will be used to update the embeddings.

    Returns
    -------
    T
        The model with an updated embedding and vocabulary.

    """
    vocab_size = tokenizer_model.vocabulary_size
    delta = tokenizer_model.model_delta
    mapping = delta.token_mapping
    embedding = model.get_input_embeddings()
    assert isinstance(embedding, torch.nn.Embedding)
    original_vocab_size = embedding.weight.shape[0]
    weight = _remap_embeddings(embedding.weight, mapping)
    embedding.weight.data = weight
    model.resize_token_embeddings(vocab_size)

    # token_mapping is new→old; invert to old→new for updating config IDs.
    inv_mapping = {old: new for new, old in mapping.items()}
    for key in vars(model.config):
        if key == "vocab_size":
            setattr(model.config, key, vocab_size)
        elif key.endswith("_id"):
            current_id = getattr(model.config, key)
            if isinstance(current_id, int):
                if current_id in inv_mapping:
                    setattr(model.config, key, inv_mapping[current_id])
                elif 0 <= current_id < original_vocab_size:
                    setattr(model.config, key, None)

    return model

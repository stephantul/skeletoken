import copy
from typing import TypeVar

import torch
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel

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


def _remap_config_token_ids(
    config: PretrainedConfig | GenerationConfig, inv_mapping: dict[int, int], original_vocab_size: int
) -> None:
    """Rewrite the token IDs stored on a config, and on its sub-configs, in place.

    Parameters
    ----------
    config : PretrainedConfig | GenerationConfig
        The config to update. Every attribute whose name ends in `_id` is treated as a
        token ID, which is how `transformers` names them.
    inv_mapping : dict[int, int]
        A mapping from old token IDs to new token IDs.
    original_vocab_size : int
        The vocabulary size before the reshape, used to tell an ID that belonged to a
        removed token from a value that never was a token ID at all.

    """
    for key in list(vars(config)):
        if not key.endswith("_id"):
            continue
        current_id = getattr(config, key)
        if isinstance(current_id, int):
            if current_id in inv_mapping:
                setattr(config, key, inv_mapping[current_id])
            elif 0 <= current_id < original_vocab_size:
                setattr(config, key, None)
        elif isinstance(current_id, list) and all(isinstance(token_id, int) for token_id in current_id):
            # A generation config routinely holds several end-of-sequence IDs.
            surviving_ids = [inv_mapping[token_id] for token_id in current_id if token_id in inv_mapping]
            setattr(config, key, surviving_ids or None)

    # A multimodal checkpoint keeps its text IDs on a sub-config, with nothing shadowing
    # them at the top level.
    for name in getattr(type(config), "sub_configs", {}):
        sub_config = getattr(config, name, None)
        if sub_config is not None:
            _remap_config_token_ids(sub_config, inv_mapping, original_vocab_size)


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
        A new model, with an updated embedding and vocabulary. The input model is left untouched.

    """
    model = copy.deepcopy(model)
    vocab_size = tokenizer_model.vocabulary_size
    delta = tokenizer_model.model_delta
    mapping = delta.token_mapping
    embedding = model.get_input_embeddings()
    assert isinstance(embedding, torch.nn.Embedding)
    original_vocab_size = embedding.weight.shape[0]
    weight = _remap_embeddings(embedding.weight, mapping)
    embedding.weight.data = weight

    head = model.get_output_embeddings()
    if head is not None:
        if not getattr(model.config, "tie_word_embeddings", False):
            head.weight.data = _remap_embeddings(head.weight, mapping)
        if getattr(head, "bias", None) is not None:
            head.bias.data = _remap_embeddings(head.bias, mapping)

    padding_idx = embedding.padding_idx

    model.resize_token_embeddings(vocab_size)

    # token_mapping is new→old; invert to old→new for updating config IDs.
    inv_mapping = {old: new for new, old in mapping.items()}

    if padding_idx is not None:
        # A resize carries `padding_idx` over unchanged, so it now names whichever token
        # landed at that index.
        resized_embedding = model.get_input_embeddings()
        assert isinstance(resized_embedding, torch.nn.Embedding)
        resized_embedding.padding_idx = inv_mapping.get(padding_idx)

    # Set separately from the walk below: a sub-config has a `vocab_size` of its own that
    # must not be overwritten with the size of the text vocabulary.
    model.config.get_text_config().vocab_size = vocab_size
    _remap_config_token_ids(model.config, inv_mapping, original_vocab_size)

    # `generate()` reads its token IDs from the generation config in preference to the
    # config; an encoder-only model does not have one.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        _remap_config_token_ids(generation_config, inv_mapping, original_vocab_size)

    return model

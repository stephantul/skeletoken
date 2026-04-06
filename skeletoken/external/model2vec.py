import numpy as np
from model2vec import StaticModel

from skeletoken import TokenizerModel


def _remap_embeddings(embeddings: np.ndarray, shift_mapping: dict[int, int]) -> np.ndarray:
    """Remap the embeddings according to the provided shift mapping.

    Parameters
    ----------
    embeddings : np.ndarray
        The original embeddings to be remapped.
    shift_mapping : dict[int, int]
        A mapping from new indices to old indices.

    Returns
    -------
    np.ndarray
        The remapped embeddings.

    """
    embeddings = embeddings.copy()
    if not shift_mapping:
        return embeddings

    to_map, from_map = zip(*shift_mapping.items(), strict=True)
    to_map_array = np.asarray(to_map, dtype=int)
    from_map_array = np.asarray(from_map, dtype=int)
    embeddings[to_map_array] = embeddings[from_map_array]

    return embeddings


def reshape_embeddings(model: StaticModel, tokenizer_model: TokenizerModel) -> StaticModel:
    """Reshape the embeddings of a given model2vec model to match the vocabulary size of a tokenizer model.

    Parameters
    ----------
    model : StaticModel
        The model whose embeddings are to be reshaped.
    tokenizer_model : TokenizerModel
        The tokenizer model whose vocabulary will be used to update the embeddings.

    Returns
    -------
    StaticModel
        The model with an updated embedding and vocabulary.

    Raises
    ------
    ValueError
        If the model is quantized.

    """
    if model.weights is not None:
        if model.weights != len(model.embedding):
            raise ValueError("Model weights must match the number of embeddings. This means your model is quantized.")
    vocab_size = tokenizer_model.vocabulary_size
    delta = tokenizer_model.model_delta
    mapping = delta.token_mapping

    embeddings = model.embedding

    rand_gen = np.random.default_rng()
    new_embeddings = rand_gen.normal(size=(vocab_size, embeddings.shape[1]))
    remapped = _remap_embeddings(embeddings, mapping)
    new_embeddings[: len(remapped)] = remapped[:vocab_size]

    return StaticModel(
        vectors=new_embeddings,
        tokenizer=tokenizer_model.to_tokenizer(),
        config=model.config,
        normalize=model.normalize,
        base_model_name=model.base_model_name,
        language=model.language,
        weights=model.weights,
        token_mapping=model.token_mapping,
    )

from typing import TypeVar, cast

from pylate.models import ColBERT
from transformers import PreTrainedModel

from skeletoken import TokenizerModel
from skeletoken.external.transformers import reshape_embeddings as _reshape_embeddings_transformers

T = TypeVar("T", bound=ColBERT)


def reshape_embeddings(model: T, tokenizer_model: TokenizerModel) -> T:
    """
    Reshape the embeddings of a given ColBERT model to match the vocabulary size of a tokenizer model.

    Parameters
    ----------
    model : T
        The model whose embeddings are to be reshaped.
    tokenizer_model : TokenizerModel
        The tokenizer model whose vocabulary size will be used.

    Returns
    -------
    T
        The model with an updated embedding and vocabulary.

    """
    auto_model = cast(PreTrainedModel, model[0].auto_model)
    auto_model = _reshape_embeddings_transformers(auto_model, tokenizer_model)
    model[0].auto_model = auto_model

    if model.query_prefix is not None:
        model.query_prefix_id = tokenizer_model.tokens_to_ids([model.query_prefix])[0]
    if model.document_prefix is not None:
        model.document_prefix_id = tokenizer_model.tokens_to_ids([model.document_prefix])[0]

    current_tokenizer = model.tokenizer
    new_tokenizer = tokenizer_model.to_transformers()
    model.tokenizer = new_tokenizer
    model.tokenizer.model_max_length = current_tokenizer.model_max_length
    model.tokenizer.model_input_names = current_tokenizer.model_input_names

    if model.skiplist_words is not None:
        model.skiplist = [new_tokenizer.convert_tokens_to_ids(word) for word in model.skiplist_words]

    return model

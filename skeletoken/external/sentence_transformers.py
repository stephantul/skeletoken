from typing import TypeVar, cast

from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel

from skeletoken import TokenizerModel
from skeletoken.external.transformers import reshape_embeddings as _reshape_embeddings_transformers

T = TypeVar("T", bound=SentenceTransformer)


def reshape_embeddings(model: T, tokenizer_model: TokenizerModel) -> T:
    """
    Reshape the embeddings of a given SentenceTransformer model to match the vocabulary size of a tokenizer model.

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
    auto_model = cast(PreTrainedModel, model[0].auto_model)
    auto_model = _reshape_embeddings_transformers(auto_model, tokenizer_model)
    model[0].auto_model = auto_model

    current_tokenizer = model.tokenizer
    new_tokenizer = tokenizer_model.to_transformers()
    model.tokenizer = new_tokenizer
    model.tokenizer.model_max_length = current_tokenizer.model_max_length
    model.tokenizer.model_input_names = current_tokenizer.model_input_names

    return model

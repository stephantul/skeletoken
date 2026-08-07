import copy
from typing import TypeVar, cast

from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel

from skeletoken import TokenizerModel
from skeletoken.external.transformers import reshape_embeddings as _reshape_embeddings_transformers

T = TypeVar("T", bound=SentenceTransformer)


def reshape_embeddings(model: T, tokenizer_model: TokenizerModel) -> T:
    """Reshape the embeddings of a given SentenceTransformer model to match the vocabulary size of a tokenizer model.

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
    transformer_module = model[0]
    auto_model = cast(PreTrainedModel, transformer_module.auto_model)
    auto_model = _reshape_embeddings_transformers(auto_model, tokenizer_model)
    # sentence-transformers >=5.4 made `auto_model` a read-only property aliasing `model`.
    if isinstance(getattr(type(transformer_module), "auto_model", None), property):
        transformer_module.model = auto_model
    else:
        transformer_module.auto_model = auto_model

    current_tokenizer = model.tokenizer
    new_tokenizer = tokenizer_model.to_transformers()
    # Ignore both types so mypy passes independently of which version is installed.
    try:
        model.tokenizer = new_tokenizer  # type: ignore
    except AttributeError:
        model[0].processor = new_tokenizer  # type: ignore
    model.tokenizer.model_max_length = current_tokenizer.model_max_length
    model.tokenizer.model_input_names = current_tokenizer.model_input_names

    return model

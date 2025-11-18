from typing import TypeVar, cast

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer
from tokenizers import Tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerFast

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

    Raises
    ------
    ValueError
        If the first module of the SentenceTransformer is not a Transformer.

    """
    sub_model = model._first_module()
    if not isinstance(sub_model, Transformer):
        raise ValueError("The first module of the SentenceTransformer is not a Transformer.")
    sub_model = cast(Transformer, sub_model)
    auto_model = cast(PreTrainedModel, sub_model.auto_model)
    new_auto_model = _reshape_embeddings_transformers(auto_model, tokenizer_model)
    current_tokenizer = sub_model.tokenizer
    if isinstance(current_tokenizer, Tokenizer):
        raise ValueError("The current tokenizer is a Tokenizer from the tokenizers library, which is not supported.")
    tokenizer_cls = current_tokenizer.__class__
    if not issubclass(tokenizer_cls, PreTrainedTokenizerFast):
        raise ValueError("The current tokenizer is not a subclass of PreTrainedTokenizerFast.")
    sub_model.auto_model = new_auto_model
    sub_model.tokenizer = tokenizer_model.to_transformers(tokenizer_class=tokenizer_cls)  # type: ignore
    sub_model.tokenizer.model_max_length = current_tokenizer.model_max_length
    sub_model.tokenizer.model_input_names = current_tokenizer.model_input_names

    return model

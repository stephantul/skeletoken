from typing import TYPE_CHECKING

from skeletoken.models import MODELS_THAT_NEED_UNK, ModelType, empty_model

if TYPE_CHECKING:
    from skeletoken.base import TokenizerModel  # pragma: no cover


def empty_tokenizer(model_type: ModelType = ModelType.BPE) -> "TokenizerModel":
    """Create an empty tokenizer model. Used to start from scratch."""
    from skeletoken.base import TokenizerModel

    model = empty_model(model_type)
    tokenizer = TokenizerModel(
        model=model,
        normalizer=None,
        pre_tokenizer=None,
        post_processor=None,
        decoder=None,
    )
    if isinstance(model, MODELS_THAT_NEED_UNK):
        tokenizer.added_tokens.upsert_token(model.unk_token, id=0, is_special=True)

    return tokenizer

import pytest

from skeletoken.empty import empty_tokenizer
from skeletoken.models import BPE, ModelType


def test_empty_tokenizer_default() -> None:
    """Test that the default empty tokenizer is an empty BPE model."""
    tokenizer = empty_tokenizer()
    assert isinstance(tokenizer.model, BPE)
    assert tokenizer.added_tokens.root == []
    assert tokenizer.normalizer is None
    assert tokenizer.pre_tokenizer is None
    assert tokenizer.post_processor is None
    assert tokenizer.decoder is None


@pytest.mark.parametrize("model_type", list(ModelType))
def test_empty_tokenizer_model_type(model_type: ModelType) -> None:
    """Test that the empty tokenizer uses the requested model type and converts to a tokenizer."""
    tokenizer = empty_tokenizer(model_type)
    assert tokenizer.model.type == model_type
    tokenizer.to_tokenizer()


@pytest.mark.parametrize("model_type", [ModelType.WORDPIECE, ModelType.WORDLEVEL])
def test_empty_tokenizer_unk_is_added_token(model_type: ModelType) -> None:
    """Test that the unk token is a special added token with id 0."""
    tokenizer = empty_tokenizer(model_type)
    (added_token,) = tokenizer.added_tokens.root
    assert added_token.content == "[UNK]"
    assert added_token.id == 0
    assert added_token.special
    assert tokenizer.to_tokenizer().encode("hello [UNK]").tokens == ["[UNK]", "[UNK]"]


@pytest.mark.parametrize("model_type", [ModelType.BPE, ModelType.UNIGRAM])
def test_empty_tokenizer_no_added_tokens(model_type: ModelType) -> None:
    """Test that models without a required unk token have no added tokens."""
    tokenizer = empty_tokenizer(model_type)
    assert tokenizer.added_tokens.root == []
    assert len(tokenizer.vocabulary) == 0

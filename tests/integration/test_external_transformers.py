import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertModel  # noqa: E402

from skeletoken import TokenizerModel  # noqa: E402
from skeletoken.external.transformers import reshape_embeddings  # noqa: E402

_TOKENIZER_PATH = "tests/data/bert-base-cased"


def _make_bert_model(tokenizer_model: TokenizerModel) -> BertModel:
    config = BertConfig(
        vocab_size=tokenizer_model.vocabulary_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        max_position_embeddings=32,
        pad_token_id=tokenizer_model.pad_token_id or 0,
    )
    torch.manual_seed(0)
    return BertModel(config)


def test_reshape_embeddings_does_not_mutate_original() -> None:
    """Test that reshape_embeddings returns a new model, leaving the input untouched."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_model(tokenizer_model)
    original_weight = model.get_input_embeddings().weight.clone()
    original_vocab_size = model.config.vocab_size

    decased = tokenizer_model.decase_vocabulary()
    reshaped = reshape_embeddings(model, decased)

    assert reshaped is not model
    # The original model's embedding matrix and config must be untouched.
    assert torch.equal(model.get_input_embeddings().weight, original_weight)
    assert model.config.vocab_size == original_vocab_size

    assert reshaped.get_input_embeddings().weight.shape[0] == decased.vocabulary_size
    assert reshaped.config.vocab_size == decased.vocabulary_size


def test_reshape_embeddings_remaps_rows() -> None:
    """Test that surviving tokens keep their embedding row after decasing shrinks the vocabulary."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_model(tokenizer_model)
    embeddings_before = model.get_input_embeddings().weight.clone()

    decased = tokenizer_model.decase_vocabulary()
    reshaped = reshape_embeddings(model, decased)
    embeddings_after = reshaped.get_input_embeddings().weight

    delta = decased.model_delta
    assert len(delta.token_mapping) > 1000
    for new_id, old_id in delta.token_mapping.items():
        assert torch.allclose(embeddings_after[new_id], embeddings_before[old_id])

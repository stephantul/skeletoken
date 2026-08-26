import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import (  # noqa: E402
    BertConfig,
    BertForMaskedLM,
    BertLMHeadModel,
    BertModel,
    EncodecConfig,
    MusicgenConfig,
    MusicgenDecoderConfig,
    T5Config,
)

from skeletoken import TokenizerModel  # noqa: E402
from skeletoken.external.transformers import _remap_config_token_ids, reshape_embeddings  # noqa: E402

_TOKENIZER_PATH = "tests/data/bert-base-cased"
# An ordinary vocabulary entry at ID 1, so removing it shifts every special token down.
_REMOVED_TOKEN = "[unused1]"


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


def _make_bert_mlm_model(tokenizer_model: TokenizerModel, tie_word_embeddings: bool) -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=tokenizer_model.vocabulary_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        max_position_embeddings=32,
        pad_token_id=tokenizer_model.pad_token_id or 0,
        tie_word_embeddings=tie_word_embeddings,
    )
    torch.manual_seed(0)
    model = BertForMaskedLM(config)
    # A freshly initialised bias is all zeros, which would hide a wrongly selected row.
    with torch.no_grad():
        model.get_output_embeddings().bias.normal_()
    return model


def _make_bert_causal_lm_model(tokenizer_model: TokenizerModel) -> BertLMHeadModel:
    config = BertConfig(
        vocab_size=tokenizer_model.vocabulary_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        max_position_embeddings=32,
        pad_token_id=tokenizer_model.pad_token_id or 0,
        is_decoder=True,
    )
    torch.manual_seed(0)
    return BertLMHeadModel(config)


class _NestedBertConfig(BertConfig):
    sub_configs = {"text_config": BertConfig}


def _make_nested_bert_model(tokenizer_model: TokenizerModel, **text_config_kwargs: int) -> BertModel:
    config = _NestedBertConfig(
        vocab_size=tokenizer_model.vocabulary_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        max_position_embeddings=32,
        pad_token_id=tokenizer_model.pad_token_id or 0,
    )
    config.text_config = BertConfig(vocab_size=tokenizer_model.vocabulary_size, **text_config_kwargs)
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

    old_id_amsterdam = tokenizer_model.vocabulary["Amsterdam"]
    new_id_amsterdam = decased.vocabulary["amsterdam"]
    assert "amsterdam" not in delta.new_tokens
    assert delta.token_mapping[new_id_amsterdam] == old_id_amsterdam
    assert torch.allclose(embeddings_after[new_id_amsterdam], embeddings_before[old_id_amsterdam])


def test_reshape_embeddings_batch_added_tokens_grow_vocab_size() -> None:
    """Test that batch-adding tokens via add_tokens_to_vocabulary grows the embedding matrix by len(tokens)."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_model(tokenizer_model)
    original_vocab_size = model.config.vocab_size

    new_tokens = ["skeletokentesttoken", "anothernewtoken"]
    added = tokenizer_model.add_tokens_to_vocabulary(new_tokens)
    reshaped = reshape_embeddings(model, added)

    assert model.config.vocab_size == original_vocab_size
    assert reshaped.get_input_embeddings().weight.shape[0] == added.vocabulary_size
    assert reshaped.config.vocab_size == added.vocabulary_size == tokenizer_model.vocabulary_size + len(new_tokens)


def test_reshape_embeddings_batch_added_special_tokens_grow_vocab_size() -> None:
    """Test that batch-adding special tokens via add_addedtokens grows the embedding matrix by len(tokens)."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_model(tokenizer_model)
    original_vocab_size = model.config.vocab_size

    new_tokens = ["[PROTEIN]", "[DISEASE]", "[DRUG]"]
    added = tokenizer_model.add_addedtokens(new_tokens, is_special=True)
    reshaped = reshape_embeddings(model, added)

    assert model.config.vocab_size == original_vocab_size
    assert reshaped.get_input_embeddings().weight.shape[0] == added.vocabulary_size
    assert reshaped.config.vocab_size == added.vocabulary_size == tokenizer_model.vocabulary_size + len(new_tokens)


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_reshape_embeddings_remaps_output_head_bias(tie_word_embeddings: bool) -> None:
    """Test that the output head bias keeps the surviving rows, tied or untied."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_mlm_model(tokenizer_model, tie_word_embeddings=tie_word_embeddings)
    bias_before = model.get_output_embeddings().bias.clone()

    decased = tokenizer_model.decase_vocabulary()
    reshaped = reshape_embeddings(model, decased)
    bias_after = reshaped.get_output_embeddings().bias

    assert bias_after.shape[0] == decased.vocabulary_size
    for new_id, old_id in decased.model_delta.token_mapping.items():
        assert torch.allclose(bias_after[new_id], bias_before[old_id])


def test_reshape_embeddings_remaps_untied_output_head() -> None:
    """Test that an untied output head keeps the surviving rows instead of a prefix."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_mlm_model(tokenizer_model, tie_word_embeddings=False)
    head_before = model.get_output_embeddings().weight.clone()

    decased = tokenizer_model.decase_vocabulary()
    reshaped = reshape_embeddings(model, decased)
    head_after = reshaped.get_output_embeddings().weight

    assert head_after.shape[0] == decased.vocabulary_size
    for new_id, old_id in decased.model_delta.token_mapping.items():
        assert torch.allclose(head_after[new_id], head_before[old_id])


def test_reshape_embeddings_keeps_tied_output_head_tied() -> None:
    """Test that a tied output head is remapped once, and stays tied to the input embedding."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_mlm_model(tokenizer_model, tie_word_embeddings=True)
    embedding_module = model.get_input_embeddings()
    assert isinstance(embedding_module, torch.nn.Embedding)
    embeddings_before = embedding_module.weight.clone()

    decased = tokenizer_model.decase_vocabulary()
    reshaped = reshape_embeddings(model, decased)
    output_embedding_after = reshaped.get_output_embeddings()
    assert isinstance(output_embedding_after, torch.nn.Linear)
    head_after = output_embedding_after.weight

    input_embedding_after = reshaped.get_input_embeddings()
    assert isinstance(input_embedding_after, torch.nn.Embedding)
    assert head_after is input_embedding_after.weight
    for new_id, old_id in decased.model_delta.token_mapping.items():
        assert torch.allclose(head_after[new_id], embeddings_before[old_id])


def test_reshape_embeddings_remaps_generation_config_token_ids() -> None:
    """Test that the generation config's token IDs follow the vocabulary, as the config's do."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_causal_lm_model(tokenizer_model)
    vocabulary = tokenizer_model.vocabulary
    model.generation_config.bos_token_id = vocabulary["[CLS]"]
    model.generation_config.eos_token_id = vocabulary["[SEP]"]
    model.generation_config.pad_token_id = vocabulary["[PAD]"]

    trimmed = tokenizer_model.remove_tokens_from_vocabulary([_REMOVED_TOKEN])
    reshaped = reshape_embeddings(model, trimmed)

    new_vocabulary = trimmed.vocabulary
    # The removed token sits at ID 1, so every special above it shifts down by one and
    # the assertions below are not vacuous.
    assert new_vocabulary["[CLS]"] == vocabulary["[CLS]"] - 1
    assert reshaped.generation_config.bos_token_id == new_vocabulary["[CLS]"]
    assert reshaped.generation_config.eos_token_id == new_vocabulary["[SEP]"]
    assert reshaped.generation_config.pad_token_id == new_vocabulary["[PAD]"]
    # The input model keeps the generation config it came with.
    assert model.generation_config.eos_token_id == vocabulary["[SEP]"]


def test_reshape_embeddings_clears_removed_generation_config_token_ids() -> None:
    """Test that a generation config ID whose token was removed is cleared, not left dangling."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_causal_lm_model(tokenizer_model)
    model.generation_config.eos_token_id = tokenizer_model.vocabulary[_REMOVED_TOKEN]

    trimmed = tokenizer_model.remove_tokens_from_vocabulary([_REMOVED_TOKEN])
    reshaped = reshape_embeddings(model, trimmed)

    assert reshaped.generation_config.eos_token_id is None


def test_reshape_embeddings_remaps_generation_config_token_id_lists() -> None:
    """Test that a list of end-of-sequence IDs keeps the surviving tokens and drops the removed ones."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_causal_lm_model(tokenizer_model)
    vocabulary = tokenizer_model.vocabulary
    model.generation_config.eos_token_id = [vocabulary["[SEP]"], vocabulary[_REMOVED_TOKEN]]

    trimmed = tokenizer_model.remove_tokens_from_vocabulary([_REMOVED_TOKEN])
    reshaped = reshape_embeddings(model, trimmed)

    assert reshaped.generation_config.eos_token_id == [trimmed.vocabulary["[SEP]"]]


def test_reshape_embeddings_remaps_sub_config_token_ids() -> None:
    """Test that token IDs on a sub-config are remapped, since nothing shadows them at the top level."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    vocabulary = tokenizer_model.vocabulary
    model = _make_nested_bert_model(
        tokenizer_model,
        eos_token_id=vocabulary["[SEP]"],
        bos_token_id=vocabulary[_REMOVED_TOKEN],
    )

    trimmed = tokenizer_model.remove_tokens_from_vocabulary([_REMOVED_TOKEN])
    reshaped = reshape_embeddings(model, trimmed)

    assert reshaped.config.text_config.eos_token_id == trimmed.vocabulary["[SEP]"]
    assert reshaped.config.text_config.bos_token_id is None


def test_reshape_embeddings_remaps_padding_idx() -> None:
    """Test that the embedding's padding_idx follows its token instead of naming another one."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_model(tokenizer_model)
    # This tokenizer pads at ID 0, which no removal can move, so use a token that does.
    model.get_input_embeddings().padding_idx = tokenizer_model.vocabulary["[SEP]"]

    trimmed = tokenizer_model.remove_tokens_from_vocabulary([_REMOVED_TOKEN])
    reshaped = reshape_embeddings(model, trimmed)

    assert reshaped.get_input_embeddings().padding_idx == trimmed.vocabulary["[SEP]"]


def test_reshape_embeddings_clears_removed_padding_idx() -> None:
    """Test that a padding_idx whose token was removed is cleared rather than pointed at its successor."""
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    model = _make_bert_model(tokenizer_model)
    model.get_input_embeddings().padding_idx = tokenizer_model.vocabulary[_REMOVED_TOKEN]

    trimmed = tokenizer_model.remove_tokens_from_vocabulary([_REMOVED_TOKEN])
    reshaped = reshape_embeddings(model, trimmed)

    assert reshaped.get_input_embeddings().padding_idx is None


def test_remap_config_token_ids_ignores_unrelated_sub_config() -> None:
    """Test that a sub-config under a name `get_text_config` does not look for, like vision_config, is left alone."""

    class _CompositeConfig(BertConfig):
        sub_configs = {"text_config": BertConfig, "vision_config": BertConfig}

    config = _CompositeConfig(vocab_size=100)
    config.text_config = BertConfig(vocab_size=100, eos_token_id=5)
    config.vision_config = BertConfig(vocab_size=999, eos_token_id=5)

    inv_mapping = {token_id: token_id - 1 for token_id in range(1, 100)}

    _remap_config_token_ids(config, inv_mapping, original_vocab_size=100)

    assert config.text_config.eos_token_id == 4
    assert config.vision_config.eos_token_id == 5


def test_remap_config_token_ids_leaves_ambiguous_composite_config_unchanged() -> None:
    """Test that an ambiguous composite, like Musicgen's text encoder and decoder, is left untouched."""
    text_encoder = T5Config(vocab_size=32100, eos_token_id=1, pad_token_id=0)
    audio_encoder = EncodecConfig()
    decoder = MusicgenDecoderConfig(bos_token_id=2048, pad_token_id=2048)
    config = MusicgenConfig(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder)

    original_vocab_size = text_encoder.vocab_size
    # Shifts every ID down by one, so a token that survived the reshape would visibly move.
    inv_mapping = {token_id: token_id - 1 for token_id in range(1, original_vocab_size)}

    _remap_config_token_ids(config, inv_mapping, original_vocab_size)

    assert config.text_encoder.eos_token_id == 1
    assert config.text_encoder.pad_token_id == 0
    assert config.decoder.bos_token_id == 2048
    assert config.decoder.pad_token_id == 2048

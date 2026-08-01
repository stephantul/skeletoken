import pytest

pytest.importorskip("pylate")
torch = pytest.importorskip("torch")

from pylate.models import ColBERT  # noqa: E402

from skeletoken import TokenizerModel  # noqa: E402
from skeletoken.external.pylate import reshape_embeddings  # noqa: E402


def test_reshape_embeddings_remaps_and_swaps_tokenizer(small_bert_checkpoint_dir: str) -> None:
    """Test that reshape_embeddings remaps existing rows and installs the new tokenizer."""
    model = ColBERT(small_bert_checkpoint_dir)
    tokenizer_model = TokenizerModel.from_transformers_tokenizer(model.tokenizer)
    decased = tokenizer_model.decase_vocabulary()

    embeddings_before = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").clone()
    original_tokenizer = model.tokenizer
    original_query_prefix_id = model.query_prefix_id

    reshaped = reshape_embeddings(model, decased)

    # reshape_embeddings must not mutate its input.
    assert reshaped is not model
    assert torch.equal(model[0].get_parameter("auto_model.embeddings.word_embeddings.weight"), embeddings_before)
    assert model.tokenizer is original_tokenizer
    assert model.query_prefix_id == original_query_prefix_id

    embeddings_after = reshaped[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
    assert embeddings_after.shape[0] == decased.vocabulary_size

    delta = decased.model_delta
    assert len(delta.token_mapping) > 1000
    for new_id, old_id in delta.token_mapping.items():
        assert torch.allclose(embeddings_after[new_id], embeddings_before[old_id])

    # query/document prefix ids were remapped to the new vocabulary rather than left dangling.
    assert reshaped.query_prefix_id == decased.tokens_to_ids([reshaped.query_prefix])[0]
    assert reshaped.document_prefix_id == decased.tokens_to_ids([reshaped.document_prefix])[0]

    assert reshaped.tokenizer("amsterdam")["input_ids"] == reshaped.tokenizer("Amsterdam")["input_ids"]


def test_reshape_embeddings_batch_added_tokens_get_rows(small_bert_checkpoint_dir: str) -> None:
    """Test that batch-adding tokens via add_tokens_to_vocabulary each get a real embedding row."""
    model = ColBERT(small_bert_checkpoint_dir)
    original_vocab_size = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0]
    tokenizer_model = TokenizerModel.from_transformers_tokenizer(model.tokenizer)
    new_tokens = ["skeletokentesttoken", "anothernewtoken"]
    added = tokenizer_model.add_tokens_to_vocabulary(new_tokens)

    reshaped = reshape_embeddings(model, added)

    assert model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0] == original_vocab_size
    embeddings_after = reshaped[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
    assert embeddings_after.shape[0] == added.vocabulary_size == tokenizer_model.vocabulary_size + len(new_tokens)

    cls_id = added.vocabulary["[CLS]"]
    sep_id = added.vocabulary["[SEP]"]
    for token in new_tokens:
        new_id = added.vocabulary[token]
        assert reshaped.tokenizer(token)["input_ids"] == [cls_id, new_id, sep_id]


def test_reshape_embeddings_batch_added_special_tokens_get_rows(small_bert_checkpoint_dir: str) -> None:
    """Test that batch-adding special tokens via add_addedtokens each get a real embedding row."""
    model = ColBERT(small_bert_checkpoint_dir)
    original_vocab_size = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0]
    tokenizer_model = TokenizerModel.from_transformers_tokenizer(model.tokenizer)
    new_tokens = ["[PROTEIN]", "[DISEASE]", "[DRUG]"]
    added = tokenizer_model.add_addedtokens(new_tokens, is_special=True)

    reshaped = reshape_embeddings(model, added)

    assert model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0] == original_vocab_size
    embeddings_after = reshaped[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
    assert embeddings_after.shape[0] == added.vocabulary_size == tokenizer_model.vocabulary_size + len(new_tokens)

    cls_id = added.vocabulary["[CLS]"]
    sep_id = added.vocabulary["[SEP]"]
    for token in new_tokens:
        new_id = added.vocabulary[token]
        assert reshaped.tokenizer(token)["input_ids"] == [cls_id, new_id, sep_id]

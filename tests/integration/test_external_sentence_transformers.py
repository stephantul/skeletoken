import pytest

pytest.importorskip("sentence_transformers")
torch = pytest.importorskip("torch")

from sentence_transformers import SentenceTransformer  # noqa: E402

from skeletoken import TokenizerModel  # noqa: E402
from skeletoken.external.sentence_transformers import reshape_embeddings  # noqa: E402

_TOKENIZER_PATH = "tests/data/bert-base-cased"


def test_reshape_embeddings_remaps_and_swaps_tokenizer(small_bert_checkpoint_dir: str) -> None:
    """Test that reshape_embeddings remaps existing rows and installs the new tokenizer."""
    model = SentenceTransformer(small_bert_checkpoint_dir, local_files_only=True)
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    decased = tokenizer_model.decase_vocabulary()

    embeddings_before = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").clone()
    original_tokenizer = model.tokenizer
    original_tokenizer_class = type(original_tokenizer)

    reshaped = reshape_embeddings(model, decased)

    # reshape_embeddings must not mutate its input: same shape/values/tokenizer as before the call.
    assert reshaped is not model
    assert model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape == embeddings_before.shape
    assert torch.equal(model[0].get_parameter("auto_model.embeddings.word_embeddings.weight"), embeddings_before)
    assert model.tokenizer is original_tokenizer

    embeddings_after = reshaped[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
    assert embeddings_after.shape[0] == decased.vocabulary_size

    delta = decased.model_delta
    assert len(delta.token_mapping) > 1000
    for new_id, old_id in delta.token_mapping.items():
        assert torch.allclose(embeddings_after[new_id], embeddings_before[old_id])

    # The tokenizer actually got replaced, not just the embedding matrix.
    assert type(reshaped.tokenizer) is original_tokenizer_class
    assert reshaped.tokenizer("amsterdam")["input_ids"] == reshaped.tokenizer("Amsterdam")["input_ids"]

    # encode() must not crash after the swap.
    vector = reshaped.encode("Hello Amsterdam")
    assert vector.shape == (8,)


def test_reshape_embeddings_new_token_gets_a_row(small_bert_checkpoint_dir: str) -> None:
    """Test that a token added after loading gets a real (non-degenerate) embedding row."""
    model = SentenceTransformer(small_bert_checkpoint_dir, local_files_only=True)
    original_vocab_size = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0]
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    added = tokenizer_model.add_token_to_vocabulary("skeletokentesttoken")

    reshaped = reshape_embeddings(model, added)

    # The original model's vocabulary size must be untouched by the reshape.
    assert model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0] == original_vocab_size

    embeddings_after = reshaped[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
    assert embeddings_after.shape[0] == added.vocabulary_size == tokenizer_model.vocabulary_size + 1

    new_id = added.vocabulary["skeletokentesttoken"]
    cls_id = added.vocabulary["[CLS]"]
    sep_id = added.vocabulary["[SEP]"]
    assert reshaped.tokenizer("skeletokentesttoken")["input_ids"] == [cls_id, new_id, sep_id]


def test_reshape_embeddings_batch_added_tokens_get_rows(small_bert_checkpoint_dir: str) -> None:
    """Test that batch-adding tokens via add_tokens_to_vocabulary each get a real embedding row."""
    model = SentenceTransformer(small_bert_checkpoint_dir, local_files_only=True)
    original_vocab_size = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0]
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
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


def test_reshape_embeddings_new_added_token_gets_a_row(small_bert_checkpoint_dir: str) -> None:
    """Test that a special/added token added after loading gets a real (non-degenerate) row."""
    model = SentenceTransformer(small_bert_checkpoint_dir, local_files_only=True)
    original_vocab_size = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0]
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
    added = tokenizer_model.add_addedtoken("[SKELETOKEN]", is_special=True)

    reshaped = reshape_embeddings(model, added)

    assert model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0] == original_vocab_size
    embeddings_after = reshaped[0].get_parameter("auto_model.embeddings.word_embeddings.weight")
    assert embeddings_after.shape[0] == added.vocabulary_size == tokenizer_model.vocabulary_size + 1

    new_id = added.vocabulary["[SKELETOKEN]"]
    cls_id = added.vocabulary["[CLS]"]
    sep_id = added.vocabulary["[SEP]"]
    assert reshaped.tokenizer("[SKELETOKEN]")["input_ids"] == [cls_id, new_id, sep_id]


def test_reshape_embeddings_batch_added_special_tokens_get_rows(small_bert_checkpoint_dir: str) -> None:
    """Test that batch-adding special tokens via add_addedtokens each get a real embedding row."""
    model = SentenceTransformer(small_bert_checkpoint_dir, local_files_only=True)
    original_vocab_size = model[0].get_parameter("auto_model.embeddings.word_embeddings.weight").shape[0]
    tokenizer_model = TokenizerModel.from_pretrained(_TOKENIZER_PATH)
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

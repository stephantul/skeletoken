from skeletoken import TokenizerModel
from skeletoken.postprocessors import TemplatePostProcessor
from skeletoken.pretokenizers import DigitsPreTokenizer
from tests.conftest import assert_vocabulary_consistent, call_tokenizer

_PATH = "tests/data/ModernBERT-base"


def test_load() -> None:
    """Test loading ModernBERT-base."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert model.vocabulary_size == 50368
    assert model.pad_token == "[PAD]"
    assert model.pad_token_id == 50283
    assert model.unk_token == "[UNK]"
    assert model.unk_token_id == 50280
    assert model.subword_prefix is None
    assert model.adds_prefix_space is False
    assert model.transforms_into_bytes
    assert isinstance(model.post_processor, TemplatePostProcessor)
    assert model.vocabulary["[CLS]"] == 50281
    assert model.vocabulary["[SEP]"] == 50282
    special_tokens = model.post_processor.special_tokens
    assert special_tokens["[CLS]"].ids[0] == model.vocabulary["[CLS]"]
    assert special_tokens["[SEP]"].ids[0] == model.vocabulary["[SEP]"]

    assert len(model.added_tokens.root) == 116
    for token in model.added_tokens.root:
        assert model.vocabulary[token.content] == token.id

    call_tokenizer(model)


def test_basic_collapse() -> None:
    """Test that consolidating the vocabulary removes no tokens (BPE + ByteLevel has no dead tokens)."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 50368
    assert not model.model_delta.new_tokens
    assert not model.model_delta.removed_tokens

    call_tokenizer(model)


def test_decase() -> None:
    """Test the decase operation."""
    model = TokenizerModel.from_pretrained(_PATH)

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[CLS]", "Ġam", "sterdam", "[SEP]"]
    assert tok.encode(" Amsterdam").tokens == ["[CLS]", "ĠAmsterdam", "[SEP]"]

    model = model.decase_vocabulary(keep=True)
    assert model.vocabulary_size == 50368

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[cls]", "Ġamsterdam", "[sep]"]
    assert tok.encode(" Amsterdam").tokens == ["[cls]", "Ġamsterdam", "[sep]"]


def test_decase_prune() -> None:
    """Test the decase operation with duplicate removal."""
    model = TokenizerModel.from_pretrained(_PATH)
    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[CLS]", "Ġam", "sterdam", "[SEP]"]
    assert tok.encode(" Amsterdam").tokens == ["[CLS]", "ĠAmsterdam", "[SEP]"]

    model = model.decase_vocabulary(keep=False)
    assert model.vocabulary_size == 40932

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[cls]", "Ġamsterdam", "[sep]"]
    assert tok.encode(" Amsterdam").tokens == ["[cls]", "Ġamsterdam", "[sep]"]


def test_add_digits_pretokenizer() -> None:
    """Test adding a DigitsPreTokenizer as prefix removes space-prefixed digit tokens."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False), prefix=True)
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 49532

    removed = model.model_delta.removed_tokens
    assert len(removed) == 836
    for token in {"Ġ100", "Ġ200", "Ġ500", "Ġ2000", "Ġ1970"}:
        assert token in removed

    tok = model.to_tokenizer()
    assert tok.encode(" 2024").tokens == ["[CLS]", "Ġ", "20", "24", "[SEP]"]
    assert tok.encode(" 100").tokens == ["[CLS]", "Ġ", "100", "[SEP]"]
    assert tok.encode(" hello world").tokens == ["[CLS]", "Ġhello", "Ġworld", "[SEP]"]

    call_tokenizer(model)


def test_add_digits_pretokenizer_keep() -> None:
    """Test that consolidate with keep=True preserves vocabulary size after adding DigitsPreTokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False), prefix=True)
    model = model.consolidate_vocabulary(keep=True)
    assert model.vocabulary_size == 50368

    call_tokenizer(model)


def test_add_special_token() -> None:
    """Test adding a special token to ModernBERT's large vocabulary."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    model = model.add_addedtoken("[QUERY]", is_special=True)
    assert model.vocabulary_size == initial_size + 1
    assert "[QUERY]" in model.vocabulary
    added = model.added_tokens.get_token("[QUERY]")
    assert added is not None
    assert model.vocabulary["[QUERY]"] == added.id
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_remove_non_essential_added_token() -> None:
    """Test removing a non-essential added token (a PII placeholder)."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    assert "|||EMAIL_ADDRESS|||" in model.vocabulary
    # |||EMAIL_ADDRESS||| is at id=50277, above all remaining added tokens after removal
    model = model.remove_token_from_vocabulary("|||EMAIL_ADDRESS|||")
    assert model.vocabulary_size == initial_size - 1
    assert "|||EMAIL_ADDRESS|||" not in model.vocabulary
    assert model.added_tokens.get_token("|||EMAIL_ADDRESS|||") is None
    ids = sorted(model.vocabulary.values())
    assert ids == list(range(model.vocabulary_size))
    call_tokenizer(model)


def test_prune_added_tokens() -> None:
    """Test that pruning 116 added tokens leaves only the 4 structural special tokens."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert len(model.added_tokens.root) == 116
    pruned = model.prune_added_tokens()
    # Only [UNK], [CLS], [SEP], [PAD] survive — all whitespace/PII/other tokens are dropped
    assert pruned.vocabulary_size == 50256
    assert len(pruned.added_tokens.root) == 4
    for content in ("[UNK]", "[CLS]", "[SEP]", "[PAD]"):
        assert content in pruned.vocabulary
        assert pruned.added_tokens.get_token(content) is not None
    assert_vocabulary_consistent(pruned)
    call_tokenizer(pruned)


def test_added_tokens_ids_consistent_after_add() -> None:
    """Test that all existing added tokens still have correct IDs after adding a new one."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_addedtoken("[PASSAGE]", is_special=True)
    for token in model.added_tokens.root:
        assert token.content in model.vocabulary
        assert model.vocabulary[token.content] == token.id, (
            f"{token.content!r}: added_token.id={token.id}, vocab={model.vocabulary[token.content]}"
        )
    call_tokenizer(model)

from skeletoken import TokenizerModel
from skeletoken.postprocessors import ByteLevelPostProcessor
from skeletoken.pretokenizers import DigitsPreTokenizer
from tests.conftest import assert_vocabulary_consistent, call_tokenizer

_PATH = "tests/data/gpt2"


def test_load() -> None:
    """Test loading bert-base-uncased."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert model.vocabulary_size == 50257
    assert model.pad_token == None
    assert model.pad_token_id == None
    assert model.unk_token == "<|endoftext|>"
    assert model.unk_token_id == 50256
    assert model.subword_prefix == ""
    assert not model.adds_prefix_space
    assert isinstance(model.post_processor, ByteLevelPostProcessor)
    assert model.post_processor.add_prefix_space
    assert model.transforms_into_bytes

    assert len(model.added_tokens.root) == 1
    for token in model.added_tokens.root:
        assert model.vocabulary[token.content] == token.id

    call_tokenizer(model)


def test_basic_collapse() -> None:
    """Test collapsing the basic tokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 50257
    new_tokens = model.model_delta.new_tokens
    assert not new_tokens

    removed_tokens = model.model_delta.removed_tokens
    assert not removed_tokens

    call_tokenizer(model)


def test_decase() -> None:
    """Test the decase operation."""
    model = TokenizerModel.from_pretrained(_PATH)

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["Ġam", "sterdam"]
    assert tok.encode(" Amsterdam").tokens == ["ĠAmsterdam"]

    model = model.decase_vocabulary(keep=True)
    assert model.vocabulary_size == 50257

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["Ġamsterdam"]
    assert tok.encode(" amsterdam").tokens == ["Ġamsterdam"]


def test_decase_prune() -> None:
    """Test the decase operation."""
    model = TokenizerModel.from_pretrained(_PATH)
    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["Ġam", "sterdam"]
    assert tok.encode(" Amsterdam").tokens == ["ĠAmsterdam"]

    model = model.decase_vocabulary(keep=False)
    assert model.vocabulary_size == 39372

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["Ġamsterdam"]
    assert tok.encode(" amsterdam").tokens == ["Ġamsterdam"]


def test_add_digits_pretokenizer() -> None:
    """Test adding a DigitsPreTokenizer as prefix removes space-prefixed digit tokens."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False), prefix=True)
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 49559

    removed = model.model_delta.removed_tokens
    assert len(removed) == 698
    for token in {"Ġ2024", "Ġ100", "Ġ2000", "Ġ1970"}:
        assert token in removed

    tok = model.to_tokenizer()
    # Space-prefixed digit strings are now split into the Ġ marker plus raw digits.
    assert tok.encode(" 2024").tokens == ["Ġ", "20", "24"]
    assert tok.encode(" 100").tokens == ["Ġ", "100"]
    assert tok.encode(" hello world").tokens == ["Ġhello", "Ġworld"]

    call_tokenizer(model)


def test_add_digits_pretokenizer_keep() -> None:
    """Test that consolidate with keep=True preserves vocabulary size after adding DigitsPreTokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False), prefix=True)
    model = model.consolidate_vocabulary(keep=True)
    assert model.vocabulary_size == 50257

    call_tokenizer(model)


def test_add_fim_special_token() -> None:
    """Test adding a fill-in-middle special token to GPT-2."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    model = model.add_addedtoken("<|fim_prefix|>", is_special=True)
    assert model.vocabulary_size == initial_size + 1
    assert "<|fim_prefix|>" in model.vocabulary
    added = model.added_tokens.get_token("<|fim_prefix|>")
    assert added is not None
    assert added.special is True
    assert_vocabulary_consistent(model)
    tok = model.to_tokenizer()
    assert tok.encode("<|fim_prefix|>").tokens == ["<|fim_prefix|>"]
    call_tokenizer(model)


def test_add_multiple_fim_tokens() -> None:
    """Test adding all three FIM tokens at once and verify consistent IDs."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    fim_tokens = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    model = model.add_addedtokens(fim_tokens, is_special=True)
    assert model.vocabulary_size == initial_size + 3
    for token in fim_tokens:
        assert token in model.vocabulary
        added = model.added_tokens.get_token(token)
        assert added is not None
        assert model.vocabulary[token] == added.id
    # IDs should be assigned in order at the end of the vocabulary
    assert model.vocabulary["<|fim_prefix|>"] < model.vocabulary["<|fim_middle|>"]
    assert model.vocabulary["<|fim_middle|>"] < model.vocabulary["<|fim_suffix|>"]
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_prune_added_tokens_keeps_unk() -> None:
    """Test that prune_added_tokens keeps <|endoftext|> since it is the UNK token."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert model.unk_token == "<|endoftext|>"
    pruned = model.prune_added_tokens()
    assert "<|endoftext|>" in pruned.vocabulary
    assert pruned.added_tokens.get_token("<|endoftext|>") is not None
    # Only the UNK token remains; no BOS/EOS/PAD in GPT-2
    assert len(pruned.added_tokens.root) == 1
    assert pruned.vocabulary_size == model.vocabulary_size
    assert_vocabulary_consistent(pruned)
    call_tokenizer(pruned)


def test_add_then_remove_special_token() -> None:
    """Test that adding and removing a token restores the original vocabulary size."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    model = model.add_addedtoken("<|system|>", is_special=True)
    assert model.vocabulary_size == initial_size + 1
    model = model.remove_token_from_vocabulary("<|system|>")
    assert model.vocabulary_size == initial_size
    assert "<|system|>" not in model.vocabulary
    assert model.added_tokens.get_token("<|system|>") is None
    assert_vocabulary_consistent(model)
    call_tokenizer(model)

from skeletoken import TokenizerModel
from skeletoken.postprocessors import TemplatePostProcessor
from skeletoken.pretokenizers import DigitsPreTokenizer
from tests.conftest import assert_vocabulary_consistent, call_tokenizer

_PATH = "tests/data/multilingual-e5-base"


def test_load() -> None:
    """Test loading multilingual-e5-base."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert model.vocabulary_size == 250002
    assert model.pad_token == "<pad>"
    assert model.pad_token_id == 1
    assert model.unk_token == "<unk>"
    assert model.unk_token_id == 3
    assert model.subword_prefix is None
    assert model.adds_prefix_space is None
    assert not model.transforms_into_bytes
    assert isinstance(model.post_processor, TemplatePostProcessor)
    assert model.vocabulary["<s>"] == 0
    assert model.vocabulary["</s>"] == 2
    special_tokens = model.post_processor.special_tokens
    assert special_tokens["<s>"].ids[0] == model.vocabulary["<s>"]
    assert special_tokens["</s>"].ids[0] == model.vocabulary["</s>"]

    assert len(model.added_tokens.root) == 5
    for token in model.added_tokens.root:
        assert model.vocabulary[token.content] == token.id

    call_tokenizer(model)


def test_basic_collapse() -> None:
    """Test collapsing the vocabulary removes only the one unreachable control character."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 250001
    assert not model.model_delta.new_tokens

    removed = model.model_delta.removed_tokens
    assert len(removed) == 1
    assert "\x85" in removed

    call_tokenizer(model)


def test_decase() -> None:
    """Test the decase operation."""
    model = TokenizerModel.from_pretrained(_PATH)

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["<s>", "▁am", "ster", "dam", "</s>"]
    assert tok.encode(" Amsterdam").tokens == ["<s>", "▁Amsterdam", "</s>"]

    model = model.decase_vocabulary(keep=True)
    assert model.vocabulary_size == 250002

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["<s>", "▁amsterdam", "</s>"]
    assert tok.encode(" Amsterdam").tokens == ["<s>", "▁amsterdam", "</s>"]


def test_decase_prune() -> None:
    """Test the decase operation with duplicate removal."""
    model = TokenizerModel.from_pretrained(_PATH)
    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["<s>", "▁am", "ster", "dam", "</s>"]
    assert tok.encode(" Amsterdam").tokens == ["<s>", "▁Amsterdam", "</s>"]

    model = model.decase_vocabulary(keep=False)
    assert model.vocabulary_size == 229866

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["<s>", "▁amsterdam", "</s>"]
    assert tok.encode(" Amsterdam").tokens == ["<s>", "▁amsterdam", "</s>"]


def test_add_digits_pretokenizer() -> None:
    """Test adding a DigitsPreTokenizer removes tokens that straddle digit/non-digit boundaries."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False), prefix=True)
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 248428

    removed = model.model_delta.removed_tokens
    assert len(removed) == 1574
    for token in {"▁3.2", "10.2014", "▁1:2", "%3"}:
        assert token in removed

    tok = model.to_tokenizer()
    assert tok.encode(" hello world").tokens == ["<s>", "▁hell", "o", "▁world", "</s>"]
    assert tok.encode(" 100 dollars").tokens == ["<s>", "▁100", "▁dollars", "</s>"]

    call_tokenizer(model)


def test_add_digits_pretokenizer_keep() -> None:
    """Test that consolidate with keep=True preserves vocabulary size after adding DigitsPreTokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False), prefix=True)
    model = model.consolidate_vocabulary(keep=True)
    assert model.vocabulary_size == 250002

    call_tokenizer(model)


def test_add_query_prefix_token() -> None:
    """Test adding a query prefix special token for embedding use-cases."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    model = model.add_addedtoken("query:", is_special=True)
    assert model.vocabulary_size == initial_size + 1
    assert "query:" in model.vocabulary
    added = model.added_tokens.get_token("query:")
    assert added is not None
    assert model.vocabulary["query:"] == added.id
    assert_vocabulary_consistent(model)
    tok = model.to_tokenizer()
    # lstrip=True, rstrip=True (defaults) cause the token to absorb surrounding whitespace
    assert tok.encode("query: multilingual search").tokens[1] == "query: "
    call_tokenizer(model)


def test_add_multiple_prefix_tokens() -> None:
    """Test adding query and passage prefix tokens used by e5-style embedding models."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    prefix_tokens = ["query:", "passage:"]
    model = model.add_addedtokens(prefix_tokens, is_special=True)
    assert model.vocabulary_size == initial_size + 2
    for token in prefix_tokens:
        assert token in model.vocabulary
        added = model.added_tokens.get_token(token)
        assert added is not None
        assert model.vocabulary[token] == added.id
    assert model.vocabulary["query:"] < model.vocabulary["passage:"]
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_remove_mask_token() -> None:
    """Test removing <mask> leaves the remaining added tokens with consistent IDs."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    assert "<mask>" in model.vocabulary
    model = model.remove_token_from_vocabulary("<mask>")
    assert model.vocabulary_size == initial_size - 1
    assert "<mask>" not in model.vocabulary
    assert model.added_tokens.get_token("<mask>") is None
    # <mask>=250001 is the highest-ID added token; all others (<s>=0, <pad>=1,
    # </s>=2, <unk>=3) are unaffected by its removal.
    assert_vocabulary_consistent(model)
    # IDs formerly below mask_id should shift down by 0 (they're all lower)
    assert model.vocabulary.get("<s>") == 0
    assert model.vocabulary.get("<pad>") == 1
    call_tokenizer(model)


def test_prune_added_tokens() -> None:
    """Test that prune_added_tokens removes <mask> and keeps BOS/EOS/UNK/PAD."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert "<mask>" in model.vocabulary
    pruned = model.prune_added_tokens()
    assert "<mask>" not in pruned.vocabulary
    assert pruned.added_tokens.get_token("<mask>") is None
    assert "<s>" in pruned.vocabulary
    assert "</s>" in pruned.vocabulary
    assert "<unk>" in pruned.vocabulary
    assert "<pad>" in pruned.vocabulary
    assert len(pruned.added_tokens.root) == 4
    assert pruned.vocabulary_size == model.vocabulary_size - 1
    assert_vocabulary_consistent(pruned)
    call_tokenizer(pruned)

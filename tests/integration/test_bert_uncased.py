from skeletoken import TokenizerModel
from skeletoken.postprocessors import TemplatePostProcessor
from skeletoken.pretokenizers import DigitsPreTokenizer
from tests.conftest import assert_vocabulary_consistent, call_tokenizer

_PATH = "tests/data/bert-base-uncased"


def test_load() -> None:
    """Test loading bert-base-uncased."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert model.vocabulary_size == 30522
    assert model.pad_token == "[PAD]"
    assert model.pad_token_id == 0
    assert model.unk_token == "[UNK]"
    assert model.unk_token_id == 100
    assert model.subword_prefix == "##"
    assert not model.adds_prefix_space
    assert isinstance(model.post_processor, TemplatePostProcessor)
    assert model.vocabulary["[CLS]"] == 101
    assert model.vocabulary["[SEP]"] == 102
    special_tokens = model.post_processor.special_tokens
    assert special_tokens["[CLS]"].ids[0] == model.vocabulary["[CLS]"]
    assert special_tokens["[SEP]"].ids[0] == model.vocabulary["[SEP]"]
    assert not model.transforms_into_bytes

    assert len(model.added_tokens.root) == 5
    for token in model.added_tokens.root:
        assert model.vocabulary[token.content] == token.id

    call_tokenizer(model)


def test_basic_collapse() -> None:
    """Test collapsing the basic tokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 29527
    assert model.model_delta.new_tokens == {"[cls]": 2, "[mask]": 4, "[sep]": 3, "[unk]": 1}
    assert model.unk_token == "[unk]"
    assert model.unk_token_id == 1
    assert model.pad_token == "[PAD]"
    assert model.pad_token_id == 0

    removed_tokens = model.model_delta.removed_tokens
    assert len(removed_tokens) == 999
    assert [x for x in removed_tokens if not x.startswith("[")] == ["..."]

    call_tokenizer(model)


def test_set_prefix() -> None:
    """Test whether setting the subword prefix removes all useless tokens."""
    model = TokenizerModel.from_pretrained(_PATH)
    model.subword_prefix = ""

    new_tokens = set(model.model_delta.new_tokens)
    for token in model.model_delta.removed_tokens:
        if token.startswith("##"):
            # Subwords
            assert token.removeprefix("##") in new_tokens
        else:
            # Special tokens like `[PAD]` get normalized to `[pad]`.
            assert token.lower() in new_tokens

    call_tokenizer(model)


def test_decase() -> None:
    """Test the decase operation."""
    model = TokenizerModel.from_pretrained(_PATH)

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[CLS]", "amsterdam", "[SEP]"]
    assert tok.encode(" Amsterdam").tokens == ["[CLS]", "amsterdam", "[SEP]"]

    model = model.decase_vocabulary(keep=True)
    assert model.vocabulary_size == 30522

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]
    assert tok.encode(" Amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]


def test_decase_prune() -> None:
    """Test the decase operation."""
    model = TokenizerModel.from_pretrained(_PATH)
    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[CLS]", "amsterdam", "[SEP]"]
    assert tok.encode(" Amsterdam").tokens == ["[CLS]", "amsterdam", "[SEP]"]

    model = model.decase_vocabulary(keep=False)
    assert model.vocabulary_size == 29527

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]
    assert tok.encode(" Amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]


def test_add_digits_pretokenizer() -> None:
    """Test adding a DigitsPreTokenizer removes tokens that straddle digit/non-digit boundaries."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False))
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 29395
    assert model.model_delta.new_tokens == {"[unk]": 1, "[cls]": 2, "[sep]": 3, "[mask]": 4}

    removed = model.model_delta.removed_tokens
    assert len(removed) == 1131
    for token in {"51st", "2000s", "00pm", "a1"}:
        assert token in removed

    tok = model.to_tokenizer()
    assert tok.encode("51st street").tokens == ["[cls]", "51", "st", "street", "[sep]"]
    assert tok.encode("2000s music").tokens == ["[cls]", "2000", "s", "music", "[sep]"]
    assert tok.encode("hello world").tokens == ["[cls]", "hello", "world", "[sep]"]

    call_tokenizer(model)


def test_add_digits_pretokenizer_keep() -> None:
    """Test that consolidate with keep=True preserves vocabulary size after adding DigitsPreTokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False))
    model = model.consolidate_vocabulary(keep=True)
    assert model.vocabulary_size == 30522

    call_tokenizer(model)


def test_add_digits_pretokenizer_individual() -> None:
    """Test adding a DigitsPreTokenizer with individual_digits=True splits each digit separately."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=True))
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 28478

    tok = model.to_tokenizer()
    assert tok.encode("in 2024").tokens == ["[cls]", "in", "2", "0", "2", "4", "[sep]"]
    assert tok.encode("100 dollars").tokens == ["[cls]", "1", "0", "0", "dollars", "[sep]"]
    assert tok.encode("hello world").tokens == ["[cls]", "hello", "world", "[sep]"]

    call_tokenizer(model)


def test_add_special_token() -> None:
    """Test adding a new special token grows the vocabulary consistently."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    # Use lstrip/rstrip=False to match BERT's existing special token convention
    model = model.add_addedtoken("[GENE]", is_special=True, lstrip=False, rstrip=False)
    assert model.vocabulary_size == initial_size + 1
    assert "[GENE]" in model.vocabulary
    added = model.added_tokens.get_token("[GENE]")
    assert added is not None
    assert added.special is True
    assert_vocabulary_consistent(model)
    tok = model.to_tokenizer()
    assert tok.encode("[GENE] expression").tokens == ["[CLS]", "[GENE]", "expression", "[SEP]"]
    call_tokenizer(model)


def test_add_regular_token() -> None:
    """Test adding a regular vocabulary token makes it encode as a single unit."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    model = model.add_token_to_vocabulary("skeletoken")
    assert model.vocabulary_size == initial_size + 1
    assert "skeletoken" in model.vocabulary
    assert_vocabulary_consistent(model)
    tok = model.to_tokenizer()
    assert tok.encode("skeletoken").tokens == ["[CLS]", "skeletoken", "[SEP]"]
    call_tokenizer(model)


def test_add_multiple_special_tokens() -> None:
    """Test adding several special tokens at once keeps vocabulary consistent."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    new_tokens = ["[PROTEIN]", "[DISEASE]", "[DRUG]"]
    model = model.add_addedtokens(new_tokens, is_special=True)
    assert model.vocabulary_size == initial_size + 3
    for token in new_tokens:
        assert token in model.vocabulary
        added = model.added_tokens.get_token(token)
        assert added is not None
        assert model.vocabulary[token] == added.id
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_remove_mask_token() -> None:
    """Test removing [MASK] shrinks the vocabulary and added_tokens consistently."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    assert "[MASK]" in model.vocabulary
    model = model.remove_token_from_vocabulary("[MASK]")
    assert model.vocabulary_size == initial_size - 1
    assert "[MASK]" not in model.vocabulary
    assert model.added_tokens.get_token("[MASK]") is None
    # Remaining added tokens ([PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102) all have
    # lower IDs than [MASK]=103, so their IDs are unaffected by the removal.
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_prune_added_tokens() -> None:
    """Test that prune_added_tokens drops [MASK] and keeps BOS/EOS/UNK/PAD."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert "[MASK]" in model.vocabulary
    model = model.prune_added_tokens()
    assert "[MASK]" not in model.vocabulary
    assert model.added_tokens.get_token("[MASK]") is None
    assert "[CLS]" in model.vocabulary
    assert "[SEP]" in model.vocabulary
    assert "[UNK]" in model.vocabulary
    assert "[PAD]" in model.vocabulary
    assert len(model.added_tokens.root) == 4
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_replace_mask_token() -> None:
    """Test replacing [MASK] with a custom token preserves the vocabulary index."""
    model = TokenizerModel.from_pretrained(_PATH)
    mask_id = model.vocabulary["[MASK]"]
    model = model.replace_token_in_vocabulary("[MASK]", "[CUSTOMMASK]", preprocess_token=False)
    assert "[MASK]" not in model.vocabulary
    assert "[CUSTOMMASK]" in model.vocabulary
    assert model.vocabulary["[CUSTOMMASK]"] == mask_id
    assert model.added_tokens.get_token("[MASK]") is None
    assert model.added_tokens.get_token("[CUSTOMMASK]") is not None
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_vocabulary_ids_contiguous_after_removal() -> None:
    """Test that vocabulary IDs are contiguous after removing [MASK]."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.remove_token_from_vocabulary("[MASK]")
    ids = sorted(model.vocabulary.values())
    assert ids == list(range(model.vocabulary_size))
    call_tokenizer(model)

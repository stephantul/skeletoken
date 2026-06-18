from skeletoken import TokenizerModel
from skeletoken.postprocessors import TemplatePostProcessor
from skeletoken.pretokenizers import DigitsPreTokenizer
from tests.conftest import assert_vocabulary_consistent, call_tokenizer

_PATH = "tests/data/bert-base-cased"


def test_load() -> None:
    """Test loading bert-base-uncased."""
    model = TokenizerModel.from_pretrained(_PATH)
    assert model.vocabulary_size == 28996
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
    assert model.vocabulary_size == 28894
    assert model.model_delta.new_tokens == {}
    assert model.unk_token == "[UNK]"
    assert model.unk_token_id == 1
    assert model.pad_token == "[PAD]"
    assert model.pad_token_id == 0

    removed_tokens = model.model_delta.removed_tokens
    assert len(removed_tokens) == 102
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
    assert tok.encode(" amsterdam").tokens == ["[CLS]", "am", "##ster", "##dam", "[SEP]"]
    assert tok.encode(" Amsterdam").tokens == ["[CLS]", "Amsterdam", "[SEP]"]

    model = model.decase_vocabulary(keep=True)
    assert model.vocabulary_size == 28996

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]
    assert tok.encode(" Amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]


def test_decase_prune() -> None:
    """Test the decase operation."""
    model = TokenizerModel.from_pretrained(_PATH)
    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[CLS]", "am", "##ster", "##dam", "[SEP]"]
    assert tok.encode(" Amsterdam").tokens == ["[CLS]", "Amsterdam", "[SEP]"]

    model = model.decase_vocabulary(keep=False)
    assert model.vocabulary_size == 25025

    tok = model.to_tokenizer()
    assert tok.encode(" amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]
    assert tok.encode(" Amsterdam").tokens == ["[cls]", "amsterdam", "[sep]"]


def test_add_digits_pretokenizer() -> None:
    """Test adding a DigitsPreTokenizer removes tokens that straddle digit/non-digit boundaries."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False))
    model = model.consolidate_vocabulary(keep=False)
    assert model.vocabulary_size == 28800
    assert model.model_delta.new_tokens == {}

    removed = model.model_delta.removed_tokens
    assert len(removed) == 196
    for token in {"1970s", "49ers", "23rd", "2000s"}:
        assert token in removed

    tok = model.to_tokenizer()
    assert tok.encode("51st street").tokens == ["[CLS]", "51", "s", "##t", "street", "[SEP]"]
    assert tok.encode("in 2024").tokens == ["[CLS]", "in", "202", "##4", "[SEP]"]
    assert tok.encode("hello world").tokens == ["[CLS]", "hello", "world", "[SEP]"]

    call_tokenizer(model)


def test_add_digits_pretokenizer_keep() -> None:
    """Test that consolidate with keep=True preserves vocabulary size after adding DigitsPreTokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.add_pre_tokenizer(DigitsPreTokenizer(individual_digits=False))
    model = model.consolidate_vocabulary(keep=True)
    assert model.vocabulary_size == 28996

    call_tokenizer(model)


def test_add_special_token() -> None:
    """Test adding a new special token to the cased vocabulary."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    # Use lstrip/rstrip=False to match BERT's existing special token convention
    model = model.add_addedtoken("[ORG]", is_special=True, lstrip=False, rstrip=False)
    assert model.vocabulary_size == initial_size + 1
    assert "[ORG]" in model.vocabulary
    added = model.added_tokens.get_token("[ORG]")
    assert added is not None
    assert added.special is True
    assert_vocabulary_consistent(model)
    tok = model.to_tokenizer()
    assert tok.encode("[ORG] Google").tokens == ["[CLS]", "[ORG]", "Google", "[SEP]"]
    call_tokenizer(model)


def test_add_regular_token() -> None:
    """Test adding a cased token that didn't exist; case is preserved in the cased model."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    # 'Skeletoken' does not exist in bert-base-cased vocabulary
    model = model.add_token_to_vocabulary("Skeletoken")
    assert model.vocabulary_size == initial_size + 1
    assert "Skeletoken" in model.vocabulary
    assert_vocabulary_consistent(model)
    tok = model.to_tokenizer()
    assert tok.encode("Skeletoken").tokens == ["[CLS]", "Skeletoken", "[SEP]"]
    call_tokenizer(model)


def test_remove_mask_token() -> None:
    """Test removing [MASK] leaves all other added tokens with consistent IDs."""
    model = TokenizerModel.from_pretrained(_PATH)
    initial_size = model.vocabulary_size
    assert "[MASK]" in model.vocabulary
    model = model.remove_token_from_vocabulary("[MASK]")
    assert model.vocabulary_size == initial_size - 1
    assert "[MASK]" not in model.vocabulary
    assert model.added_tokens.get_token("[MASK]") is None
    # [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102 all lie below [MASK]=103 and are unaffected
    assert_vocabulary_consistent(model)
    call_tokenizer(model)


def test_prune_added_tokens() -> None:
    """Test that prune_added_tokens removes [MASK] and keeps BOS/EOS/UNK/PAD."""
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
    """Test that replacing [MASK] with [BLANK] preserves the vocabulary index."""
    model = TokenizerModel.from_pretrained(_PATH)
    mask_id = model.vocabulary["[MASK]"]
    model = model.replace_token_in_vocabulary("[MASK]", "[BLANK]", preprocess_token=False)
    assert "[MASK]" not in model.vocabulary
    assert "[BLANK]" in model.vocabulary
    assert model.vocabulary["[BLANK]"] == mask_id
    assert model.added_tokens.get_token("[MASK]") is None
    assert model.added_tokens.get_token("[BLANK]") is not None
    assert_vocabulary_consistent(model)
    call_tokenizer(model)

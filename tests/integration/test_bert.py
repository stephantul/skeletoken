from skeletoken import TokenizerModel
from skeletoken.postprocessors import TemplatePostProcessor

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


def test_basic_collapse() -> None:
    """Test collapsing the basic tokenizer."""
    model = TokenizerModel.from_pretrained(_PATH)
    model = model.collapse_vocabulary(keep_duplicates=False)
    assert model.vocabulary_size == 29527
    assert model.model_delta.new_tokens == {"[cls]": 2, "[mask]": 4, "[sep]": 3, "[unk]": 1}
    assert model.unk_token == "[unk]"
    assert model.unk_token_id == 1
    assert model.pad_token == "[PAD]"
    assert model.pad_token_id == 0

    removed_tokens = model.model_delta.removed_tokens
    assert len(removed_tokens) == 999
    assert [x for x in removed_tokens if not x.startswith("[")] == ["..."]


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

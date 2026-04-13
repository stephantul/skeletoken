from skeletoken import TokenizerModel
from skeletoken.postprocessors import ByteLevelPostProcessor
from tests.conftest import call_tokenizer

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
    model = model.collapse_vocabulary(keep_duplicates=False)
    assert model.vocabulary_size == 50257
    new_tokens = model.model_delta.new_tokens
    assert not new_tokens

    removed_tokens = model.model_delta.removed_tokens
    assert not removed_tokens

    call_tokenizer(model)

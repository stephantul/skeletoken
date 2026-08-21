from transformers import PreTrainedTokenizerFast

from skeletoken import TokenizerModel
from skeletoken.padding import PaddingDirection
from skeletoken.truncation import Truncation, TruncationDirection, TruncationStrategy
from tests.conftest import assert_to_transformers_roundtrip

_BERT_PATH = "tests/data/bert-base-cased"
_GPT2_PATH = "tests/data/gpt2"


def test_roundtrip_preserves_special_tokens() -> None:
    """Special tokens on the exported transformers tokenizer match the skeletoken model."""
    model = TokenizerModel.from_pretrained(_BERT_PATH)
    transformers_tokenizer = model.to_transformers()
    assert model.bos is not None
    assert model.eos is not None
    assert transformers_tokenizer.pad_token == model.pad_token
    assert transformers_tokenizer.unk_token == model.unk_token
    assert transformers_tokenizer.bos_token == model.bos[0]
    assert transformers_tokenizer.eos_token == model.eos[0]


def test_to_transformers_propagates_padding_config() -> None:
    """`model.padding` carries over to the exported tokenizer's padding attributes."""
    model = TokenizerModel.from_pretrained(_BERT_PATH)
    model.pad_token = "[PAD]"
    assert model.padding is not None
    model.padding.direction = PaddingDirection.LEFT
    model.padding.pad_to_multiple_of = 8
    model.padding.pad_type_id = 3

    transformers_tokenizer = model.to_transformers()

    assert transformers_tokenizer.padding_side == "left"
    assert transformers_tokenizer.pad_to_multiple_of == 8
    assert transformers_tokenizer.pad_token_type_id == 3


def test_to_transformers_round_trips_clean_up_tokenization_spaces() -> None:
    """`clean_up_tokenization_spaces=True` survives a `from_transformers_tokenizer()` -> `to_transformers()` round trip."""
    model = TokenizerModel.from_pretrained(_BERT_PATH)
    base_tokenizer = model.to_transformers()
    source = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer.backend_tokenizer,
        clean_up_tokenization_spaces=True,
    )

    reloaded = TokenizerModel.from_transformers_tokenizer(source)
    exported = reloaded.to_transformers()

    assert exported.clean_up_tokenization_spaces is True

    ids = source.encode("the weather isn't great , unfortunately .", add_special_tokens=False)
    assert source.decode(ids) == "the weather isn't great, unfortunately."
    assert exported.decode(ids) == source.decode(ids)
    assert reloaded.to_tokenizer().decode(ids) == source.decode(ids)


def test_to_transformers_propagates_add_prefix_space() -> None:
    """`model.adds_prefix_space` carries over to the exported tokenizer's `add_prefix_space`."""
    model = TokenizerModel.from_pretrained(_GPT2_PATH)
    model.adds_prefix_space = True
    transformers_tokenizer = model.to_transformers()
    assert transformers_tokenizer.add_prefix_space is True
    assert transformers_tokenizer.tokenize("hello world") == ["Ġhello", "Ġworld"]


def test_to_transformers_truncation_max_length() -> None:
    """Truncation max_length propagates to `model_max_length` on export."""
    model = TokenizerModel.from_pretrained(_BERT_PATH)
    model.truncation = Truncation(
        direction=TruncationDirection.RIGHT,
        max_length=16,
        strategy=TruncationStrategy.LONGEST_FIRST,
        stride=0,
    )
    transformers_tokenizer = model.to_transformers()
    assert transformers_tokenizer.model_max_length == 16


def test_to_transformers_does_not_mutate_padding() -> None:
    """Regression: exporting must not clear the source model's padding config.

    Setting `pad_token` creates a "basic" Fixed(0) padding hack, and `is_basic_padding`
    used to be stripped by mutating `self.padding` in place inside `to_transformers`,
    permanently losing the pad token on the source model.
    """
    model = TokenizerModel.from_pretrained(_BERT_PATH)
    model.pad_token = "[PAD]"
    assert model.padding is not None
    padding_before = model.padding.model_copy(deep=True)

    transformers_tokenizer = model.to_transformers()

    assert model.padding == padding_before
    assert model.pad_token == "[PAD]"
    assert transformers_tokenizer.pad_token == "[PAD]"

    # A second export from the same (unmutated) model must behave identically.
    second_transformers_tokenizer = model.to_transformers()
    assert second_transformers_tokenizer.pad_token == "[PAD]"


def test_to_transformers_reflects_pad_token_change_after_warming_cache() -> None:
    """Regression: the cached `.tokenizer` used internally must not go stale.

    Accessing `model.tokenizer` builds and caches a real `Tokenizer`. If a later
    in-place vocabulary change (like setting `pad_token`) doesn't invalidate that
    cache, exports made afterwards can silently use a stale vocabulary.
    """
    model = TokenizerModel.from_pretrained(_GPT2_PATH)
    _ = model.tokenizer  # warm the cache

    model.pad_token = "<|pad|>"
    transformers_tokenizer = model.to_transformers()

    assert transformers_tokenizer.pad_token == "<|pad|>"
    assert "<|pad|>" in model.vocabulary
    assert transformers_tokenizer.convert_tokens_to_ids("<|pad|>") == model.vocabulary["<|pad|>"]


def test_to_transformers_after_prompt_and_pad_token_change() -> None:
    """Regression: prompt + pad_token edits on the same model must both survive export."""
    model = TokenizerModel.from_pretrained(_BERT_PATH)
    model.prompt = "search query:"
    model.pad_token = "[PAD]"

    assert model.padding is not None
    assert_to_transformers_roundtrip(model, ["Amsterdam is a city", "Amsterdam"])

    tokenizer = model.to_tokenizer()
    assert model.prompt is not None
    prompt_len = len(model.prompt)
    encoded = tokenizer.encode("Amsterdam")
    assert encoded.tokens[1 : 1 + prompt_len] == model.prompt

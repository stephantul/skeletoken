import json
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE as TokenizersBPE
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2 import GPT2Tokenizer

from skeletoken.addedtoken import AddedTokens
from skeletoken.base import TokenizerModel
from skeletoken.common import PrependScheme
from skeletoken.merges import Merges
from skeletoken.models import BPE, ModelType, WordPiece
from skeletoken.normalizers import (
    ByteLevelNormalizer,
    LowercaseNormalizer,
    NFKCNormalizer,
    Normalizer,
    NormalizerSequence,
    ReplaceNormalizer,
)
from skeletoken.padding import Padding
from skeletoken.post_processors import (
    BertPostProcessor,
    ByteLevelPostProcessor,
    PostProcessorSequence,
    RobertaPostProcessor,
    SpecialTokenInfo,
    TemplatePostProcessor,
    Token,
    TokenSequence,
    TokenType,
)
from skeletoken.pre_tokenizers import (
    Behavior,
    BertPreTokenizer,
    ByteLevelPreTokenizer,
    FixedLengthPreTokenizer,
    MetaspacePreTokenizer,
    PreTokenizerSequence,
    SplitPreTokenizer,
    StringPattern,
    WhitespacePreTokenizer,
)
from skeletoken.vocabulary import Vocabulary
from tests.conftest import assert_bpe_merges_consistent, assert_vocabulary_consistent, call_tokenizer


def test_post_init(small_tokenizer_json: dict[str, Any]) -> None:
    """Test the post-initialization of the TokenizerModel."""
    # Remove UNK from the vocab
    small_tokenizer_json["model"]["vocab"].pop("[UNK]")
    # Remove UNK as an added token.
    small_tokenizer_json["added_tokens"] = []
    # Because the unk_token on the model is still [UNK], we will re-add it.
    model = TokenizerModel.model_validate(small_tokenizer_json)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.model, WordPiece)
    assert isinstance(model.added_tokens, AddedTokens)
    assert model.normalizer is None
    assert model.pre_tokenizer is None
    assert model.post_processor is None
    assert model.decoder is None
    assert model.unk_token == "[UNK]"
    assert "[UNK]" in model.model.vocab.vocabulary
    assert model.added_tokens.get_token("[UNK]") is not None


def test_add_addedtoken(small_tokenizer: Tokenizer) -> None:
    """Test the add_addedtoken functionality."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_addedtoken("[OSTENTATIOUS]")
    # Test if it gets added to both the model and the vocabulary.
    assert model.added_tokens.get_token("[OSTENTATIOUS]") is not None
    assert "[OSTENTATIOUS]" in model.model.vocab


def test_turn_into_addedtoken(small_tokenizer: Tokenizer) -> None:
    """Test turning a regular token into an added token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_addedtoken("[OSTENTATIOUS]")
    # Should not crash, it is already an added token.
    model._turn_into_addedtoken("[OSTENTATIOUS]")
    # Should crash because it is not a regular token.
    with pytest.raises(ValueError):
        model._turn_into_addedtoken("bababbaa")


def test_pad_vocabulary_to_multiple_of(small_tokenizer: Tokenizer) -> None:
    """Test padding the vocabulary to a multiple of `n`."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    original_size = model.vocabulary_size
    assert original_size % 8 != 0

    padded = model.pad_vocabulary_to_multiple_of(n=8)
    assert padded.vocabulary_size % 8 == 0
    assert padded.vocabulary_size > original_size

    # The original model is untouched.
    assert model.vocabulary_size == original_size

    new_tokens = [token for token in padded.added_tokens.root if token.content.startswith("new_")]
    assert len(new_tokens) == padded.vocabulary_size - original_size
    for token in new_tokens:
        assert token.special
        assert token.content in padded.vocabulary

    assert_vocabulary_consistent(padded)
    call_tokenizer(padded)


def test_pad_vocabulary_to_multiple_of_noop(small_tokenizer: Tokenizer) -> None:
    """Test that no tokens are added when the vocabulary is already a multiple of `n`."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    n = model.vocabulary_size

    padded = model.pad_vocabulary_to_multiple_of(n=n)
    assert padded.vocabulary_size == model.vocabulary_size
    assert not [token for token in padded.added_tokens.root if token.content.startswith("new_")]


def test_pad_vocabulary_to_multiple_of_skips_existing_tokens(small_tokenizer: Tokenizer) -> None:
    """Test that padding skips candidate names that already exist in the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_token_to_vocabulary("new_0", preprocess_token=False)
    original_size = model.vocabulary_size

    padded = model.pad_vocabulary_to_multiple_of(n=8)
    new_tokens = sorted(token.content for token in padded.added_tokens.root if token.content.startswith("new_"))
    assert "new_0" not in new_tokens
    assert len(new_tokens) == padded.vocabulary_size - original_size

    assert_vocabulary_consistent(padded)


def test_pad_vocabulary_to_multiple_of_does_not_add_merges() -> None:
    """Test that padding tokens are added as AddedTokens and never touch the BPE merge table."""
    vocab_tokens = list("abcdefghijklmnop") + ["ab", "cd"]
    bpe = BPE(
        vocab=Vocabulary({c: i for i, c in enumerate(vocab_tokens)}),
        merges=Merges([("a", "b"), ("c", "d")]),
        dropout=None,
        unk_token=None,
        continuing_subword_prefix="",
        end_of_word_suffix=None,
        fuse_unk=False,
        byte_fallback=False,
        ignore_merges=False,
    )
    model = TokenizerModel(model=bpe)
    assert isinstance(model.model, BPE)
    original_merges = list(model.model.merges.root)
    original_size = model.vocabulary_size

    padded = model.pad_vocabulary_to_multiple_of(n=32)
    assert padded.vocabulary_size == 32
    assert padded.vocabulary_size > original_size
    assert isinstance(padded.model, BPE)

    assert list(padded.model.merges.root) == original_merges

    assert_vocabulary_consistent(padded)
    assert_bpe_merges_consistent(padded)


def test_tokenizer_model_from_tokenizer(small_tokenizer: Tokenizer) -> None:
    """Test creating a TokenizerModel from a Tokenizer instance."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.model, WordPiece)
    assert isinstance(model.added_tokens, AddedTokens)
    assert model.normalizer is None
    assert model.pre_tokenizer is None
    assert model.post_processor is None
    assert model.decoder is None


def test_add_pre_tokenizer(small_tokenizer: Tokenizer) -> None:
    """Test adding a pre-tokenizer to the TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    model = model.add_pre_tokenizer(pre_tokenizer)

    # Tests removing None and adding a pre-tokenizer
    assert model.pre_tokenizer is not None
    assert isinstance(model.pre_tokenizer, ByteLevelPreTokenizer)

    model = model.add_pre_tokenizer(pre_tokenizer)

    # Tests adding a second pre-tokenizer and turning it into a sequence
    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 2

    model = model.add_pre_tokenizer(pre_tokenizer)

    # Tests adding a third pre-tokenizer and keeping it as a sequence
    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 3


def test_add_byte_pretokenizer(small_tokenizer: Tokenizer) -> None:
    """Test adding a byte-level pre-tokenizer to the TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=True, trim_offsets=False)
    model = model.add_pre_tokenizer(pre_tokenizer)

    assert model.pre_tokenizer is not None
    assert isinstance(model.pre_tokenizer, ByteLevelPreTokenizer)

    model = model.add_pre_tokenizer(BertPreTokenizer(), prefix=True)

    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 2
    assert isinstance(model.pre_tokenizer.pretokenizers[0], BertPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[1], ByteLevelPreTokenizer)

    model = model.add_pre_tokenizer(WhitespacePreTokenizer())

    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 3

    assert isinstance(model.pre_tokenizer.pretokenizers[0], BertPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[1], ByteLevelPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[2], WhitespacePreTokenizer)

    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_pre_tokenizer(BertPreTokenizer())

    assert model.pre_tokenizer is not None
    assert isinstance(model.pre_tokenizer, BertPreTokenizer)

    model = model.add_pre_tokenizer(WhitespacePreTokenizer())

    assert model.pre_tokenizer is not None
    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 2
    assert isinstance(model.pre_tokenizer.pretokenizers[0], BertPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[1], WhitespacePreTokenizer)

    model = model.add_pre_tokenizer(FixedLengthPreTokenizer(length=10))

    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 3
    assert isinstance(model.pre_tokenizer.pretokenizers[0], BertPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[1], WhitespacePreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[2], FixedLengthPreTokenizer)

    model = model.add_pre_tokenizer(FixedLengthPreTokenizer(length=50), prefix=True)

    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 4
    assert isinstance(model.pre_tokenizer.pretokenizers[0], FixedLengthPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[1], BertPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[2], WhitespacePreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[3], FixedLengthPreTokenizer)


def test_add_normalizer(small_tokenizer: Tokenizer) -> None:
    """Test adding a normalizer to the TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    normalizer = ByteLevelNormalizer()
    model = model.add_normalizer(normalizer)

    # Tests removing None and adding a normalizer
    assert model.normalizer is not None
    assert isinstance(model.normalizer, ByteLevelNormalizer)

    model = model.add_normalizer(normalizer)

    # Tests adding a second normalizer and turning it into a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 2

    model = model.add_normalizer(normalizer)

    # Tests adding a third normalizer and keeping it as a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 3

    model = model.add_normalizer(LowercaseNormalizer(), prefix=True)

    # Tests adding a third normalizer and keeping it as a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 4
    assert isinstance(model.normalizer.normalizers[0], LowercaseNormalizer)


def test_add_normalizer_prefix(small_tokenizer: Tokenizer) -> None:
    """Add a normalizer prefix."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    normalizer: Normalizer = ByteLevelNormalizer()
    model = model.add_normalizer(normalizer, prefix=True)

    # Tests removing None and adding a normalizer
    assert model.normalizer is not None
    assert isinstance(model.normalizer, ByteLevelNormalizer)

    normalizer = LowercaseNormalizer()
    model = model.add_normalizer(normalizer, prefix=True)

    # Tests adding a second normalizer and keeping it as a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 2
    assert isinstance(model.normalizer.normalizers[0], LowercaseNormalizer)
    assert isinstance(model.normalizer.normalizers[1], ByteLevelNormalizer)

    normalizer = NFKCNormalizer()
    model = model.add_normalizer(normalizer, prefix=False)

    # Tests adding a third normalizer and keeping it as a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 3
    assert isinstance(model.normalizer.normalizers[0], LowercaseNormalizer)
    assert isinstance(model.normalizer.normalizers[1], ByteLevelNormalizer)
    assert isinstance(model.normalizer.normalizers[2], NFKCNormalizer)


def test_add_processor(small_tokenizer: Tokenizer) -> None:
    """Test adding a post-processor to the TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    post_processor = ByteLevelPostProcessor(add_prefix_space=True, trim_offsets=True, use_regex=True)
    model = model.add_post_processor(post_processor)

    # Tests removing None and adding a post-processor
    assert model.post_processor is not None
    assert isinstance(model.post_processor, ByteLevelPostProcessor)

    model = model.add_post_processor(post_processor)

    # Tests adding a second post-processor and turning it into a sequence
    assert isinstance(model.post_processor, PostProcessorSequence)
    assert len(model.post_processor.processors) == 2

    model = model.add_post_processor(post_processor)

    # Tests adding a third post-processor and keeping it as a sequence
    assert isinstance(model.post_processor, PostProcessorSequence)
    assert len(model.post_processor.processors) == 3

    call_tokenizer(model)


def test_from_pretrained(small_tokenizer: Tokenizer) -> None:
    """Test creating a TokenizerModel from a pretrained tokenizer."""
    with pytest.raises(FileNotFoundError), TemporaryDirectory() as tempdir:
        model = TokenizerModel.from_pretrained(tempdir)

    with TemporaryDirectory() as temp_dir:
        small_tokenizer.save(f"{temp_dir}/tokenizer.json")

        model = TokenizerModel.from_pretrained(f"{temp_dir}/tokenizer.json")

        assert model.version == "1.0"
        assert model.model is not None
        assert isinstance(model.model, WordPiece)
        assert isinstance(model.added_tokens, AddedTokens)
        assert model.normalizer is None
        assert model.pre_tokenizer is None
        assert model.post_processor is None
        assert model.decoder is None

        model = TokenizerModel.from_pretrained(temp_dir)

        assert model.version == "1.0"
        assert model.model is not None
        assert isinstance(model.model, WordPiece)
        assert isinstance(model.added_tokens, AddedTokens)
        assert model.normalizer is None
        assert model.pre_tokenizer is None
        assert model.post_processor is None
        assert model.decoder is None


def test_from_pretrained_falls_back_when_transformers_loading_fails(
    small_tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that from_pretrained loads tokenizer.json directly when the transformers path fails."""

    def _raise(cls: type[TokenizerModel], path: Any) -> TokenizerModel:
        raise ValueError("simulated transformers loading failure")

    monkeypatch.setattr(TokenizerModel, "from_transformers", classmethod(_raise))

    with TemporaryDirectory() as temp_dir:
        small_tokenizer.save(f"{temp_dir}/tokenizer.json")

        model = TokenizerModel.from_pretrained(temp_dir)

        assert model.version == "1.0"
        assert isinstance(model.model, WordPiece)
        assert isinstance(model.added_tokens, AddedTokens)


def test_make_greedy(small_tokenizer: Tokenizer) -> None:
    """Test whether the make greedy function works."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_pre_tokenizer(ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True))
    model = model.make_model_greedy()
    assert model.model.type == ModelType.WORDPIECE
    call_tokenizer(model)
    pre_tok_length = None
    if model.pre_tokenizer is not None:
        if isinstance(model.pre_tokenizer, PreTokenizerSequence):
            for pt in model.pre_tokenizer.pretokenizers:
                if isinstance(pt, FixedLengthPreTokenizer):
                    pre_tok_length = pt.length
        elif isinstance(model.pre_tokenizer, FixedLengthPreTokenizer):
            pre_tok_length = model.pre_tokenizer.length
    assert pre_tok_length is not None
    assert pre_tok_length == model.model.max_input_chars_per_word


def test_lowercase(small_tokenizer: Tokenizer) -> None:
    """Tests whether the model performs the lowercase test correctly."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not model.lowercases_input
    model.normalizer = LowercaseNormalizer()
    assert model.lowercases_input
    assert model.lowercases_input == model.normalizer._lowercases

    call_tokenizer(model)


def test_byte_normalizes(small_tokenizer: Tokenizer) -> None:
    """Tests whether the model performs byte normalization correctly."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not model.transforms_into_bytes
    model.normalizer = ByteLevelNormalizer()
    assert model.transforms_into_bytes
    model.normalizer = None
    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=False)
    assert model.transforms_into_bytes

    call_tokenizer(model)


def test_remove_token(small_tokenizer: Tokenizer) -> None:
    """Test removing a token from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.remove_token_from_vocabulary("a")
    with pytest.raises(ValueError):
        model = model.remove_token_from_vocabulary("a")
    assert "a" not in model.model.vocab.vocabulary

    call_tokenizer(model)


def test_add_token(small_tokenizer: Tokenizer) -> None:
    """Test adding a token to the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        model = model.add_token_to_vocabulary("a")
    model = model.add_token_to_vocabulary("new_token")
    assert "new_token" in model.model.vocab.vocabulary

    call_tokenizer(model)


def test_replace_token(small_tokenizer: Tokenizer) -> None:
    """Test replace token interface."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        model = model.replace_token_in_vocabulary("a", "b")
    model = model.replace_token_in_vocabulary("b", "x")
    assert "b" not in model.model.vocab.vocabulary
    assert "x" in model.model.vocab.vocabulary

    call_tokenizer(model)


def test_replace_unk_and_pad_token(small_tokenizer: Tokenizer) -> None:
    """Test replace token interface."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.unk_token = "a"
    model = model.replace_token_in_vocabulary("a", "x")
    assert model.unk_token == "x"

    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pad_token = "a"
    model = model.replace_token_in_vocabulary("a", "x")
    assert model.pad_token == "x"

    call_tokenizer(model)


def test_replace_token_template(small_tokenizer: Tokenizer) -> None:
    """Test replace token interface."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    t = SpecialTokenInfo(id="bos", ids=[1], tokens=["a"])
    s = {"bos": t}
    tok = Token(id="bos", type_id=0, type=TokenType.SPECIAL)
    seq = Token(id="A", type_id=0, type=TokenType.SEQUENCE)
    seq2 = Token(id="B", type_id=1, type=TokenType.SEQUENCE)
    sequence = TokenSequence((tok, seq, tok))
    pair_seq = TokenSequence((tok, seq, tok, seq2, tok))
    model.post_processor = TemplatePostProcessor(special_tokens=s, single=sequence, pair=pair_seq)
    with pytest.raises(ValueError):
        model = model.replace_token_in_vocabulary("a", "b")
    model = model.replace_token_in_vocabulary("b", "x")
    assert "b" not in model.model.vocab.vocabulary
    assert "x" in model.model.vocab.vocabulary

    call_tokenizer(model)


def test_decase_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test the decasing of the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.added_tokens = AddedTokens([])
    model = model.decase_vocabulary()
    # This tokenizer does not assign any special tokens, so this is true.
    if isinstance(model.normalizer, NormalizerSequence):
        assert any(isinstance(norm, LowercaseNormalizer) for norm in model.normalizer.normalizers)
    else:
        assert isinstance(model.normalizer, LowercaseNormalizer)
        assert model.model.vocab.sorted_vocabulary == [
            "[pad]",
            "[sep]",
            "[unk]",
            "[cls]",
            "[mask]",
            "a",
            "b",
            "c",
            "d",
            " ",
            "f",
        ]
    call_tokenizer(model)


def test_decase_vocabulary_with_added_token(small_tokenizer: Tokenizer) -> None:
    """Test the decasing of the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_addedtoken("ADD", is_special=False, normalized=True)
    model = model.add_addedtoken("ADD_KEEP", is_special=False, normalized=False)
    model = model.decase_vocabulary(keep=True)
    # This tokenizer does not assign any special tokens, so this is true.
    # Added tokens are matched against raw input before the normalizer/pretokenizer ever run,
    # so their content stays unchanged regardless of `normalized`.
    assert model.model.vocab.sorted_vocabulary == [
        "[pad]",
        "[sep]",
        "[UNK]",
        "[cls]",
        "[mask]",
        "a",
        "b",
        "c",
        "d",
        " ",
        "F",
        "ADD",
        "ADD_KEEP",
    ]

    call_tokenizer(model)


def test_remove_uppercase_with_added_token(small_tokenizer: Tokenizer) -> None:
    """Test the decasing of the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_addedtoken("ADD", is_special=False, normalized=True)
    model = model.add_addedtoken("ADD_KEEP", is_special=False, normalized=False)
    model = model.decase_vocabulary()
    assert model.model.vocab.sorted_vocabulary == [
        "[pad]",
        "[sep]",
        "[UNK]",
        "[cls]",
        "[mask]",
        "a",
        "b",
        "c",
        "d",
        " ",
        "F",
        "ADD",
        "ADD_KEEP",
    ]

    call_tokenizer(model)


def test_eos(small_tokenizer: Tokenizer) -> None:
    """Test getting the eos token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.eos is None
    model.post_processor = ByteLevelPostProcessor(add_prefix_space=True, trim_offsets=True, use_regex=True)
    assert model.eos is None
    model.post_processor = None
    model.post_processor = RobertaPostProcessor(
        sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False
    )
    assert model.eos == ["[SEP]"]
    model.post_processor = None
    model.post_processor = BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0))
    assert model.eos == ["[SEP]"]
    model.post_processor = PostProcessorSequence(
        processors=[
            RobertaPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False),
            BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0)),
        ]
    )
    assert model.eos is None

    call_tokenizer(model)


def test_bos(small_tokenizer: Tokenizer) -> None:
    """Test getting the eos token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.bos is None
    model.post_processor = ByteLevelPostProcessor(add_prefix_space=True, trim_offsets=True, use_regex=True)
    assert model.bos is None
    model.post_processor = None
    model.post_processor = RobertaPostProcessor(
        sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False
    )
    assert model.bos == ["[CLS]"]
    model.post_processor = None
    model.post_processor = BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0))
    assert model.bos == ["[CLS]"]
    model.post_processor = PostProcessorSequence(
        processors=[
            RobertaPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False),
            BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0)),
        ]
    )
    assert model.bos is None

    call_tokenizer(model)


def test_split(small_tokenizer: Tokenizer) -> None:
    """Test whether the split works correctly."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not model.splits
    pretokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=False, trim_offsets=True)
    model = model.add_pre_tokenizer(pretokenizer)
    assert not model.splits
    pretokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    model = model.add_pre_tokenizer(pretokenizer)
    assert model.splits

    call_tokenizer(model)


def test_subword_prefix(small_tokenizer: Tokenizer) -> None:
    """Test getting the subword prefix token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.subword_prefix == "##"

    call_tokenizer(model)


def test_word_prefix(small_tokenizer: Tokenizer) -> None:
    """Test getting the word prefix token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.word_prefix is None

    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    assert model.word_prefix == "Ġ"

    model.pre_tokenizer = MetaspacePreTokenizer(replacement="▁", split=True, prepend_scheme=PrependScheme.ALWAYS)
    assert model.word_prefix == "▁"

    call_tokenizer(model)


def test_replace_token_in_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test replacing a token in the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.replace_token_in_vocabulary("F", "G")
    assert model.model.vocab["G"] is not None
    assert model.added_tokens.get_token("G") is not None

    call_tokenizer(model)


def test_remove_token_from_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test removing a token from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.added_tokens.get_token("F") is not None
    model = model.remove_token_from_vocabulary("F")
    assert model.added_tokens.get_token("F") is None

    call_tokenizer(model)


def test_set_unk_token(small_tokenizer: Tokenizer) -> None:
    """Test setting the unknown token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.unk_token = "new_unk"
    assert model.model.vocab["new_unk"] is not None
    assert model.added_tokens.get_token("new_unk") is not None

    with pytest.raises(ValueError):
        model.unk_token = None

    assert isinstance(model.model, WordPiece)
    model.model = BPE(
        vocab=model.model.vocab,
        merges=Merges([]),
        dropout=0.0,
        unk_token="a",
        continuing_subword_prefix="",
        end_of_word_suffix="",
        fuse_unk=True,
        byte_fallback=False,
        ignore_merges=False,
    )
    model.unk_token = None

    model.unk_token = "a"
    model.unk_token = "b"
    model.unk_token = "a"

    model.unk_token = None
    model.unk_token = "OSTENTATIOUS"
    call_tokenizer(model)


def test_set_unk_token_template(small_tokenizer: Tokenizer) -> None:
    """Test setting the unknown token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    t = SpecialTokenInfo(id="bos", ids=[1], tokens=["a"])
    s = {"bos": t}
    tok = Token(id="bos", type_id=1, type=TokenType.SPECIAL)
    seq = Token(id="A", type_id=0, type=TokenType.SEQUENCE)
    seq2 = Token(id="B", type_id=0, type=TokenType.SEQUENCE)
    sequence = TokenSequence((tok, seq, tok))
    pair_seq = TokenSequence((tok, seq, tok, seq2, tok))
    model.post_processor = TemplatePostProcessor(special_tokens=s, single=sequence, pair=pair_seq)
    model.unk_token = "a"
    model.unk_token = "b"

    assert isinstance(model.post_processor, TemplatePostProcessor)
    assert model.post_processor.special_tokens["bos"].tokens == ["b"]

    call_tokenizer(model)


def test_get_padding_token(small_tokenizer: Tokenizer) -> None:
    """Get the padding token from the tokenizer model."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.padding is None
    assert model.pad_token is None

    model.padding = Padding(pad_token="[PAD]", pad_id=3, pad_type_id=0)
    assert model.pad_token == "[PAD]"

    call_tokenizer(model)


def test_set_padding_token(small_tokenizer: Tokenizer) -> None:
    """Set the padding token for the tokenizer model."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pad_token = "[PAD]"
    assert model.added_tokens.get_token("[PAD]") is not None
    model.pad_token = None
    model.pad_token = "OSTENTATIOUS"
    assert model.added_tokens.get_token("OSTENTATIOUS") is not None
    assert model.added_tokens.get_token("[PAD]") is not None
    model.added_tokens = AddedTokens([])
    model.pad_token = "OSTENTATIOUS"
    assert model.added_tokens.get_token("OSTENTATIOUS") is not None
    model.pad_token = "OSTENTATIOUS"
    assert model.added_tokens.get_token("OSTENTATIOUS") is not None
    model.pad_token = "FUN"
    assert model.added_tokens.get_token("FUN") is not None

    call_tokenizer(model)


def test_from_transformers(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.model, WordPiece)
    assert isinstance(model.added_tokens, AddedTokens)

    call_tokenizer(model)


def test_from_transformers_missing_unk(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    transformers_tokenizer.SPECIAL_TOKENS_ATTRIBUTES.remove("unk_token")
    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.added_tokens, AddedTokens)

    call_tokenizer(model)


def test_from_transformers_missing_pad(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    transformers_tokenizer.SPECIAL_TOKENS_ATTRIBUTES.remove("pad_token")
    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.model, WordPiece)
    assert isinstance(model.added_tokens, AddedTokens)

    call_tokenizer(model)


def test_missing_unk_model(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    transformers_tokenizer.backend_tokenizer.model = TokenizersBPE(
        vocab={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, " ": 6}, merges=[], unk_token=None
    )  # type: ignore
    transformers_tokenizer.backend_tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.added_tokens, AddedTokens)

    call_tokenizer(model)


def test_deep_copy(small_tokenizer: Tokenizer) -> None:
    """Test that deep copying doesn't crash."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_normalizer(ReplaceNormalizer(pattern="hello", content="yello"))
    prep = model.preprocessor
    assert prep.normalizer is not None
    # This should not crash. It used to crash before
    copied_model = model.deep_copy()
    assert model._preprocessor is None
    assert copied_model._preprocessor is None


def test_model_delta(small_tokenizer: Tokenizer) -> None:
    """Test the model delta functionality."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_token_to_vocabulary("new_token")
    model = model.remove_token_from_vocabulary("a")
    model = model.replace_token_in_vocabulary("b", "x")
    model = model.add_addedtoken("[ADDED]")
    model.unk_token = "new_unk"
    model.pad_token = "[PAD]"
    delta = model.model_delta
    assert delta.token_mapping == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 7, 7: 8, 8: 9, 9: 10}
    assert delta.new_vocabulary_size == 13
    assert delta.new_tokens == {"x": 5, "[ADDED]": 11, "new_token": 10, "new_unk": 12}


def test_model_same_unk_and_pad(small_tokenizer: Tokenizer) -> None:
    """Test the model delta functionality."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.add_pre_tokenizer(
        SplitPreTokenizer(pattern=StringPattern(String=" "), behavior=Behavior.ISOLATED, invert=False)
    )
    model = model.make_model_greedy()
    model.unk_token = "[UNK]"
    unk = model.added_tokens.get_token("[UNK]")
    length_tokens = len(model.added_tokens)
    assert unk is not None
    unk_id = unk.id
    model.pad_token = "[UNK]"
    pad = model.added_tokens.get_token("[UNK]")
    assert pad is not None
    pad_id = pad.id
    assert unk_id == pad_id
    assert len(model.added_tokens) == length_tokens

    assert model.unk_token == "[UNK]"
    assert model.pad_token == "[UNK]"

    tokenizer = model.to_tokenizer()
    encoded = tokenizer.encode("abcdeF")
    assert encoded.tokens == ["[UNK]", "F"]
    assert encoded.ids == [2, 10]


def test_token_to_id_and_id_to_token(small_tokenizer: Tokenizer) -> None:
    """Test the tokens_to_ids and ids_to_tokens methods."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    tokens = ["a", "b", "c", "D", "F"]
    ids = model.tokens_to_ids(tokens)
    assert ids == [5, 6, 7, 8, 10]
    recovered_tokens = model.ids_to_tokens(ids)
    assert recovered_tokens == tokens

    tokenizer = model.to_tokenizer()
    encoded = tokenizer.encode_batch(["a", "b", "c", "D", "F"])
    assert [enc.tokens[0] for enc in encoded] == tokens
    assert [enc.ids[0] for enc in encoded] == ids


def test_id_to_token_with_vocabulary_gap(small_tokenizer_json: dict[str, Any]) -> None:
    """`ids_to_tokens` must resolve tokens by their real ID even if the vocabulary has a gap.

    Positional lookups (indexing into a list sorted by ID) silently return the wrong
    token, or raise an IndexError, whenever an ID is missing below the queried ID.
    """
    small_tokenizer_json["model"]["vocab"] = {
        "[PAD]": 0,
        "[SEP]": 1,
        "[UNK]": 2,
        "[CLS]": 3,
        "[MASK]": 4,
        "a": 5,
        "b": 7,
        "c": 8,
        "D": 9,
        " ": 10,
    }
    small_tokenizer_json["added_tokens"] = [t for t in small_tokenizer_json["added_tokens"] if t["content"] != "F"]
    model = TokenizerModel.model_validate(small_tokenizer_json)

    assert model.tokens_to_ids(["b"]) == [7]
    assert model.ids_to_tokens([7]) == ["b"]
    assert model.ids_to_tokens([10]) == [" "]


def test_to_transformers(small_tokenizer: Tokenizer) -> None:
    """Test saving and loading a tokenizer model."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    transformers_tokenizer = model.to_transformers()
    assert transformers_tokenizer.pad_token is None
    assert transformers_tokenizer.unk_token == "[UNK]"

    model.pad_token = "[PAD]"
    transformers_tokenizer = model.to_transformers()
    assert transformers_tokenizer.pad_token == "[PAD]"

    model.post_processor = RobertaPostProcessor(
        sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False
    )
    assert model.eos == ["[SEP]"]
    assert model.bos == ["[CLS]"]
    transformers_tokenizer = model.to_transformers()
    assert transformers_tokenizer.eos_token == "[SEP]"
    assert transformers_tokenizer.bos_token == "[CLS]"

    model.post_processor = TemplatePostProcessor(
        special_tokens={
            "sep": SpecialTokenInfo(id="sep", ids=[1, 2], tokens=["[SEP]", "</s>"]),
            "cls": SpecialTokenInfo(id="cls", ids=[0, 1], tokens=["[CLS]", "<s>"]),
        },
        single=TokenSequence(
            (
                Token(id="cls", type_id=0, type=TokenType.SPECIAL),
                Token(id="A", type_id=0, type=TokenType.SEQUENCE),
                Token(id="sep", type_id=1, type=TokenType.SPECIAL),
            )
        ),
        pair=TokenSequence(
            (
                Token(id="cls", type_id=0, type=TokenType.SPECIAL),
                Token(id="A", type_id=0, type=TokenType.SEQUENCE),
                Token(id="sep", type_id=1, type=TokenType.SPECIAL),
                Token(id="B", type_id=1, type=TokenType.SEQUENCE),
                Token(id="sep", type_id=1, type=TokenType.SPECIAL),
            )
        ),
    )
    assert model.eos == ["[SEP]", "</s>"]
    assert model.bos == ["[CLS]", "<s>"]

    transformers_tokenizer = model.to_transformers()
    assert transformers_tokenizer.eos_token is None
    assert transformers_tokenizer.bos_token is None


def test_to_transformers_to_class(small_tokenizer: Tokenizer) -> None:
    """Test saving and loading a tokenizer model."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    transformers_tokenizer = model.to_transformers(tokenizer_class=PreTrainedTokenizerFast)
    assert isinstance(transformers_tokenizer, PreTrainedTokenizerFast)

    transformers_tokenizer = model.to_transformers(tokenizer_class=GPT2Tokenizer)
    assert isinstance(transformers_tokenizer, GPT2Tokenizer)

    model._original_class = GPT2Tokenizer
    transformers_tokenizer = model.to_transformers()
    assert isinstance(transformers_tokenizer, GPT2Tokenizer)


def test_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test the vocabulary property."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    vocab = model.vocabulary
    assert isinstance(vocab, dict)
    assert len(vocab) == len(model.model.vocab)
    assert model.vocabulary_size == len(model.model.vocab)
    assert model.vocabulary == model.model.vocab.vocabulary
    assert model.sorted_vocabulary == model.model.vocab.sorted_vocabulary

    tokenizer = model.to_tokenizer()
    assert len(tokenizer.get_vocab()) == len(model.model.vocab)
    assert tokenizer.get_vocab() == model.model.vocab.vocabulary


def test_batch_remove_tokens(small_tokenizer: Tokenizer) -> None:
    """Test removing multiple tokens from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    tokens_to_remove = ["a", "b", "c"]
    model = model.remove_tokens_from_vocabulary(tokens_to_remove)
    for token in tokens_to_remove:
        assert token not in model.model.vocab.vocabulary

    call_tokenizer(model)


def _model_with_hardcoded_ids(small_tokenizer_json: dict[str, Any]) -> TokenizerModel:
    """Build a model that hard-codes token ids in its padding and post-processor."""
    model = TokenizerModel.from_string(json.dumps(small_tokenizer_json))
    model.pad_token = "[MASK]"
    return model.add_post_processor(TemplatePostProcessor.from_tokens(None, ("F", model.vocabulary["F"])))


def test_remove_tokens_remaps_hardcoded_ids(small_tokenizer_json: dict[str, Any]) -> None:
    """Test that removing tokens moves the ids hard-coded outside the vocabulary."""
    model = _model_with_hardcoded_ids(small_tokenizer_json)
    assert model.vocabulary["F"] == 10
    assert model.pad_token_id == 4

    model = model.remove_tokens_from_vocabulary(["[PAD]", "[SEP]"])

    # Both tokens sat below "[MASK]" and "F", so both ids shift down by two.
    assert model.vocabulary["F"] == 8
    assert model.pad_token_id == 2
    assert isinstance(model.post_processor, TemplatePostProcessor)
    assert model.post_processor.special_tokens["F"].ids == [8]
    assert_vocabulary_consistent(model)

    # The symptom this guards against: an id pointing past the end of the vocabulary.
    assert max(model.to_tokenizer().encode("a b c").ids) < model.vocabulary_size

    call_tokenizer(model)


def test_remove_tokens_remaps_hardcoded_ids_when_chained(small_tokenizer_json: dict[str, Any]) -> None:
    """Test that removing tokens one at a time gives the same ids as removing them at once."""
    model = _model_with_hardcoded_ids(small_tokenizer_json)

    at_once = model.remove_tokens_from_vocabulary(["[PAD]", "[SEP]"])
    chained = model.remove_token_from_vocabulary("[PAD]").remove_token_from_vocabulary("[SEP]")

    assert chained.vocabulary == at_once.vocabulary
    assert chained.added_tokens.root == at_once.added_tokens.root
    assert chained.pad_token_id == at_once.pad_token_id
    assert chained.post_processor == at_once.post_processor
    assert_vocabulary_consistent(chained)

    call_tokenizer(chained)


def test_remove_uppercase(small_tokenizer: Tokenizer) -> None:
    """Test the removal of uppercase tokens from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.decase_vocabulary(keep=True)
    assert model.sorted_vocabulary == ["[pad]", "[sep]", "[UNK]", "[cls]", "[mask]", "a", "b", "c", "d", " ", "F"]

    call_tokenizer(model)


def test_remove_uppercaser(small_tokenizer: Tokenizer) -> None:
    """Test the removal of uppercase tokens from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_token_to_vocabulary("apa")
    model = model.add_normalizer(ReplaceNormalizer(pattern="a", content="G"))
    model = model.consolidate_vocabulary(keep=True)
    assert model.sorted_vocabulary == [
        "[PAD]",
        "[SEP]",
        "[UNK]",
        "[CLS]",
        "[MASK]",
        "G",
        "b",
        "c",
        "D",
        " ",
        "F",
        "GpG",
    ]

    call_tokenizer(model)


def test_prune(small_tokenizer: Tokenizer) -> None:
    """Test the removal of uppercase tokens from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_token_to_vocabulary("apa")
    model = model.add_normalizer(ReplaceNormalizer(pattern="a", content="G"))
    model = model.consolidate_vocabulary()
    assert model.sorted_vocabulary == [
        "[PAD]",
        "[SEP]",
        "[UNK]",
        "[CLS]",
        "[MASK]",
        "G",
        "b",
        "c",
        "D",
        " ",
        "F",
        "GpG",
    ]

    call_tokenizer(model)


def test_remap_added_token_ids(small_tokenizer: Tokenizer) -> None:
    """Test remapping an added token to a different token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_addedtoken("[NEW_TOKEN]")
    new_token = model.added_tokens.get_token("[NEW_TOKEN]")
    assert new_token is not None
    new_token_id = new_token.id
    assert new_token_id is not None
    assert model.model.vocab["[NEW_TOKEN]"] == new_token_id
    model.post_processor = TemplatePostProcessor(
        special_tokens={
            "bos": SpecialTokenInfo(id="bos", ids=[3], tokens=["[NEW_TOKEN]"]),
        },
        single=TokenSequence(
            (
                Token(id="bos", type_id=0, type=TokenType.SPECIAL),
                Token(id="A", type_id=0, type=TokenType.SEQUENCE),
            )
        ),
        pair=TokenSequence(
            (
                Token(id="bos", type_id=0, type=TokenType.SPECIAL),
                Token(id="A", type_id=0, type=TokenType.SEQUENCE),
                Token(id="B", type_id=1, type=TokenType.SEQUENCE),
            )
        ),
    )
    model._remap_added_token_ids()
    assert model.post_processor.special_tokens["bos"].ids == [new_token.id]

    call_tokenizer(model)


def test_get_pad_token_id(small_tokenizer: Tokenizer) -> None:
    """Test getting the padding token ID from the tokenizer model."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.pad_token_id is None

    model.pad_token = "[PAD_IN_DE_KORF]"
    assert model.pad_token_id == 11

    call_tokenizer(model)


def test_get_unk_token_id(small_tokenizer: Tokenizer) -> None:
    """Test getting the unknown token ID from the tokenizer model."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.model = BPE(
        vocab=model.model.vocab,  # type: ignore
        merges=Merges([]),
        dropout=0.0,
        unk_token="[UNK]",
        continuing_subword_prefix="",
        end_of_word_suffix="",
        fuse_unk=True,
        byte_fallback=False,
        ignore_merges=False,
    )
    assert model.unk_token_id == 2

    model.unk_token = "[UNKY]"
    assert model.unk_token_id == 11

    model.unk_token = None
    assert model.unk_token_id is None

    call_tokenizer(model)


def test_add_pad_token_post_init(small_tokenizer_json: dict[str, Any]) -> None:
    """Test adding a pad token after initialization from a file."""
    small_tokenizer_json["padding"] = {
        "strategy": {"Fixed": 0},
        "direction": "Right",
        "pad_to_multiple_of": None,
        "pad_id": 40000,
        "pad_type_id": 0,
        "pad_token": "[ZAAAA]",
    }
    model = TokenizerModel.model_validate(small_tokenizer_json)
    assert model.pad_token == "[ZAAAA]"
    assert model.pad_token_id == 11
    added_token = model.added_tokens.get_token("[ZAAAA]")
    assert added_token is not None
    assert model.pad_token_id == added_token.id


def test_add_pad_token_post_init_overlap(small_tokenizer_json: dict[str, Any]) -> None:
    """Test adding a pad token after initialization from a file."""
    small_tokenizer_json["padding"] = {
        "strategy": {"Fixed": 0},
        "direction": "Right",
        "pad_to_multiple_of": None,
        "pad_id": 5,
        "pad_type_id": 0,
        "pad_token": "[ZAAAA]",
    }
    model = TokenizerModel.model_validate(small_tokenizer_json)
    assert model.pad_token == "[ZAAAA]"
    assert model.pad_token_id == 11
    added_token = model.added_tokens.get_token("[ZAAAA]")
    assert added_token is not None
    assert model.pad_token_id == added_token.id


def test_preprocessor_property(small_tokenizer_json: dict[str, Any]) -> None:
    """Test the preprocessor property."""
    model = TokenizerModel.model_validate(small_tokenizer_json)
    assert model._preprocessor is None
    preprocessor = model.preprocessor
    assert model._preprocessor is not None
    assert model._preprocessor is preprocessor

    assert preprocessor.normalizer is None
    assert preprocessor.pretokenizer is None

    model = model.add_pre_tokenizer(WhitespacePreTokenizer())
    assert model._preprocessor is None
    new_preprocessor = model.preprocessor
    assert model._preprocessor is not None
    assert model._preprocessor is new_preprocessor

    assert new_preprocessor.normalizer is None
    assert new_preprocessor.pretokenizer is not None

    model = model.add_normalizer(LowercaseNormalizer())
    assert model._preprocessor is None
    newer_preprocessor = model.preprocessor
    assert model._preprocessor is not None
    assert model._preprocessor is newer_preprocessor

    assert newer_preprocessor.normalizer is not None
    assert newer_preprocessor.pretokenizer is not None


def test_preprocess_token(small_tokenizer_json: dict[str, Any]) -> None:
    """Test whether the token preprocessor works."""
    model = TokenizerModel.model_validate(small_tokenizer_json)
    assert model._preprocessor is None
    assert model._preprocess_token("hello") == "hello"
    assert model._preprocessor is not None


def test_preprocess_token_is_initial(small_tokenizer_json: dict[str, Any]) -> None:
    """Test that is_initial=False produces a continuation-piece form (e.g. "##hello" for WordPiece)."""
    model = TokenizerModel.model_validate(small_tokenizer_json)
    assert model.subword_prefix == "##"
    # Default (is_initial=True): a plain, word-initial piece.
    assert model._preprocess_token("hello") == "hello"
    # is_initial=False: the continuation marker is added instead.
    assert model._preprocess_token("hello", is_initial=False) == "##hello"


def test_add_token_to_vocabulary_is_initial(small_tokenizer: Tokenizer) -> None:
    """Test that add_token_to_vocabulary(is_initial=False) adds a usable mid-word continuation piece."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_token_to_vocabulary("skeletonize", is_initial=False)
    assert "##skeletonize" in model.vocabulary
    assert "skeletonize" not in model.vocabulary
    call_tokenizer(model)


def test_preprocess_token_is_initial_byte_level_add_prefix_space() -> None:
    """Test is_initial on byte-level BPE with add_prefix_space=True, where "Ġ" IS deterministic."""
    bpe = BPE(
        vocab=Vocabulary({c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyzĠ")}),
        merges=Merges([]),
        dropout=None,
        unk_token=None,
        continuing_subword_prefix="",
        end_of_word_suffix=None,
        fuse_unk=False,
        byte_fallback=False,
        ignore_merges=False,
    )
    model = TokenizerModel(model=bpe)
    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    assert model.transforms_into_bytes
    assert model.adds_prefix_space is True

    assert model._preprocess_token("skeletonize", is_initial=True) == "Ġskeletonize"
    assert model._preprocess_token("skeletonize", is_initial=False) == "skeletonize"


def test_preprocess_token_with_pretokenizer(small_tokenizer_json: dict[str, Any]) -> None:
    """Test whether the token preprocessor works."""
    model = TokenizerModel.model_validate(small_tokenizer_json)
    model = model.add_pre_tokenizer(WhitespacePreTokenizer())
    assert model._preprocessor is None

    with pytest.raises(ValueError):
        model._preprocess_token("hello    beepboop")
    with pytest.raises(ValueError):
        model._preprocess_token("   ")

    assert model._preprocessor is not None


def test_replace_tokens_in_vocabulary(small_tokenizer_json: dict[str, Any]) -> None:
    """Test replacement length issues."""
    model = TokenizerModel.model_validate(small_tokenizer_json)
    with pytest.raises(ValueError):
        model.replace_tokens_in_vocabulary(["a", "b"], ["a"])


def test_bos_ids(small_tokenizer_json: dict[str, Any]) -> None:
    """Test replacement length issues."""
    model = TokenizerModel.model_validate(small_tokenizer_json)

    assert model.eos is None
    assert model.bos is None
    assert model.eos_ids is None
    assert model.bos_ids is None

    post = BertPostProcessor(sep=("a", 0), cls=("b", 1))
    model = model.add_post_processor(post)

    assert model.eos == ["a"]
    assert model.bos == ["b"]
    assert model.eos_ids == [5]
    assert model.bos_ids == [6]


def test_adds_pretokenizer(small_tokenizer_json: dict[str, Any]) -> None:
    """Test whether the add pretokenizer is correct."""
    model = TokenizerModel.model_validate(small_tokenizer_json)

    assert model.adds_prefix_space is None

    with pytest.raises(ValueError):
        model.adds_prefix_space = True
    bert = BertPreTokenizer()
    model = model.add_pre_tokenizer(bert)
    with pytest.raises(ValueError):
        model.adds_prefix_space = True

    assert model.adds_prefix_space is None
    byt = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=True, trim_offsets=True)
    model = model.add_pre_tokenizer(byt)
    assert not model.adds_prefix_space
    model.adds_prefix_space = True
    assert model.adds_prefix_space
    # Done in place.
    assert byt.add_prefix_space


def test_load_with_added(small_tokenizer_json: dict[str, Any]) -> None:
    """Tests whether added tokens are added to the vocabulary when loading."""
    json_data = json.dumps(small_tokenizer_json)
    model = TokenizerModel.from_string(json_data)
    assert "F" in model.sorted_vocabulary


def test_prune_added_tokens(small_tokenizer_json: dict[str, Any]) -> None:
    """Tests whether added tokens get pruned."""
    json_data = json.dumps(small_tokenizer_json)
    model = TokenizerModel.from_string(json_data)
    assert [x.content for x in model.added_tokens.root] == ["F", "[UNK]"]
    # Only tokens that aren't special tokens should get pruned.
    model = model.prune_added_tokens()
    assert [x.content for x in model.added_tokens.root] == ["[UNK]"]

    model.pad_token = "jello"
    assert [x.content for x in model.added_tokens.root] == ["[UNK]", "jello"]
    model = model.prune_added_tokens()
    assert [x.content for x in model.added_tokens.root] == ["[UNK]", "jello"]

    model = model.add_post_processor(
        RobertaPostProcessor(sep=("[B]", 0), cls=("[A]", 1), trim_offsets=True, add_prefix_space=True)
    )
    assert [x.content for x in model.added_tokens.root] == ["[UNK]", "jello", "[B]", "[A]"]
    model = model.prune_added_tokens()

    assert [x.content for x in model.added_tokens.root] == ["[UNK]", "jello", "[B]", "[A]"]


def test_prune_added_tokens_post_processor(
    small_tokenizer_json: dict[str, Any], template_post_processor: TemplatePostProcessor
) -> None:
    """Test that pruning keeps the post processor intact."""
    json_data = json.dumps(small_tokenizer_json)
    model = TokenizerModel.from_string(json_data)

    assert [(x.content, x.id) for x in model.added_tokens.root] == [("F", 10), ("[UNK]", 2)]

    for token in template_post_processor.special_tokens.values():
        for t in token.tokens:
            if t in model.vocabulary:
                model._turn_into_addedtoken(t)
            else:
                model.add_addedtoken(t)

    model = model.add_post_processor(template_post_processor)
    assert [(x.content, x.id) for x in model.added_tokens.root] == [
        ("F", 10),
        ("[UNK]", 2),
        ("[CLS]", 3),
        ("[MASK]", 4),
        ("[PAD]", 0),
        ("[SEP]", 1),
    ]

    model = model.prune_added_tokens()
    assert [(x.content, x.id) for x in model.added_tokens.root] == [("[UNK]", 1), ("[CLS]", 2), ("[SEP]", 0)]


def test_remove_token_unk_token(small_tokenizer_json: dict[str, Any]) -> None:
    """Test whether removing a token removes the unk token."""
    json_data = json.dumps(small_tokenizer_json)
    model = TokenizerModel.from_string(json_data)

    assert model.unk_token == "[UNK]"
    model = model.remove_token_from_vocabulary(model.unk_token)
    assert model.unk_token == "[UNK]"
    assert model.unk_token in model.vocabulary

    pad = "[OPAD]"

    assert pad not in model.vocabulary
    assert model.pad_token is None
    model.pad_token = pad
    assert pad in model.vocabulary
    assert model.pad_token == pad
    model = model.remove_token_from_vocabulary(pad)
    assert model.pad_token is None
    assert pad not in model.vocabulary


def test_consolidate_with_post_processor_and_none_token(small_tokenizer: Tokenizer) -> None:
    """Test _consolidate covers the post_processor update and None-token continue paths."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    # 'A' after decasing collides with existing 'a' → maps to None (keep=False), covering the continue branch
    model = model.add_token_to_vocabulary("A")
    model.post_processor = BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 3))
    model = model.decase_vocabulary(keep=False)
    assert "A" not in model.sorted_vocabulary
    assert "a" in model.sorted_vocabulary
    assert model.post_processor is not None
    call_tokenizer(model)


def test_consolidate_pad_token_update(small_tokenizer: Tokenizer) -> None:
    """Test _consolidate exercises the pad_token update path (lines 461 and 468).

    The pad_token setter adds 'D' to added_tokens as normalized=True, so _process returns it
    unchanged. Lines 461 and 468 are still reached: pad_token_update is set and re-applied.
    """
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pad_token = "D"
    assert model.pad_token == "D"
    model = model.decase_vocabulary()
    # 'D' is a special normalized added token, so it is not renamed during consolidation
    assert model.pad_token == "D"
    call_tokenizer(model)


def test_set_subword_prefix(small_tokenizer: Tokenizer) -> None:
    """Test that setting subword_prefix re-encodes the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.subword_prefix == "##"
    model.subword_prefix = "@@"
    assert model.subword_prefix == "@@"
    call_tokenizer(model)


def test_set_word_prefix_no_existing_pretokenizer(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix with no pretokenizer inserts a non-splitting Metaspace."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.word_prefix is None
    model.word_prefix = "▁"
    assert model.word_prefix == "▁"
    assert isinstance(model.pre_tokenizer, MetaspacePreTokenizer)
    assert model.pre_tokenizer.split is False
    # Bare (non-"##") vocabulary tokens are inferred to be word-initial and get the new prefix.
    assert "▁a" in model.vocabulary
    assert "a" not in model.vocabulary

    call_tokenizer(model)


def test_set_word_prefix_appends_after_existing_splitter(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix when a splitting pretokenizer exists appends Metaspace after it."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = BertPreTokenizer()
    model = model.add_token_to_vocabulary("ing", is_initial=False)
    model = model.add_token_to_vocabulary("run", is_initial=True)

    model.word_prefix = "▁"
    assert model.word_prefix == "▁"
    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert isinstance(model.pre_tokenizer.pretokenizers[0], BertPreTokenizer)
    assert isinstance(model.pre_tokenizer.pretokenizers[1], MetaspacePreTokenizer)

    # Continuation pieces keep their subword marker and don't gain a word prefix.
    assert "##ing" in model.vocabulary
    # Previously word-initial (bare) tokens now carry the new word prefix.
    assert "▁run" in model.vocabulary

    tokenizer = model.to_tokenizer()
    assert tokenizer.encode("run").tokens == ["▁run"]
    assert tokenizer.encode("runing").tokens == ["▁run", "##ing"]


def test_set_word_prefix_replaces_existing_metaspace(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix when a Metaspace pretokenizer already exists replaces its character in place."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = MetaspacePreTokenizer(replacement="▁", split=True, prepend_scheme=PrependScheme.ALWAYS)
    model = model.consolidate_vocabulary(keep=True)

    model.word_prefix = "_"
    assert model.word_prefix == "_"
    assert isinstance(model.pre_tokenizer, MetaspacePreTokenizer)
    assert model.pre_tokenizer.split is True
    assert model.pre_tokenizer.prepend_scheme == PrependScheme.ALWAYS

    call_tokenizer(model)


def test_set_word_prefix_none_removes_existing_metaspace(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix to None removes a standalone Metaspace pretokenizer entirely."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = MetaspacePreTokenizer(replacement="▁", split=True, prepend_scheme=PrependScheme.ALWAYS)
    model = model.consolidate_vocabulary(keep=True)
    assert model.word_prefix == "▁"

    model.word_prefix = None
    assert model.word_prefix is None
    assert model.pre_tokenizer is None

    call_tokenizer(model)


def test_set_word_prefix_none_removes_metaspace_from_sequence(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix to None removes Metaspace from a sequence, keeping the other pretokenizers."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = BertPreTokenizer()
    model = model.add_token_to_vocabulary("ing", is_initial=False)
    model = model.add_token_to_vocabulary("run", is_initial=True)
    model.word_prefix = "▁"
    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)

    model.word_prefix = None
    assert model.word_prefix is None
    assert isinstance(model.pre_tokenizer, BertPreTokenizer)

    call_tokenizer(model)


def test_set_word_prefix_none_is_no_op_without_existing_pretokenizer(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix to None when there's no pretokenizer at all is a no-op."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.word_prefix is None
    model.word_prefix = None
    assert model.word_prefix is None
    assert model.pre_tokenizer is None


def test_set_word_prefix_no_op_for_byte_transforming_model(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix on a model that transforms into bytes is a no-op."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    assert model.word_prefix == "Ġ"
    before = model.vocabulary_size

    model.word_prefix = "X"

    assert model.word_prefix == "Ġ"
    assert model.vocabulary_size == before


def test_set_word_prefix_requires_single_character(small_tokenizer: Tokenizer) -> None:
    """Setting word_prefix to a string that isn't a single character raises."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        model.word_prefix = "ab"


def test_to_normal_form(small_tokenizer: Tokenizer) -> None:
    """to_normal_form gives the vocabulary a space word prefix and drops the subword prefix."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.word_prefix is None
    assert model.subword_prefix == "##"

    normal = model.to_normal_form()

    assert normal.word_prefix == " "
    assert normal.subword_prefix == ""
    # The original model is untouched.
    assert model.word_prefix is None
    assert model.subword_prefix == "##"

    call_tokenizer(normal)


def test_to_normal_form_no_op_for_byte_transforming_model(small_tokenizer: Tokenizer) -> None:
    """to_normal_form is a no-op for tokenizers that transform their input into bytes."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    assert model.transforms_into_bytes
    before = model.vocabulary_size

    normal = model.to_normal_form()

    assert normal.word_prefix == "Ġ"
    assert normal.vocabulary_size == before
    assert normal is not model

    call_tokenizer(normal)

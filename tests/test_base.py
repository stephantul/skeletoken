from tempfile import TemporaryDirectory

import pytest
from tokenizers import Tokenizer

from skeletoken.base import TokenizerModel
from skeletoken.common import PrependScheme
from skeletoken.models import BPE, ModelType, WordPiece
from skeletoken.normalizers import (
    ByteLevelNormalizer,
    LowercaseNormalizer,
    NFKCNormalizer,
    Normalizer,
    NormalizerSequence,
)
from skeletoken.padding import Padding
from skeletoken.postprocessors import (
    BertPostProcessor,
    ByteLevelPostProcessor,
    PostProcessorSequence,
    RobertaPostProcessor,
)
from skeletoken.pretokenizers import ByteLevelPreTokenizer, MetaspacePreTokenizer, PreTokenizerSequence


def test_tokenizer_model_from_tokenizer(small_tokenizer: Tokenizer) -> None:
    """Test creating a TokenizerModel from a Tokenizer instance."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.model, WordPiece)
    assert isinstance(model.added_tokens, list)
    assert model.normalizer is None
    assert model.pre_tokenizer is None
    assert model.post_processor is None
    assert model.decoder is None


def test_add_pre_tokenizer(small_tokenizer: Tokenizer) -> None:
    """Test adding a pre-tokenizer to the TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    model.add_pre_tokenizer(pre_tokenizer)

    # Tests removing None and adding a pre-tokenizer
    assert model.pre_tokenizer is not None
    assert isinstance(model.pre_tokenizer, ByteLevelPreTokenizer)

    model.add_pre_tokenizer(pre_tokenizer)

    # Tests adding a second pre-tokenizer and turning it into a sequence
    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 2

    model.add_pre_tokenizer(pre_tokenizer)

    # Tests adding a third pre-tokenizer and keeping it as a sequence
    assert isinstance(model.pre_tokenizer, PreTokenizerSequence)
    assert len(model.pre_tokenizer.pretokenizers) == 3


def test_add_normalizer(small_tokenizer: Tokenizer) -> None:
    """Test adding a normalizer to the TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    normalizer = ByteLevelNormalizer()
    model.add_normalizer(normalizer)

    # Tests removing None and adding a normalizer
    assert model.normalizer is not None
    assert isinstance(model.normalizer, ByteLevelNormalizer)

    model.add_normalizer(normalizer)

    # Tests adding a second normalizer and turning it into a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 2

    model.add_normalizer(normalizer)

    # Tests adding a third normalizer and keeping it as a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 3


def test_add_normalizer_prefix(small_tokenizer: Tokenizer) -> None:
    """Add a normalizer prefix."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)

    normalizer: Normalizer = ByteLevelNormalizer()
    model.add_normalizer(normalizer, prefix=True)

    # Tests removing None and adding a normalizer
    assert model.normalizer is not None
    assert isinstance(model.normalizer, ByteLevelNormalizer)

    normalizer = LowercaseNormalizer()
    model.add_normalizer(normalizer, prefix=True)

    # Tests adding a second normalizer and keeping it as a sequence
    assert isinstance(model.normalizer, NormalizerSequence)
    assert len(model.normalizer.normalizers) == 2
    assert isinstance(model.normalizer.normalizers[0], LowercaseNormalizer)
    assert isinstance(model.normalizer.normalizers[1], ByteLevelNormalizer)

    normalizer = NFKCNormalizer()
    model.add_normalizer(normalizer, prefix=False)

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
    model.add_post_processor(post_processor)

    # Tests removing None and adding a post-processor
    assert model.post_processor is not None
    assert isinstance(model.post_processor, ByteLevelPostProcessor)

    model.add_post_processor(post_processor)

    # Tests adding a second post-processor and turning it into a sequence
    assert isinstance(model.post_processor, PostProcessorSequence)
    assert len(model.post_processor.post_processors) == 2

    model.add_post_processor(post_processor)

    # Tests adding a third post-processor and keeping it as a sequence
    assert isinstance(model.post_processor, PostProcessorSequence)
    assert len(model.post_processor.post_processors) == 3


def test_from_pretrained(small_tokenizer: Tokenizer) -> None:
    """Test creating a TokenizerModel from a pretrained tokenizer."""
    with pytest.raises(FileNotFoundError):
        model = TokenizerModel.from_pretrained(".")

    with TemporaryDirectory() as temp_dir:
        small_tokenizer.save(f"{temp_dir}/tokenizer.json")
        model = TokenizerModel.from_pretrained(temp_dir)

        assert model.version == "1.0"
        assert model.model is not None
        assert isinstance(model.model, WordPiece)
        assert isinstance(model.added_tokens, list)
        assert model.normalizer is None
        assert model.pre_tokenizer is None
        assert model.post_processor is None
        assert model.decoder is None

        model = TokenizerModel.from_pretrained(f"{temp_dir}/tokenizer.json")

        assert model.version == "1.0"
        assert model.model is not None
        assert isinstance(model.model, WordPiece)
        assert isinstance(model.added_tokens, list)
        assert model.normalizer is None
        assert model.pre_tokenizer is None
        assert model.post_processor is None
        assert model.decoder is None


def test_make_greedy(small_tokenizer: Tokenizer) -> None:
    """Test whether the make greedy function works."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    tok_model.make_model_greedy()
    assert tok_model.model.type == ModelType.WORDPIECE
    assert tok_model.to_tokenizer()


def test_lowercase(small_tokenizer: Tokenizer) -> None:
    """Tests whether the model performs the lowercase test correctly."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not tok_model.lowercases_input
    tok_model.normalizer = LowercaseNormalizer()
    assert tok_model.lowercases_input
    assert tok_model.lowercases_input == tok_model.normalizer._lowercases


def test_byte_normalizes(small_tokenizer: Tokenizer) -> None:
    """Tests whether the model performs byte normalization correctly."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not tok_model.transforms_into_bytes
    tok_model.normalizer = ByteLevelNormalizer()
    assert tok_model.transforms_into_bytes
    tok_model.normalizer = None
    tok_model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=False)
    assert tok_model.transforms_into_bytes


def test_remove_token(small_tokenizer: Tokenizer) -> None:
    """Test removing a token from the vocabulary."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    tok_model.remove_token_from_vocabulary("a")
    with pytest.raises(ValueError):
        tok_model.remove_token_from_vocabulary("a")
    assert "a" not in tok_model.model.vocab.vocabulary


def test_add_token(small_tokenizer: Tokenizer) -> None:
    """Test adding a token to the vocabulary."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        tok_model.add_token_to_vocabulary("a")
    tok_model.add_token_to_vocabulary("new_token")
    assert "new_token" in tok_model.model.vocab.vocabulary


def test_replace_token(small_tokenizer: Tokenizer) -> None:
    """Test replace token interface."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        tok_model.replace_token_in_vocabulary("a", "b")
    tok_model.replace_token_in_vocabulary("b", "x")
    assert "b" not in tok_model.model.vocab.vocabulary
    assert "x" in tok_model.model.vocab.vocabulary


def test_decase_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test the decasing of the vocabulary."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    tok_model.added_tokens = []
    vocabulary = tok_model.model.vocab.sorted_vocabulary
    tok_model.decase_vocabulary()
    # This tokenizer does not assign any special tokens, so this is true.
    assert tok_model.model.vocab.sorted_vocabulary == [x.lower() for x in vocabulary]


def test_eos(small_tokenizer: Tokenizer) -> None:
    """Test getting the eos token."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert tok_model.eos is None
    tok_model.post_processor = ByteLevelPostProcessor(add_prefix_space=True, trim_offsets=True, use_regex=True)
    assert tok_model.eos is None
    tok_model.post_processor = None
    tok_model.post_processor = RobertaPostProcessor(
        sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False
    )
    assert tok_model.eos == "[SEP]"
    tok_model.post_processor = None
    tok_model.post_processor = BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0))
    assert tok_model.eos == "[SEP]"
    tok_model.post_processor = PostProcessorSequence(
        post_processors=[
            RobertaPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False),
            BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0)),
        ]
    )
    assert tok_model.eos is None


def test_bos(small_tokenizer: Tokenizer) -> None:
    """Test getting the eos token."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert tok_model.bos is None
    tok_model.post_processor = ByteLevelPostProcessor(add_prefix_space=True, trim_offsets=True, use_regex=True)
    assert tok_model.bos is None
    tok_model.post_processor = None
    tok_model.post_processor = RobertaPostProcessor(
        sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False
    )
    assert tok_model.bos == "[CLS]"
    tok_model.post_processor = None
    tok_model.post_processor = BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0))
    assert tok_model.bos == "[CLS]"
    tok_model.post_processor = PostProcessorSequence(
        post_processors=[
            RobertaPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0), trim_offsets=True, add_prefix_space=False),
            BertPostProcessor(sep=("[SEP]", 1), cls=("[CLS]", 0)),
        ]
    )
    assert tok_model.bos is None


def test_split(small_tokenizer: Tokenizer) -> None:
    """Test whether the split works correctly."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not tok_model.splits
    pretokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=False, trim_offsets=True)
    tok_model.add_pre_tokenizer(pretokenizer)
    assert not tok_model.splits
    pretokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    tok_model.add_pre_tokenizer(pretokenizer)
    assert tok_model.splits


def test_subword_prefix(small_tokenizer: Tokenizer) -> None:
    """Test getting the subword prefix token."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert tok_model.subword_prefix == "##"


def test_word_prefix(small_tokenizer: Tokenizer) -> None:
    """Test getting the word prefix token."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert tok_model.word_prefix == None

    tok_model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    assert tok_model.word_prefix == "Ġ"

    tok_model.pre_tokenizer = MetaspacePreTokenizer(replacement="▁", split=True, prepend_scheme=PrependScheme.ALWAYS)
    assert tok_model.word_prefix == "▁"


def test_get_added_token_for_form(small_tokenizer: Tokenizer) -> None:
    """Test getting the added token for a form."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    added_token = tok_model.get_added_token_for_form("f")
    assert added_token is not None
    assert added_token.content == "f"

    added_token = tok_model.get_added_token_for_form("zyx")
    assert added_token is None


def test_replace_token_in_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test replacing a token in the vocabulary."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    tok_model.replace_token_in_vocabulary("f", "g")
    assert tok_model.model.vocab["g"] is not None
    assert tok_model.get_added_token_for_form("g") is not None


def test_remove_token_from_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test removing a token from the vocabulary."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert tok_model.get_added_token_for_form("f") is not None
    tok_model.remove_token_from_vocabulary("f")
    assert tok_model.get_added_token_for_form("f") is None


def test_set_unk_token(small_tokenizer: Tokenizer) -> None:
    """Test setting the unknown token."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    tok_model.unk_token = "new_unk"
    assert tok_model.model.vocab["new_unk"] is not None
    assert tok_model.get_added_token_for_form("new_unk") is not None

    with pytest.raises(ValueError):
        tok_model.unk_token = None

    assert isinstance(tok_model.model, WordPiece)
    tok_model.model = BPE(
        vocab=tok_model.model.vocab,
        merges=[],
        dropout=0.0,
        unk_token="a",
        continuing_subword_prefix="",
        end_of_word_suffix="",
        fuse_unk=True,
        byte_fallback=True,
        ignore_merges=True,
    )
    tok_model.unk_token = None

    tok_model.unk_token = "a"
    tok_model.unk_token = "b"
    tok_model.unk_token = "a"

    tok_model.unk_token = None
    tok_model.unk_token = "OSTENTATIOUS"


def test_get_padding_token(small_tokenizer: Tokenizer) -> None:
    """Get the padding token from the tokenizer model."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert tok_model.padding is None
    assert tok_model.pad_token is None

    tok_model.padding = Padding(pad_token="[PAD]", pad_id=3, pad_type_id=0)
    assert tok_model.pad_token == "[PAD]"


def test_set_padding_token(small_tokenizer: Tokenizer) -> None:
    """Set the padding token for the tokenizer model."""
    tok_model = TokenizerModel.from_tokenizer(small_tokenizer)
    tok_model.pad_token = "[PAD]"
    assert tok_model.get_added_token_for_form("[PAD]") is not None
    tok_model.pad_token = None
    tok_model.pad_token = "OSTENTATIOUS"
    assert tok_model.get_added_token_for_form("OSTENTATIOUS") is not None
    assert tok_model.get_added_token_for_form("[PAD]") is None
    tok_model.added_tokens = []
    tok_model.pad_token = "OSTENTATIOUS"
    assert tok_model.get_added_token_for_form("OSTENTATIOUS") is not None
    tok_model.pad_token = "OSTENTATIOUS"
    assert tok_model.get_added_token_for_form("OSTENTATIOUS") is not None
    assert tok_model.get_added_token_for_form("[PAD]") is None
    tok_model.pad_token = "FUN"
    assert tok_model.get_added_token_for_form("FUN") is not None

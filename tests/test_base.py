from tempfile import TemporaryDirectory

import pytest
from tokenizers import Tokenizer

from skeletoken.base import TokenizerModel
from skeletoken.models import ModelType, WordPiece
from skeletoken.normalizers import (
    ByteLevelNormalizer,
    LowercaseNormalizer,
    NormalizerSequence,
)
from skeletoken.postprocessors import ByteLevelPostProcessor, PostProcessorSequence
from skeletoken.pretokenizers import ByteLevelPreTokenizer, PreTokenizerSequence


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

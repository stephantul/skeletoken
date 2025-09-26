import pytest
from tokenizers import Regex, Tokenizer
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.normalizers import Sequence as TokenizersNormalizerSequence
from tokenizers.pre_tokenizers import Sequence as TokenizersPreTokenizerSequence
from tokenizers.pre_tokenizers import Whitespace

from skeletoken import TokenizerModel
from skeletoken.normalizers import LowercaseNormalizer, NormalizerSequence
from skeletoken.preprocessor.normalizer import create_normalizer
from skeletoken.preprocessor.preprocessor import Preprocessor
from skeletoken.preprocessor.pretokenizer import create_pretokenizer
from skeletoken.preprocessor.utils import replace_pattern
from skeletoken.pretokenizers import PreTokenizerSequence, WhitespacePreTokenizer


def test_preprocessor_none() -> None:
    """Test the Preprocessor class."""
    preprocessor = Preprocessor()
    assert preprocessor.normalizer is None
    assert preprocessor.pretokenizer is None
    assert preprocessor("This is a test.") == ["This is a test."]


def test_preprocessor() -> None:
    """Test the Preprocessor class with normalizer and pretokenizer."""
    normalizer = TokenizersNormalizerSequence(normalizers=[NFD(), Lowercase()])  # type: ignore[arg-type]
    pretokenizer = TokenizersPreTokenizerSequence(pre_tokenizers=[Whitespace()])  # type: ignore[arg-type]
    preprocessor = Preprocessor(normalizer=normalizer, pretokenizer=pretokenizer)
    assert preprocessor.normalizer is not None
    assert preprocessor.pretokenizer is not None
    assert preprocessor("This is a test.") == ["this", "is", "a", "test", "."]
    assert preprocessor("This is a test.") == preprocessor.preprocess("This is a test.")

    assert preprocessor.preprocess_sequences(["This is a test.", "Another test!"]) == [
        ["this", "is", "a", "test", "."],
        ["another", "test", "!"],
    ]


def test_preprocessor_from_model(small_tokenizer: Tokenizer) -> None:
    """Test the Preprocessor class from a TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.add_normalizer(LowercaseNormalizer())
    model.add_pre_tokenizer(WhitespacePreTokenizer())
    preprocessor = Preprocessor.from_tokenizer_model(model)
    assert preprocessor.normalizer is not None
    assert preprocessor.pretokenizer is not None
    assert preprocessor("This is a test.") == ["this", "is", "a", "test", "."]


def test_create_normalizer_sequence() -> None:
    """Test the NormalizerSequence class."""
    normalizer = NormalizerSequence(normalizers=[LowercaseNormalizer()])
    base_normalizer = create_normalizer(normalizer)
    assert isinstance(base_normalizer, TokenizersNormalizerSequence)
    assert base_normalizer.normalize_str("This is a test.") == "this is a test."


def test_create_pretokenizer_sequence() -> None:
    """Test the PreTokenizerSequence class."""
    pretokenizer = PreTokenizerSequence(pretokenizers=[WhitespacePreTokenizer()])
    base_pretokenizer = create_pretokenizer(pretokenizer)
    assert isinstance(base_pretokenizer, TokenizersPreTokenizerSequence)
    assert base_pretokenizer.pre_tokenize_str("This is a test.") == [
        ("This", (0, 4)),
        ("is", (5, 7)),
        ("a", (8, 9)),
        ("test", (10, 14)),
        (".", (14, 15)),
    ]


def test_replace_pattern() -> None:
    """Test the replace_pattern function."""
    obj = {
        "pattern": {"Regex": r"\s+"},
        "replacement": " ",
        "another_key": "value",
    }
    replaced_obj = replace_pattern(obj)
    assert isinstance(replaced_obj["pattern"], Regex)
    assert replaced_obj["replacement"] == " "
    assert replaced_obj["another_key"] == "value"

    obj = {
        "pattern": {"String": "hello"},
        "replacement": " ",
        "another_key": "value",
    }
    replaced_obj = replace_pattern(obj)
    assert isinstance(replaced_obj["pattern"], str)
    assert replaced_obj["pattern"] == "hello"
    assert replaced_obj["replacement"] == " "
    assert replaced_obj["another_key"] == "value"

    obj = {
        "pattern": "hahahaha",
        "replacement": " ",
        "another_key": "value",
    }
    with pytest.raises(ValueError):
        replace_pattern(obj)

from tempfile import TemporaryDirectory

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as TokenizersByteLevelDecoder
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.normalizers import Sequence as TokenizersNormalizerSequence
from tokenizers.pre_tokenizers import Sequence as TokenizersPreTokenizerSequence
from tokenizers.pre_tokenizers import Whitespace

from skeletoken import TokenizerModel
from skeletoken.normalizers import LowercaseNormalizer
from skeletoken.pre_tokenizers import ByteLevelPreTokenizer, WhitespacePreTokenizer
from skeletoken.preprocessor.preprocessor import Preprocessor, decoder_from_model


def test_preprocessor_none() -> None:
    """Test the Preprocessor class."""
    preprocessor = Preprocessor()
    assert preprocessor.normalizer is None
    assert preprocessor.pretokenizer is None
    assert preprocessor.preprocess("This is a test.") == ["This is a test."]


def test_preprocessor() -> None:
    """Test the Preprocessor class with normalizer and pretokenizer."""
    normalizer = TokenizersNormalizerSequence(normalizers=[NFD(), Lowercase()])  # type: ignore[arg-type]
    pretokenizer = TokenizersPreTokenizerSequence(pre_tokenizers=[Whitespace()])  # type: ignore[arg-type]
    preprocessor = Preprocessor(normalizer=normalizer, pretokenizer=pretokenizer)
    assert preprocessor.normalizer is not None
    assert preprocessor.pretokenizer is not None
    assert preprocessor.preprocess("This is a test.") == ["this", "is", "a", "test", "."]
    assert preprocessor.preprocess("This is a test.") == preprocessor.preprocess("This is a test.")


def test_preprocessor_empty() -> None:
    """Test the Preprocessor class with normalizer and pretokenizer."""
    normalizer = TokenizersNormalizerSequence(normalizers=[NFD(), Lowercase()])  # type: ignore[arg-type]
    pretokenizer = TokenizersPreTokenizerSequence(pre_tokenizers=[Whitespace()])  # type: ignore[arg-type]
    preprocessor = Preprocessor(normalizer=normalizer, pretokenizer=pretokenizer)
    assert preprocessor.normalizer is not None
    assert preprocessor.pretokenizer is not None
    assert preprocessor.preprocess("") == []
    assert preprocessor.preprocess("", empty_sequence_is_token=True) == [""]
    assert preprocessor.preprocess("  ") == preprocessor.preprocess("  ")


def test_preprocessor_empty_sequence_is_token_with_prefixes() -> None:
    """An empty sequence that decoded from a prefixed token round-trips to that prefix."""
    preprocessor = Preprocessor(word_prefix="▁", subword_prefix="##")
    assert preprocessor.preprocess("", had_word_prefix=True, empty_sequence_is_token=True) == ["▁"]
    assert preprocessor.preprocess("", had_subword_prefix=True, empty_sequence_is_token=True) == ["##"]


def test_preprocessor_from_model(small_tokenizer: Tokenizer) -> None:
    """Test the Preprocessor class from a TokenizerModel."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model = model.add_normalizer(LowercaseNormalizer())
    model = model.add_pre_tokenizer(WhitespacePreTokenizer())
    preprocessor = Preprocessor.from_tokenizer_model(model)
    assert preprocessor.normalizer is not None
    assert preprocessor.pretokenizer is not None
    assert preprocessor.preprocess("This is a test.") == ["this", "is", "a", "test", "."]


def test_from_pretrained(small_tokenizer: Tokenizer) -> None:
    """Test the Preprocessor class from a pretrained model."""
    with TemporaryDirectory() as tmpdir:
        small_tokenizer.save(f"{tmpdir}/tokenizer.json")
        # Implicit test.
        preprocessor = Preprocessor.from_pretrained(f"{tmpdir}/tokenizer.json")
        assert preprocessor.normalizer is None
        assert preprocessor.pretokenizer is None
        assert preprocessor.preprocess("This is a test.") == ["This is a test."]


def test_decoder_from_model_byte_level(small_tokenizer: Tokenizer) -> None:
    """Test decoder_from_model returns a decoder when the model uses byte-level encoding."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    decoder = decoder_from_model(model)
    assert isinstance(decoder, TokenizersByteLevelDecoder)


def test_preprocessor_from_model_byte_level_without_add_prefix_space(small_tokenizer: Tokenizer) -> None:
    """Test that word_prefix is forced to None for byte-level models without add_prefix_space."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False, use_regex=True, trim_offsets=True)
    preprocessor = Preprocessor.from_tokenizer_model(model)
    assert preprocessor.word_prefix is None


def test_preprocessor_word_and_subword_prefix_interplay() -> None:
    """Test that had_word_prefix and had_subword_prefix compose: strip one marker, add the other."""
    from tokenizers.pre_tokenizers import Metaspace

    preprocessor = Preprocessor(
        pretokenizer=Metaspace(replacement="▁", prepend_scheme="always"),
        word_prefix="▁",
        subword_prefix="##",
    )

    # Word-initial (default): the pretokenizer's "▁" marker is kept.
    assert preprocessor.preprocess("hello", had_word_prefix=True, had_subword_prefix=False) == ["▁hello"]
    # Non-initial: the "▁" is stripped
    assert preprocessor.preprocess("hello", had_word_prefix=False, had_subword_prefix=False) == ["hello"]
    # Non-initial: "▁" is stripped.
    assert preprocessor.preprocess("hello", had_word_prefix=False, had_subword_prefix=True) == ["##hello"]


def test_preprocessor_decode_with_byte_transformer() -> None:
    """Test Preprocessor.decode applies the byte_transformer when set."""
    preprocessor = Preprocessor(byte_transformer=TokenizersByteLevelDecoder())
    result = preprocessor.decode("hello")
    assert result.original == "hello"
    assert result.decoded == "hello"

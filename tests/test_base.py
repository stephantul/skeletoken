from tempfile import TemporaryDirectory
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE as TokenizersBPE
from transformers import PreTrainedTokenizerFast

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
)
from skeletoken.padding import Padding
from skeletoken.postprocessors import (
    BertPostProcessor,
    ByteLevelPostProcessor,
    PostProcessorSequence,
    RobertaPostProcessor,
    SequenceToken,
    SpecialToken,
    SpecialTokens,
    TemplatePostProcessor,
    TokenContent,
    TokenInfo,
    TokenSequence,
)
from skeletoken.pretokenizers import ByteLevelPreTokenizer, MetaspacePreTokenizer, PreTokenizerSequence


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
    model.add_addedtoken("[OSTENTATIOUS]")
    # Test if it gets added to both the model and the vocabulary.
    assert model.added_tokens.get_token("[OSTENTATIOUS]") is not None
    assert "[OSTENTATIOUS]" in model.model.vocab


def test_turn_into_addedtoken(small_tokenizer: Tokenizer) -> None:
    """Test turning a regular token into an added token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.add_addedtoken("[OSTENTATIOUS]")
    # Should not crash, it is already an added token.
    model.turn_into_addedtoken("[OSTENTATIOUS]")
    # Should crash because it is not a regular token.
    with pytest.raises(ValueError):
        model.turn_into_addedtoken("bababbaa")


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
    assert len(model.post_processor.processors) == 2

    model.add_post_processor(post_processor)

    # Tests adding a third post-processor and keeping it as a sequence
    assert isinstance(model.post_processor, PostProcessorSequence)
    assert len(model.post_processor.processors) == 3

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


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
        assert isinstance(model.added_tokens, AddedTokens)
        assert model.normalizer is None
        assert model.pre_tokenizer is None
        assert model.post_processor is None
        assert model.decoder is None

        model = TokenizerModel.from_pretrained(f"{temp_dir}/tokenizer.json")

        assert model.version == "1.0"
        assert model.model is not None
        assert isinstance(model.model, WordPiece)
        assert isinstance(model.added_tokens, AddedTokens)
        assert model.normalizer is None
        assert model.pre_tokenizer is None
        assert model.post_processor is None
        assert model.decoder is None


def test_make_greedy(small_tokenizer: Tokenizer) -> None:
    """Test whether the make greedy function works."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.add_pre_tokenizer(ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True))
    model.make_model_greedy()
    assert model.model.type == ModelType.WORDPIECE
    assert model.to_tokenizer()


def test_make_greedy_fails(small_tokenizer: Tokenizer) -> None:
    """Test whether the make greedy function fails if no split is present."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        model.make_model_greedy()


def test_lowercase(small_tokenizer: Tokenizer) -> None:
    """Tests whether the model performs the lowercase test correctly."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not model.lowercases_input
    model.normalizer = LowercaseNormalizer()
    assert model.lowercases_input
    assert model.lowercases_input == model.normalizer._lowercases

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_byte_normalizes(small_tokenizer: Tokenizer) -> None:
    """Tests whether the model performs byte normalization correctly."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not model.transforms_into_bytes
    model.normalizer = ByteLevelNormalizer()
    assert model.transforms_into_bytes
    model.normalizer = None
    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=False)
    assert model.transforms_into_bytes

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_remove_token(small_tokenizer: Tokenizer) -> None:
    """Test removing a token from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.remove_token_from_vocabulary("a")
    with pytest.raises(ValueError):
        model.remove_token_from_vocabulary("a")
    assert "a" not in model.model.vocab.vocabulary

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_add_token(small_tokenizer: Tokenizer) -> None:
    """Test adding a token to the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        model.add_token_to_vocabulary("a")
    model.add_token_to_vocabulary("new_token")
    assert "new_token" in model.model.vocab.vocabulary

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_replace_token(small_tokenizer: Tokenizer) -> None:
    """Test replace token interface."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    with pytest.raises(ValueError):
        model.replace_token_in_vocabulary("a", "b")
    model.replace_token_in_vocabulary("b", "x")
    assert "b" not in model.model.vocab.vocabulary
    assert "x" in model.model.vocab.vocabulary

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_replace_token_template(small_tokenizer: Tokenizer) -> None:
    """Test replace token interface."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    t = TokenInfo(id="bos", ids=[1], tokens=["a"])
    s = SpecialTokens({"bos": t})
    tok = SpecialToken(SpecialToken=TokenContent(id="bos", type_id=1))
    seq = SequenceToken(Sequence=TokenContent(id="A", type_id=0))
    sequence = TokenSequence((tok, seq, tok))
    pair_seq = TokenSequence((tok, seq, tok, seq, tok))
    model.post_processor = TemplatePostProcessor(special_tokens=s, single=sequence, pair=pair_seq)
    with pytest.raises(ValueError):
        model.replace_token_in_vocabulary("a", "b")
    model.replace_token_in_vocabulary("b", "x")
    assert "b" not in model.model.vocab.vocabulary
    assert "x" in model.model.vocab.vocabulary

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_decase_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test the decasing of the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.added_tokens = AddedTokens([])
    vocabulary = model.model.vocab.sorted_vocabulary
    model.decase_vocabulary()
    # This tokenizer does not assign any special tokens, so this is true.
    assert model.model.vocab.sorted_vocabulary == [x.lower() for x in vocabulary]

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


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

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


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

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_split(small_tokenizer: Tokenizer) -> None:
    """Test whether the split works correctly."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert not model.splits
    pretokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=False, trim_offsets=True)
    model.add_pre_tokenizer(pretokenizer)
    assert not model.splits
    pretokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    model.add_pre_tokenizer(pretokenizer)
    assert model.splits

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_subword_prefix(small_tokenizer: Tokenizer) -> None:
    """Test getting the subword prefix token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.subword_prefix == "##"

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_word_prefix(small_tokenizer: Tokenizer) -> None:
    """Test getting the word prefix token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.word_prefix == None

    model.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True, use_regex=True, trim_offsets=True)
    assert model.word_prefix == "Ġ"

    model.pre_tokenizer = MetaspacePreTokenizer(replacement="▁", split=True, prepend_scheme=PrependScheme.ALWAYS)
    assert model.word_prefix == "▁"

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_replace_token_in_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test replacing a token in the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    model.replace_token_in_vocabulary("f", "g")
    assert model.model.vocab["g"] is not None
    assert model.added_tokens.get_token("g") is not None

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_remove_token_from_vocabulary(small_tokenizer: Tokenizer) -> None:
    """Test removing a token from the vocabulary."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.added_tokens.get_token("f") is not None
    model.remove_token_from_vocabulary("f")
    assert model.added_tokens.get_token("f") is None

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


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
    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_set_unk_token_template(small_tokenizer: Tokenizer) -> None:
    """Test setting the unknown token."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    t = TokenInfo(id="bos", ids=[1], tokens=["a"])
    s = SpecialTokens({"bos": t})
    tok = SpecialToken(SpecialToken=TokenContent(id="bos", type_id=1))
    seq = SequenceToken(Sequence=TokenContent(id="A", type_id=0))
    sequence = TokenSequence((tok, seq, tok))
    pair_seq = TokenSequence((tok, seq, tok, seq, tok))
    model.post_processor = TemplatePostProcessor(special_tokens=s, single=sequence, pair=pair_seq)
    model.unk_token = "a"
    model.unk_token = "b"

    assert isinstance(model.post_processor, TemplatePostProcessor)
    assert model.post_processor.special_tokens["bos"].tokens == ["b"]

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_get_padding_token(small_tokenizer: Tokenizer) -> None:
    """Get the padding token from the tokenizer model."""
    model = TokenizerModel.from_tokenizer(small_tokenizer)
    assert model.padding is None
    assert model.pad_token is None

    model.padding = Padding(pad_token="[PAD]", pad_id=3, pad_type_id=0)
    assert model.pad_token == "[PAD]"

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


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

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_from_transformers(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.model, WordPiece)
    assert isinstance(model.added_tokens, AddedTokens)

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_from_transformers_missing_unk(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    transformers_tokenizer.SPECIAL_TOKENS_ATTRIBUTES.remove("unk_token")
    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.added_tokens, AddedTokens)

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_from_transformers_missing_pad(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    transformers_tokenizer.SPECIAL_TOKENS_ATTRIBUTES.remove("pad_token")
    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.model, WordPiece)
    assert isinstance(model.added_tokens, AddedTokens)

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()


def test_missing_unk_model(transformers_tokenizer: PreTrainedTokenizerFast) -> None:
    """Test creating a TokenizerModel from a transformers tokenizer."""
    transformers_tokenizer._tokenizer.model = TokenizersBPE(
        vocab={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, " ": 6}, merges=[], unk_token=None
    )  # type: ignore
    transformers_tokenizer._tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")

    model = TokenizerModel.from_transformers_tokenizer(transformers_tokenizer)
    assert model.version == "1.0"
    assert model.model is not None
    assert isinstance(model.added_tokens, AddedTokens)

    # Implicit test. If this fails, the model is incorrect.
    model.to_tokenizer()

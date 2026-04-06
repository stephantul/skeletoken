from __future__ import annotations

from dataclasses import dataclass

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Normalizer as BaseNormalizer
from tokenizers.pre_tokenizers import PreTokenizer as BasePretokenizer

from skeletoken.base import TokenizerModel


@dataclass
class Decoded:
    original: str
    decoded: str


class Preprocessor:
    def __init__(
        self,
        byte_transformer: ByteLevelDecoder | None = None,
        normalizer: BaseNormalizer | None = None,
        pretokenizer: BasePretokenizer | None = None,
        subword_prefix: str | None = None,
        word_prefix: str | None = None,
    ) -> None:
        """Initialize the Preprocessor with optional normalizer and pretokenizer."""
        self.normalizer = normalizer
        self.pretokenizer = pretokenizer
        self.byte_transformer = byte_transformer
        self.subword_prefix = subword_prefix
        self.word_prefix = word_prefix

    def decode(self, sequence: str) -> Decoded:
        """Preprocess a single sequence."""
        if self.byte_transformer:
            decoded = self.byte_transformer.decode([sequence])
        else:
            decoded = sequence
        return Decoded(
            original=sequence,
            decoded=decoded,
        )

    def decode_sequences(self, sequences: list[str]) -> list[Decoded]:
        """Preprocess a list of sequences using multithreading."""
        return [self.decode(seq) for seq in sequences]

    def preprocess(self, sequence: str) -> list[str]:
        """Preprocess a single sequence."""
        removed_prefix = False
        if self.subword_prefix and sequence.startswith(self.subword_prefix):
            sequence = sequence.removeprefix(self.subword_prefix)
            removed_prefix = True

        removed_word_prefix = False
        if self.word_prefix and sequence.startswith(self.word_prefix):
            sequence = sequence.removeprefix(self.word_prefix)
            removed_word_prefix = True

        if self.normalizer is not None:
            sequence = self.normalizer.normalize_str(sequence)
        if self.pretokenizer is not None:
            x = [text for text, _ in self.pretokenizer.pre_tokenize_str(sequence)]
        else:
            x = [sequence]
        if removed_prefix and x:
            x[0] = f"{self.subword_prefix}{x[0]}"
        if self.word_prefix and not removed_word_prefix and x:
            x[0] = x[0].removeprefix(self.word_prefix)
        return x

    def preprocess_sequences(self, sequences: list[str]) -> list[list[str]]:
        """Preprocess a list of sequences using multithreading."""
        return [self.preprocess(seq) for seq in sequences]

    @classmethod
    def from_tokenizer_model(cls, model: TokenizerModel) -> Preprocessor:
        """Set the normalizer and pretokenizer from a TokenizerModel."""
        tokenizer = model.to_tokenizer()
        return cls(
            normalizer=tokenizer.normalizer,
            pretokenizer=tokenizer.pre_tokenizer,
            byte_transformer=ByteLevelDecoder() if model.transforms_into_bytes else None,
            subword_prefix=model.subword_prefix,
            word_prefix=None if model.transforms_into_bytes else model.word_prefix,
        )

    @classmethod
    def from_pretrained(cls, name: str) -> Preprocessor:
        """Create a Preprocessor from a pretrained tokenizer model name."""
        model = TokenizerModel.from_pretrained(name)
        return cls.from_tokenizer_model(model)

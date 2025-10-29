from __future__ import annotations

from tokenizers.normalizers import Normalizer as BaseNormalizer
from tokenizers.pre_tokenizers import PreTokenizer as BasePretokenizer

from skeletoken.base import TokenizerModel


class Preprocessor:
    def __init__(self, normalizer: BaseNormalizer | None = None, pretokenizer: BasePretokenizer | None = None) -> None:
        """Initialize the Preprocessor with optional normalizer and pretokenizer."""
        self.normalizer = normalizer
        self.pretokenizer = pretokenizer

    def __call__(self, sequence: str) -> list[str]:
        """Apply the normalizer and pretokenizer to the input sequence."""
        return self.preprocess(sequence)

    def preprocess(self, sequence: str) -> list[str]:
        """Preprocess a single sequence."""
        if self.normalizer is not None:
            sequence = self.normalizer.normalize_str(sequence)
        if self.pretokenizer is not None:
            return [text for text, _ in self.pretokenizer.pre_tokenize_str(sequence)]
        return [sequence]

    def preprocess_sequences(self, sequences: list[str]) -> list[list[str]]:
        """Preprocess a list of sequences using multithreading."""
        return [self(seq) for seq in sequences]

    @classmethod
    def from_tokenizer_model(cls, model: TokenizerModel) -> Preprocessor:
        """Set the normalizer and pretokenizer from a TokenizerModel."""
        tokenizer = model.to_tokenizer()
        return cls(normalizer=tokenizer.normalizer, pretokenizer=tokenizer.pre_tokenizer)

    @classmethod
    def from_pretrained(cls, name: str) -> Preprocessor:
        """Create a Preprocessor from a pretrained tokenizer model name."""
        model = TokenizerModel.from_pretrained(name)
        return cls.from_tokenizer_model(model)

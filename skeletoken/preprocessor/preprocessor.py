from __future__ import annotations

from tokenizers.normalizers import Normalizer as BaseNormalizer
from tokenizers.pre_tokenizers import PreTokenizer as BasePretokenizer

from skeletoken.base import TokenizerModel
from skeletoken.preprocessor.normalizer import create_normalizer
from skeletoken.preprocessor.pretokenizer import create_pretokenizer


class Preprocessor:
    def __init__(self, normalizer: BaseNormalizer | None = None, pretokenizer: BasePretokenizer | None = None) -> None:
        """Initialize the Preprocessor with optional normalizer and pretokenizer."""
        self.normalizer = normalizer
        self.pretokenizer = pretokenizer

    def __call__(self, sequence: str) -> list[str]:
        """Apply the normalizer and pretokenizer to the input sequence."""
        if self.normalizer is not None:
            sequence = self.normalizer.normalize_str(sequence)
        if self.pretokenizer is not None:
            return [text for text, offsets in self.pretokenizer.pre_tokenize_str(sequence)]
        return [sequence]

    @classmethod
    def from_tokenizer_model(cls, model: TokenizerModel) -> Preprocessor:
        """Set the normalizer and pretokenizer from a TokenizerModel."""
        normalizer = create_normalizer(model.normalizer) if model.normalizer is not None else None
        pretokenizer = create_pretokenizer(model.pre_tokenizer) if model.pre_tokenizer is not None else None
        return cls(normalizer=normalizer, pretokenizer=pretokenizer)

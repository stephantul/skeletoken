from __future__ import annotations

from dataclasses import dataclass

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Normalizer as BaseNormalizer
from tokenizers.pre_tokenizers import PreTokenizer as BasePretokenizer

from skeletoken.base import TokenizerModel


@dataclass
class Decoded:
    # Original string form
    original: str
    # The base string form
    decoded: str
    # The subword prefix the token had
    subword_prefix: bool
    # The word prefix the token had
    word_prefix: bool


def _remove_prefix(sequence: str, prefix: str | None) -> tuple[str, bool]:
    """Removes a prefix and indicates whether something changed."""
    if prefix is None:
        return sequence, False
    new_sequence = sequence.removeprefix(prefix)
    return new_sequence, new_sequence != sequence


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
        decoded, has_subword = _remove_prefix(sequence, self.subword_prefix)
        decoded, has_word = _remove_prefix(decoded, self.word_prefix)

        if self.byte_transformer:
            decoded = self.byte_transformer.decode([decoded])
        return Decoded(
            original=sequence,
            decoded=decoded,
            subword_prefix=has_subword,
            word_prefix=has_word,
        )

    def decode_sequences(self, sequences: list[str]) -> list[Decoded]:
        """Preprocess a list of sequences using multithreading."""
        return [self.decode(seq) for seq in sequences]

    def preprocess(self, sequence: str, had_word_prefix: bool = False, had_subword_prefix: bool = False) -> list[str]:
        """Preprocess a single sequence.

        Note that word prefix and subword prefix tokens like '##' and `_` are
        treated as regular characters here. So any tokens you input here should be
        decoded using `decode`. This removes these tokens and stores whether they had
        such a prefix.
        """
        if self.normalizer is not None:
            sequence = self.normalizer.normalize_str(sequence)
        if self.pretokenizer is not None:
            processed = [text for text, _ in self.pretokenizer.pre_tokenize_str(sequence)]
        else:
            processed = [sequence]
        # This is annoying: pretokenizers turn the empty string into an empty list.
        if not sequence:
            processed = [self.word_prefix or "" if had_word_prefix else ""]
        if processed:
            first_token = processed[0]
            if not had_word_prefix and self.word_prefix:
                first_token = first_token.removeprefix(self.word_prefix)
            if had_subword_prefix:
                # This should not happen.
                assert self.subword_prefix is not None
                first_token = f"{self.subword_prefix}{first_token}"
            processed[0] = first_token

        return processed

    @classmethod
    def from_tokenizer_model(cls, model: TokenizerModel) -> Preprocessor:
        """Set the normalizer and pretokenizer from a TokenizerModel."""
        tokenizer = model.to_tokenizer()
        return cls(
            normalizer=tokenizer.normalizer,
            pretokenizer=tokenizer.pre_tokenizer,
            byte_transformer=ByteLevelDecoder() if model.transforms_into_bytes else None,
            subword_prefix=model.subword_prefix,
            word_prefix=model.word_prefix,
        )

    @classmethod
    def from_pretrained(cls, name: str) -> Preprocessor:
        """Create a Preprocessor from a pretrained tokenizer model name."""
        model = TokenizerModel.from_pretrained(name)
        return cls.from_tokenizer_model(model)

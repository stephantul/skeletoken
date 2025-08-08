from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer

from tokenizerdatamodels.addedtoken import AddedToken
from tokenizerdatamodels.decoders import DecoderDiscriminator
from tokenizerdatamodels.models import BPE, ModelDiscriminator, Unigram, WordLevel, WordPiece
from tokenizerdatamodels.normalizers import NormalizerDiscriminator, NormalizerSequence
from tokenizerdatamodels.postprocessors import PostProcessorDiscriminator, PostProcessorSequence
from tokenizerdatamodels.pretokenizers import PreTokenizerDiscriminator, PretokenizerSequence

logger = logging.getLogger(__name__)


class TokenizerModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: Literal["1.0"] = "1.0"
    truncation: None
    padding: None
    added_tokens: list[AddedToken]
    normalizer: None | NormalizerDiscriminator
    pre_tokenizer: None | PreTokenizerDiscriminator
    post_processor: None | PostProcessorDiscriminator
    decoder: None | DecoderDiscriminator
    model: ModelDiscriminator

    def add_pre_tokenizer(self, pre_tokenizer: PreTokenizerDiscriminator) -> None:
        """Add a pre-tokenizer to the tokenizer model."""
        if self.pre_tokenizer is None:
            self.pre_tokenizer = pre_tokenizer
        elif isinstance(self.pre_tokenizer, PretokenizerSequence):
            self.pre_tokenizer.pretokenizers.append(pre_tokenizer)
        else:
            self.pre_tokenizer = PretokenizerSequence(pretokenizers=[self.pre_tokenizer, pre_tokenizer])

    def add_post_processor(self, post_processor: PostProcessorDiscriminator) -> None:
        """Add a post-processor to the tokenizer model."""
        if self.post_processor is None:
            self.post_processor = post_processor
        elif isinstance(self.post_processor, PostProcessorSequence):
            self.post_processor.post_processors.append(post_processor)
        else:
            self.post_processor = PostProcessorSequence(post_processors=[self.post_processor, post_processor])

    def add_normalizer(self, normalizer: NormalizerDiscriminator) -> None:
        """Add a normalizer to the tokenizer model."""
        if self.normalizer is None:
            self.normalizer = normalizer
        elif isinstance(self.normalizer, NormalizerSequence):
            self.normalizer.normalizers.append(normalizer)
        else:
            self.normalizer = NormalizerSequence(normalizers=[self.normalizer, normalizer])

    @classmethod
    def from_tokenizer(cls: type[TokenizerModel], tokenizer: Tokenizer) -> "TokenizerModel":
        """Create a TokenizerModel from a Tokenizer instance."""
        return cls.from_string(tokenizer.to_str())

    @classmethod
    def from_pretrained(cls: type[TokenizerModel], path_or_repo: str | Path) -> "TokenizerModel":
        """Create a TokenizerModel from a pretrained tokenizer."""
        path = Path(path_or_repo)
        tokenizer: Tokenizer
        if path.exists() and path.is_dir():
            if not (path / "tokenizer.json").exists():
                raise FileNotFoundError(
                    f"No tokenizer.json found in the directory: {path}. Please provide a valid path."
                )
            # If a tokenizer.json file exists, load it directly
            tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        elif path.exists() and path.is_file():
            tokenizer = Tokenizer.from_file(str(path))
        else:
            tokenizer = Tokenizer.from_pretrained(str(path))
        return cls.from_tokenizer(tokenizer)

    @classmethod
    def from_string(cls: type[TokenizerModel], json_str: str) -> TokenizerModel:
        """Create a TokenizerModel from a JSON string."""
        return cls.model_validate_json(json_str)

    def to_tokenizer(self) -> Tokenizer:
        """Convert the TokenizerModel back to a Tokenizer instance."""
        return Tokenizer.from_str(self.model_dump_json())

    def make_model_greedy(self) -> TokenizerModel:
        """Convert the TokenizerModel to a greedy tokenizer model."""
        self_copy = self.copy(deep=True)
        self_copy.model = self_copy.model.to_greedy()
        return self_copy

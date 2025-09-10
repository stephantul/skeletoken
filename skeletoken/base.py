from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer

from skeletoken.addedtoken import AddedTokens
from skeletoken.decase.decase import decase_vocabulary
from skeletoken.decoders import DecoderDiscriminator
from skeletoken.models import MODELS_THAT_NEED_UNK, ModelDiscriminator, get_subword_prefix_token
from skeletoken.normalizers import LowercaseNormalizer, NormalizerDiscriminator, NormalizerSequence
from skeletoken.padding import Padding
from skeletoken.postprocessors import (
    PostProcessorDiscriminator,
    PostProcessorSequence,
    get_bos_token_from_post_processor,
    get_eos_token_from_post_processor,
    maybe_replace_token_in_post_processor,
)
from skeletoken.pretokenizers import PreTokenizerDiscriminator, PreTokenizerSequence, get_metaspace
from skeletoken.truncation import Truncation

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class TokenizerModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: Literal["1.0"] = "1.0"
    truncation: None | Truncation = None
    padding: None | Padding = None
    added_tokens: AddedTokens = AddedTokens.model_validate([])
    normalizer: None | NormalizerDiscriminator = None
    pre_tokenizer: None | PreTokenizerDiscriminator = None
    post_processor: None | PostProcessorDiscriminator = None
    decoder: None | DecoderDiscriminator = None
    model: ModelDiscriminator

    def model_post_init(self, __context: dict) -> None:
        """Post-initialization processing."""
        # Add any missing added tokens to the vocabulary.
        for token in self.added_tokens.root:
            # Get the content of the token
            content = token.content
            if content not in self.model.vocab.vocabulary:
                logger.warning(f"Adding missing added token '{content}' to the vocabulary.")
                self._add_token_to_vocabulary(content, is_added_token=True)
            # Retro-actively set the ID.
            curr_id = token.id
            new_id = self.model.vocab[content]
            if curr_id != new_id:
                logger.warning(
                    f"Updating ID of added token '{content}' from {curr_id} to {new_id} to match vocabulary index."
                )
            token.id = new_id
        unk_token = self.unk_token
        if unk_token:
            if unk_token not in self.model.vocab:
                logger.warning(f"Adding unk_token '{unk_token}' to the vocabulary.")
                self._add_token_to_vocabulary(unk_token, is_added_token=True)
            added_token = self.added_tokens.get_token(unk_token)
            if not added_token:
                logger.warning(f"Turning unk_token '{unk_token}' into an AddedToken.")
                self.turn_into_addedtoken(
                    unk_token, is_special=True, normalized=False, single_word=True, lstrip=False, rstrip=False
                )

    def add_addedtoken(
        self,
        token: str,
        is_special: bool = False,
        normalized: bool = False,
        single_word: bool = True,
        lstrip: bool = True,
        rstrip: bool = True,
    ) -> None:
        """Adds an added token to the tokenizer model."""
        self._add_token_to_vocabulary(token, is_added_token=True)
        self.turn_into_addedtoken(
            token, is_special=is_special, normalized=normalized, single_word=single_word, lstrip=lstrip, rstrip=rstrip
        )

    def turn_into_addedtoken(
        self,
        token: str,
        is_special: bool = False,
        normalized: bool = False,
        single_word: bool = True,
        lstrip: bool = True,
        rstrip: bool = True,
    ) -> None:
        """Turns an existing token into an an added token and add it to added_tokens."""
        if token not in self.model.vocab.vocabulary:
            raise ValueError(
                f"Token '{token}' not found in the vocabulary. Please add it first using `add_token_to_vocabulary` or `add_addedtoken`."
            )
        self.added_tokens.maybe_add_token(
            token,
            is_special=is_special,
            normalized=normalized,
            single_word=single_word,
            lstrip=lstrip,
            rstrip=rstrip,
            id=self.model.vocab[token],
        )

    def add_token_to_vocabulary(self, token: str) -> None:
        """Adds a token to the tokenizer's vocabulary."""
        self._add_token_to_vocabulary(token, is_added_token=False)

    def _add_token_to_vocabulary(self, token: str, is_added_token: bool = False) -> None:
        """Adds an added token to the vocabulary."""
        self.model.add_token(token, is_added_token=is_added_token)

    def replace_token_in_vocabulary(self, old_token: str, new_token: str) -> None:
        """Replaces a token with another one."""
        self._replace_token_in_vocabulary(old_token, new_token, is_added_token=False)

    def _replace_token_in_vocabulary(self, old_token: str, new_token: str, is_added_token: bool = False) -> None:
        """Replaces a token with another one. It keeps the old index in the vocabulary."""
        self.model.replace_token(old_token, new_token, is_added_token=is_added_token)
        self.added_tokens.maybe_replace_token(old_token, new_token)
        if self.post_processor is not None:
            self.post_processor = maybe_replace_token_in_post_processor(
                old_token, new_token, self.model.vocab[new_token], self.post_processor
            )

    def remove_token_from_vocabulary(self, token: str) -> None:
        """Removes a token from the vocabulary."""
        self.model.remove_token(token)
        self.added_tokens.maybe_remove_token(token)

    def decase_vocabulary(self, remove_collisions: bool = False) -> TokenizerModel:
        """
        Decases the vocabulary.

        Parameters
        ----------
        remove_collisions: bool
            If this is set, any tokens that are colliding after lowercasing will be removed from the vocabulary.
            This leads to smaller models, but also a mismatch between tokenizers, which
            should be remedied manually.

        """
        # Special tokens and unnormalized added tokens need to be skipped.
        special_tokens = self.added_tokens.get_special_tokens() + self.added_tokens.get_unnormalized_tokens()
        sorted_vocab = self.model.vocab.sorted_vocabulary
        vocabulary = decase_vocabulary(
            sorted_vocab,
            [x.content for x in special_tokens],
            is_byte=self.transforms_into_bytes,
            remove_collisions=remove_collisions,
        )
        self.model.vocab.replace_vocabulary(vocabulary)
        if not self.lowercases_input:
            self.add_normalizer(LowercaseNormalizer(), prefix=True)

        return self

    def add_pre_tokenizer(self, pre_tokenizer: PreTokenizerDiscriminator) -> TokenizerModel:
        """Add a pre-tokenizer to the tokenizer model."""
        if self.pre_tokenizer is None:
            self.pre_tokenizer = pre_tokenizer
        elif isinstance(self.pre_tokenizer, PreTokenizerSequence):
            self.pre_tokenizer.pretokenizers.append(pre_tokenizer)
        else:
            self.pre_tokenizer = PreTokenizerSequence(pretokenizers=[self.pre_tokenizer, pre_tokenizer])

        return self

    def add_post_processor(self, post_processor: PostProcessorDiscriminator) -> TokenizerModel:
        """Add a post-processor to the tokenizer model."""
        if self.post_processor is None:
            self.post_processor = post_processor
        elif isinstance(self.post_processor, PostProcessorSequence):
            self.post_processor.processors.append(post_processor)
        else:
            self.post_processor = PostProcessorSequence(processors=[self.post_processor, post_processor])

        return self

    def add_normalizer(self, normalizer: NormalizerDiscriminator, prefix: bool = False) -> TokenizerModel:
        """
        Add a normalizer to the tokenizer model.

        Parameters
        ----------
        normalizer: NormalizerDiscriminator
            The normalizer to add.
        prefix: bool
            Whether to add the normalizer before the other normalizers.
            This can be useful if, for example, one of your normalizers performs
            a destructive transform.

        """
        if self.normalizer is None:
            self.normalizer = normalizer
        elif isinstance(self.normalizer, NormalizerSequence):
            self.normalizer.normalizers.append(normalizer)
        else:
            if prefix:
                self.normalizer = NormalizerSequence(normalizers=[normalizer, self.normalizer])
            else:
                self.normalizer = NormalizerSequence(normalizers=[self.normalizer, normalizer])

        return self

    @classmethod
    def from_tokenizer(cls: type[TokenizerModel], tokenizer: Tokenizer) -> TokenizerModel:
        """Create a TokenizerModel from a Tokenizer instance."""
        return cls.from_string(tokenizer.to_str())

    @classmethod
    def from_pretrained(cls: type[TokenizerModel], path_or_repo: str | Path) -> TokenizerModel:
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
            tokenizer = Tokenizer.from_pretrained(str(path))  # pragma: nocover
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
        if not self.splits:
            raise ValueError("Cannot make a tokenizer greedy if it does not split the input.")
        self.model = self.model.to_greedy()
        return self

    @property
    def lowercases_input(self) -> bool:
        """Check if the tokenizer lowercases the input."""
        return self.normalizer is not None and self.normalizer.lowercases

    @property
    def transforms_into_bytes(self) -> bool:
        """
        Check if the tokenizer transforms the input into bytes.

        There's two ways this can happen:
            1. If the pretokenizer is a ByteLevelPreTokenizer.
            2. If the normalizer is a ByteLevelNormalizer.

        This is a bit more complicated, because the pretokenizer can be a sequence of pretokenizers,
        and the normalizer can also be a sequence of normalizers.
        """
        if self.pre_tokenizer is not None and self.pre_tokenizer._byte_pretokenizes:
            return True
        if self.normalizer is not None and self.normalizer.byte_normalizes:
            return True
        return False

    @property
    def eos(self) -> list[str] | None:
        """Get the end-of-sequence tokens."""
        if self.post_processor is None:
            return None
        return get_eos_token_from_post_processor(self.post_processor)

    @property
    def bos(self) -> list[str] | None:
        """Get the beginning-of-sequence tokens."""
        if self.post_processor is None:
            return None
        return get_bos_token_from_post_processor(self.post_processor)

    @property
    def splits(self) -> bool:
        """Whether the tokenizer can split the string somehow."""
        if self.pre_tokenizer is None:
            return False
        return self.pre_tokenizer._splits

    @property
    def subword_prefix(self) -> str | None:
        """Get the prefix token, if any."""
        return get_subword_prefix_token(self.model)

    @property
    def word_prefix(self) -> str | None:
        """Get the word prefix token, if any."""
        if self.pre_tokenizer is None:
            return None
        # Word prefixes are not handled by the model, but added
        # by pretokenizers
        if self.transforms_into_bytes:
            return "Ġ"
        return get_metaspace(self.pre_tokenizer)

    @property
    def unk_token(self) -> str | None:
        """Get the unk token, if any."""
        return self.model.unk_token

    @unk_token.setter
    def unk_token(self, token: str | None) -> None:
        """Set the unk token of the tokenizer model."""
        old_unk_token = self.unk_token
        if token is None:
            if isinstance(self.model, MODELS_THAT_NEED_UNK):
                raise ValueError("Cannot unset unk_token for WordPiece or WordLevel models.")
            if old_unk_token is not None:
                logger.info(f"Removing unk_token '{self.model.unk_token}' from the tokenizer.")
            self.model.unk_token = None
            return
        if old_unk_token is None:
            if not token in self.model.vocab:
                logger.info(f"Adding {token} to the vocabulary.")
                self._add_token_to_vocabulary(token, is_added_token=True)
            logger.info(f"Setting unk_token to '{token}'.")
            index = self.model.vocab[token]
            self.added_tokens.maybe_add_token(
                token=token, is_special=True, normalized=True, single_word=True, rstrip=True, lstrip=True, id=index
            )
        elif old_unk_token in self.model.vocab:
            logger.info(f"Changing unk_token from '{old_unk_token}' to '{token}'.")
            self.added_tokens.maybe_replace_token(old_unk_token, token)
            if token not in self.model.vocab:
                self._add_token_to_vocabulary(token, is_added_token=True)
            if self.post_processor is not None:
                self.post_processor = maybe_replace_token_in_post_processor(
                    old_unk_token, token, self.model.vocab[token], self.post_processor
                )
        self.model.unk_token = token

    @property
    def pad_token(self) -> str | None:
        """Get the padding token, if any."""
        if self.padding is None:
            return None
        return self.padding.pad_token

    @pad_token.setter
    def pad_token(self, token: str | None) -> None:
        """Set the padding token of the tokenizer model."""
        if token is None:
            current_token = self.pad_token
            if current_token is not None:
                logger.info("Removing padding token from the tokenizer.")

            self.padding = None
            return
        # No padding module set
        if self.padding is None:
            logger.info(f"Setting padding token to '{token}' and creating a padding module.")
            if token not in self.model.vocab:
                self.model.vocab.add_token(token)
            index = self.model.vocab[token]
            self.padding = Padding(pad_id=index, pad_type_id=0, pad_token=token)
        elif token in self.model.vocab:
            logger.info(f"Changing padding token to existing token '{token}'.")
            self.padding.pad_id = self.model.vocab[token]
            self.padding.pad_token = token
        else:
            logger.info(f"Changing padding token to new token '{token}'.")
            old_pad_token = self.padding.pad_token
            self._replace_token_in_vocabulary(old_pad_token, token, is_added_token=True)
            self.padding.pad_token = token

        # We know token is in vocab here.
        self.added_tokens.maybe_add_token(
            token,
            id=self.model.vocab[token],
            is_special=True,
            normalized=True,
            single_word=True,
            lstrip=True,
            rstrip=True,
        )

    @classmethod
    def from_transformers_tokenizer(cls: type[TokenizerModel], hf_tokenizer: PreTrainedTokenizerFast) -> TokenizerModel:
        """Load a HuggingFace tokenizer from a local path or a model repo."""
        special_tokens = hf_tokenizer.special_tokens_map
        unk_token = special_tokens.get("unk_token", None)
        pad_token = special_tokens.get("pad_token", None)

        model = cls.from_tokenizer(hf_tokenizer._tokenizer)
        if unk_token is not None and isinstance(unk_token, str):
            if model.unk_token is not None and model.unk_token != unk_token:
                logger.warning(
                    f"Overriding existing unk_token '{model.unk_token}' with the one from the HuggingFace tokenizer: '{unk_token}'."
                )
            if model.unk_token is None:
                logger.warning(
                    f"HuggingFace tokenizer defines an unk_token, but the Skeletoken model does not. Setting it to '{unk_token}'."
                )
            model.unk_token = unk_token
        if pad_token is not None and isinstance(pad_token, str):
            if model.pad_token is not None and model.pad_token != pad_token:
                logger.warning(
                    f"Overriding existing pad_token '{model.pad_token}' with the one from the HuggingFace tokenizer: '{pad_token}'."
                )
            if model.pad_token is None:
                logger.warning(
                    f"HuggingFace tokenizer defines a pad_token, but the Skeletoken model does not. Setting it to '{pad_token}'."
                )
            model.pad_token = pad_token

        return model

    @classmethod
    def from_transformers(cls, path: str) -> TokenizerModel:  # pragma: nocover
        """Load a HuggingFace tokenizer from a local path or a model repo."""
        # Disable the transformers logger to avoid cluttering the output.
        # If skeletoken is just installed, it will print an annoying warning.
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.disabled = True
        from transformers import AutoTokenizer

        transformers_logger.disabled = False
        hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path)
        return cls.from_transformers_tokenizer(hf_tokenizer)

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from skeletoken.addedtoken import AddedTokens
from skeletoken.common import PathLike
from skeletoken.decase.decase import decase_vocabulary
from skeletoken.decoders import DecoderDiscriminator
from skeletoken.models import MODELS_THAT_NEED_UNK, ModelDiscriminator, WordPiece, get_subword_prefix_token
from skeletoken.normalizers import LowercaseNormalizer, NormalizerDiscriminator, NormalizerSequence
from skeletoken.padding import Padding
from skeletoken.postprocessors import (
    PostProcessorDiscriminator,
    PostProcessorSequence,
    get_bos_token_from_post_processor,
    get_eos_token_from_post_processor,
    maybe_replace_token_in_post_processor,
)
from skeletoken.pretokenizers import (
    ByteLevelPreTokenizer,
    FixedLengthPreTokenizer,
    PreTokenizerDiscriminator,
    PreTokenizerSequence,
    get_add_prefix_space,
    get_metaspace,
    get_pretokenizer_of_type,
)
from skeletoken.truncation import Truncation

if TYPE_CHECKING:
    from skeletoken.model_delta import ModelDelta  # pragma: nocover
    from skeletoken.preprocessor import Preprocessor  # pragma: nocover

logger = logging.getLogger(__name__)


class TokenizerModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: Literal["1.0"] = "1.0"
    truncation: Truncation | None = None
    padding: Padding | None = None
    added_tokens: AddedTokens = Field(default_factory=lambda: AddedTokens.model_validate([]))
    normalizer: NormalizerDiscriminator | None = None
    pre_tokenizer: PreTokenizerDiscriminator | None = None
    post_processor: PostProcessorDiscriminator | None = None
    decoder: DecoderDiscriminator | None = None
    model: ModelDiscriminator
    _original_tokenizer: TokenizerModel = PrivateAttr(init=False)
    # Remapping from old token IDs to new token IDs after vocabulary changes.
    _original_class: type[PreTrainedTokenizerFast] | None = PrivateAttr(init=False, default=None)
    _preprocessor: Preprocessor | None = None

    def deep_copy(self) -> TokenizerModel:
        """Return a deep copy of this TokenizerModel."""
        self._preprocessor = None
        return self.model_copy(deep=True)

    def model_post_init(self, __context: dict[Any, Any]) -> None:  # noqa: C901
        """Post-initialization processing."""
        # Helper for context. To be consolidated somewhere
        self._original_tokenizer = self.deep_copy()
        # Add any missing added tokens to the vocabulary.
        # Sort to fill up the vocabulary in order.
        for token in sorted(self.added_tokens.root, key=lambda x: x.id):
            # Get the content of the token
            content = token.content
            in_vocab = content in self.model.vocab.vocabulary
            if not in_vocab:
                logger.warning(f"Adding missing added token '{content}' to the vocabulary.")
                self._add_token_to_vocabulary(content, is_added_token=True)
                in_vocab = True
            # Retro-actively set the ID.
            if in_vocab:
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
                self._turn_into_addedtoken(
                    unk_token, is_special=True, normalized=False, single_word=True, lstrip=False, rstrip=False
                )
        pad_token = self.pad_token
        if pad_token:
            if pad_token not in self.model.vocab:
                current_pad_token_id = self.pad_token_id
                assert current_pad_token_id is not None
                if current_pad_token_id > self.vocabulary_size:
                    logger.warning(
                        f"pad_token_id {current_pad_token_id} is greater than vocabulary size {self.vocabulary_size}."
                    )
                else:
                    current_index_token = self.sorted_vocabulary[current_pad_token_id]
                    logger.warning(
                        f"pad_token '{pad_token}' not found in vocabulary, but pad_token_id {current_pad_token_id} "
                        f"maps to existing token '{current_index_token}'."
                    )
                self._add_token_to_vocabulary(pad_token, is_added_token=True)
                logger.warning(
                    f"Adding pad_token '{pad_token}' to the vocabulary with id: {self.model.vocab[pad_token]}."
                )
            added_token = self.added_tokens.get_token(pad_token)
            if not added_token:
                logger.warning(f"Turning pad_token '{pad_token}' into an AddedToken.")
                self._turn_into_addedtoken(
                    pad_token, is_special=True, normalized=True, single_word=True, lstrip=True, rstrip=True
                )
            self.pad_token = pad_token

    @property
    def preprocessor(self) -> Preprocessor:
        """Return a cached preprocessor."""
        # To avoid circular import
        from skeletoken.preprocessor import Preprocessor

        if self._preprocessor is None:
            self._preprocessor = Preprocessor.from_tokenizer_model(self)
        return self._preprocessor

    def add_addedtoken(
        self,
        token: str,
        is_special: bool = False,
        normalized: bool = False,
        single_word: bool = True,
        lstrip: bool = True,
        rstrip: bool = True,
    ) -> TokenizerModel:
        """Add an added token to the tokenizer model."""
        return self.add_addedtokens([token], is_special, normalized, single_word, lstrip, rstrip)

    def add_addedtokens(
        self,
        tokens: list[str],
        is_special: bool = False,
        normalized: bool = False,
        single_word: bool = True,
        lstrip: bool = True,
        rstrip: bool = True,
    ) -> TokenizerModel:
        """Add multiple added tokens to the tokenizer model."""
        model = self.deep_copy()
        for token in tokens:
            model._add_token_to_vocabulary(token, is_added_token=True)
            model._turn_into_addedtoken(
                token,
                is_special=is_special,
                normalized=normalized,
                single_word=single_word,
                lstrip=lstrip,
                rstrip=rstrip,
            )

        return model

    def _turn_into_addedtoken(
        self,
        token: str,
        is_special: bool = False,
        normalized: bool = False,
        single_word: bool = True,
        lstrip: bool = True,
        rstrip: bool = True,
    ) -> None:
        """Turn an existing token into an an added token and add it to added_tokens."""
        if token not in self.model.vocab.vocabulary:
            raise ValueError(
                f"Token '{token}' not found in the vocabulary. Please add it first using "
                "`add_token_to_vocabulary` or `add_addedtoken`."
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

    def add_tokens_to_vocabulary(self, tokens: list[str], preprocess_tokens: bool = True) -> TokenizerModel:
        """Add multiple tokens to the vocabulary.

        This can be much faster than calling `add_token_to_vocabulary` multiple times,
        because the copy only happens once.
        """
        model = self.deep_copy()
        if preprocess_tokens:
            tokens = [self._preprocess_token(token) for token in tokens]

        for token in tokens:
            model._add_token_to_vocabulary(token, is_added_token=False)

        return model

    def add_token_to_vocabulary(self, token: str, preprocess_token: bool = True) -> TokenizerModel:
        """Add a token to the tokenizer's vocabulary."""
        return self.add_tokens_to_vocabulary([token], preprocess_tokens=preprocess_token)

    def _preprocess_token(self, token: str) -> str:
        """Preprocesses a token."""
        token_list = self.preprocessor.preprocess(token)
        if len(token_list) > 1:
            tokens_formatted = [f" '{token}'" for token in token_list]
            token_list_string = f"{','.join(tokens_formatted)}".strip()
            raise ValueError(
                f"You tried to add the token '{token}' to the vocabulary, but during preprocessing, this got split into"
                f" the following tokens: ({token_list_string}). This token is not suitable for addition. Consider "
                "adding it as an added token."
            )
        if len(token_list) == 0:
            raise ValueError(f"The token {token} got removed during preprocessing.")
        new_token, *_ = token_list

        logger.info(f"Preprocessed '{token}' into '{new_token}'.")

        return new_token

    def _add_token_to_vocabulary(self, token: str, is_added_token: bool = False) -> None:
        """Add an added token to the vocabulary."""
        self.model.add_token(token, is_added_token=is_added_token)

    def replace_tokens_in_vocabulary(
        self, old_tokens: list[str], new_tokens: list[str], preprocess_tokens: bool = True
    ) -> TokenizerModel:
        """Replace a list of tokens with another list of tokens.

        The lists need to be matched: for each index in `old_tokens`, we expect a new token in `new_tokens`.
        The `preprocess_tokens` flag only applies to the new tokens, as the old tokens have been preprocessed already.

        This can be much faster than repeated calls to `replace_token_in_vocabulary`.
        """
        model = self.deep_copy()
        if len(old_tokens) != len(new_tokens):
            raise ValueError(
                "The lists of tokens need to be of the same length. "
                f"'old_tokens' is {len(old_tokens)} while 'new_tokens' is {len(new_tokens)}"
            )

        if preprocess_tokens:
            new_tokens = [self._preprocess_token(token) for token in new_tokens]

        # Strict = True is not really necessary.
        for old_token, new_token in zip(old_tokens, new_tokens, strict=True):
            model._replace_token_in_vocabulary(old_token, new_token)

        return model

    def replace_token_in_vocabulary(
        self, old_token: str, new_token: str, preprocess_token: bool = True
    ) -> TokenizerModel:
        """Replace a token with another one.

        The `preprocess_token` flag only applies to the new tokens, as the old tokens have been preprocessed already.
        """
        return self.replace_tokens_in_vocabulary([old_token], [new_token], preprocess_tokens=preprocess_token)

    def _replace_token_in_vocabulary(self, old_token: str, new_token: str, is_added_token: bool = False) -> None:
        """Replace a token with another one. It keeps the old index in the vocabulary."""
        self.model.replace_token(old_token, new_token, is_added_token=is_added_token)
        self.added_tokens.maybe_replace_token(old_token, new_token)
        if self.post_processor is not None:
            self.post_processor = maybe_replace_token_in_post_processor(
                old_token, new_token, self.model.vocab[new_token], self.post_processor
            )

    def _remap_added_token_ids(self) -> None:
        """Remap the IDs of added tokens to match the vocabulary."""
        remapping = self.model_delta.token_mapping
        for token in self.added_tokens.root:
            remapped = remapping.get(token.id, token.id)
            if token.id != remapped:
                logger.info(f"Remapping ID of added token '{token.content}' from {token.id} to {remapped}.")
                token.id = remapped
                token.content = self.sorted_vocabulary[remapped]
        self.pad_token = self.pad_token  # Trigger pad_token remapping
        if self.post_processor is not None:
            for added_token in self.added_tokens.root:
                self.post_processor = maybe_replace_token_in_post_processor(
                    added_token.content, added_token.content, added_token.id, self.post_processor
                )

    def remove_token_from_vocabulary(self, token: str) -> TokenizerModel:
        """Remove a token from the vocabulary."""
        return self.remove_tokens_from_vocabulary([token])

    def remove_tokens_from_vocabulary(self, tokens: list[str]) -> TokenizerModel:
        """Remove multiple tokens from the vocabulary.

        This is a convenience method that removes tokens from the vocabulary.
        Because removal requires compactifying the original vocabulary, this is more
        efficient than doing it in multiple passes.

        Parameters
        ----------
        tokens: list[str]
            The list of tokens to remove from the vocabulary.

        Returns
        -------
        TokenizerModel
            The tokenizer model with the tokens removed.

        """
        model = self.deep_copy()
        for token in tokens:
            model.added_tokens.maybe_remove_token(token)
        model.model.remove_tokens(tokens)

        return model

    def remove_uppercase(self) -> TokenizerModel:
        """Remove all uppercase tokens from the vocabulary.

        Returns
        -------
        TokenizerModel
            The tokenizer model with uppercase tokens removed.

        """
        model = self.deep_copy()
        return model._decase(lower=False)

    def decase_vocabulary(self) -> TokenizerModel:
        """Decases the vocabulary.

        Returns
        -------
        TokenizerModel
            The tokenizer model with a decased vocabulary.

        """
        model = self.deep_copy()
        return model._decase(lower=True)

    def _decase(self, lower: bool) -> TokenizerModel:
        """Private method to decase the vocabulary."""
        # Special tokens and unnormalized added tokens need to be skipped.
        sorted_vocab = self.model.vocab.sorted_vocabulary
        vocabulary = decase_vocabulary(
            sorted_vocab,
            self.added_tokens.root,
            is_byte=self.transforms_into_bytes,
            lower=lower,
        )
        mapping: dict[int, int] = {}
        for i, token in enumerate(vocabulary):
            if token is None:
                continue
            mapping[i] = len(mapping)
        self.model.replace_vocabulary(vocabulary)
        if not self.lowercases_input:
            self._add_normalizer_inplace(LowercaseNormalizer(), prefix=True)
        self._remap_added_token_ids()

        return self

    def add_pre_tokenizer(self, pre_tokenizer: PreTokenizerDiscriminator) -> TokenizerModel:
        """Add a pre-tokenizer to the tokenizer model."""
        model = self.deep_copy()
        model._add_pretokenizer_inplace(pre_tokenizer)
        return model

    def _add_pretokenizer_inplace(self, pre_tokenizer: PreTokenizerDiscriminator) -> None:
        """Add a pre-tokenizer to the tokenizer model in place."""
        self._preprocessor = None
        if self.pre_tokenizer is None:
            self.pre_tokenizer = pre_tokenizer
        elif isinstance(self.pre_tokenizer, PreTokenizerSequence):
            self.pre_tokenizer.pretokenizers.append(pre_tokenizer)
        else:
            self.pre_tokenizer = PreTokenizerSequence(pretokenizers=[self.pre_tokenizer, pre_tokenizer])

    def add_post_processor(self, post_processor: PostProcessorDiscriminator) -> TokenizerModel:
        """Add a post-processor to the tokenizer model."""
        model = self.deep_copy()
        model._add_post_processor_inplace(post_processor)
        model = model._add_tokens_from_post_processor()

        return model

    def _add_tokens_from_post_processor(self) -> TokenizerModel:
        """Adds tokens that are in the post processor as added tokens."""
        all_special_tokens_from_post_processor = [*(self.eos or []), *(self.bos or [])]
        pruned = [token for token in all_special_tokens_from_post_processor if token not in self.vocabulary]
        return self.add_addedtokens(pruned)

    def _add_post_processor_inplace(self, post_processor: PostProcessorDiscriminator) -> None:
        """Add a post-processor to the tokenizer model in place."""
        if self.post_processor is None:
            self.post_processor = post_processor
        elif isinstance(self.post_processor, PostProcessorSequence):
            self.post_processor.processors.append(post_processor)
        else:
            self.post_processor = PostProcessorSequence(processors=[self.post_processor, post_processor])

    def add_normalizer(self, normalizer: NormalizerDiscriminator, prefix: bool = False) -> TokenizerModel:
        """Add a normalizer to the tokenizer model.

        Parameters
        ----------
        normalizer: NormalizerDiscriminator
            The normalizer to add.
        prefix: bool
            Whether to add the normalizer before the other normalizers.
            This can be useful if, for example, one of your normalizers performs
            a destructive transform.

        Returns
        -------
        TokenizerModel
            The tokenizer model with the added normalizer.

        """
        model = self.deep_copy()
        model._add_normalizer_inplace(normalizer, prefix)
        return model

    def _add_normalizer_inplace(self, normalizer: NormalizerDiscriminator, prefix: bool = False) -> None:
        self._preprocessor = None
        if self.normalizer is None:
            self.normalizer = normalizer
        elif isinstance(self.normalizer, NormalizerSequence):
            self.normalizer.normalizers.append(normalizer)
        else:
            if prefix:
                self.normalizer = NormalizerSequence(normalizers=[normalizer, self.normalizer])
            else:
                self.normalizer = NormalizerSequence(normalizers=[self.normalizer, normalizer])

    @classmethod
    def from_tokenizer(cls: type[TokenizerModel], tokenizer: Tokenizer) -> TokenizerModel:
        """Create a TokenizerModel from a Tokenizer instance."""
        return cls.from_string(tokenizer.to_str())

    @classmethod
    def from_pretrained(cls: type[TokenizerModel], path_or_repo: PathLike) -> TokenizerModel:
        """Create a TokenizerModel from a pretrained tokenizer."""
        path = Path(path_or_repo)
        if path.exists() and path.is_dir():
            try:
                return cls.from_transformers(str(path))
            except Exception:
                pass
            if not (path / "tokenizer.json").exists():
                raise FileNotFoundError(
                    f"No tokenizer.json found in the directory: {path}. Please provide a valid path."
                )
            # If a tokenizer.json file exists, load it directly
            return cls.from_tokenizer(Tokenizer.from_file(str(path / "tokenizer.json")))
        elif path.exists() and path.is_file():
            return cls.from_tokenizer(Tokenizer.from_file(str(path)))

        return cls._load_remote(path_or_repo)  # pragma: nocover

    @classmethod
    def _load_remote(cls: type[TokenizerModel], path_or_repo: PathLike) -> TokenizerModel:  # pragma: nocover
        """Load a remote tokenizer from a HuggingFace repo."""
        try:
            return cls.from_transformers(str(path_or_repo))
        except Exception as e:
            logger.info(f"Tried to load tokenizer as a Hugging Face tokenizer, but failed: {e}")
            pass
        tokenizer = Tokenizer.from_pretrained(str(path_or_repo))

        return cls.from_tokenizer(tokenizer)

    @classmethod
    def from_string(cls: type[TokenizerModel], json_str: str) -> TokenizerModel:
        """Create a TokenizerModel from a JSON string."""
        return cls.model_validate_json(json_str)

    def to_tokenizer(self) -> Tokenizer:
        """Convert the TokenizerModel back to a Tokenizer instance."""
        return Tokenizer.from_str(self.model_dump_json())

    def make_model_greedy(self, max_input_chars_per_word: int = 100) -> TokenizerModel:
        """Convert the TokenizerModel to a greedy tokenizer model."""
        model = self.deep_copy()
        model.model = model.model.to_greedy()
        assert isinstance(model.model, WordPiece)
        model.model.max_input_chars_per_word = max_input_chars_per_word
        model = model.add_pre_tokenizer(FixedLengthPreTokenizer(length=max_input_chars_per_word))
        return model

    @property
    def lowercases_input(self) -> bool:
        """Check if the tokenizer lowercases the input."""
        return self.normalizer is not None and self.normalizer.lowercases

    @property
    def transforms_into_bytes(self) -> bool:
        """Check if the tokenizer transforms the input into bytes.

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
    def eos_ids(self) -> list[int] | None:
        """Get the end-of-sequence tokens."""
        if eos_tokens := self.eos:
            return self.tokens_to_ids(eos_tokens)
        return None

    @property
    def bos(self) -> list[str] | None:
        """Get the beginning-of-sequence tokens."""
        if self.post_processor is None:
            return None
        return get_bos_token_from_post_processor(self.post_processor)

    @property
    def bos_ids(self) -> list[int] | None:
        """Get the end-of-sequence tokens."""
        if bos_tokens := self.bos:
            return self.tokens_to_ids(bos_tokens)
        return None

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
        if old_unk_token == token:
            return
        if token is None:
            if isinstance(self.model, MODELS_THAT_NEED_UNK):
                raise ValueError("Cannot unset unk_token for WordPiece or WordLevel models.")
            if old_unk_token is not None:
                logger.info(f"Removing unk_token '{self.model.unk_token}' from the tokenizer.")
            self.model.unk_token = None
            return
        if old_unk_token is None:
            if token not in self.model.vocab:
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

        model = cls.from_tokenizer(hf_tokenizer.backend_tokenizer)
        if unk_token is not None and isinstance(unk_token, str):
            if model.unk_token is not None and model.unk_token != unk_token:
                logger.warning(
                    f"Overriding existing unk_token '{model.unk_token}' with the one from "
                    f"the HuggingFace tokenizer: '{unk_token}'."
                )
            if model.unk_token is None:
                logger.warning(
                    "HuggingFace tokenizer defines an unk_token, but the Skeletoken model does not. "
                    f"Setting it to '{unk_token}'."
                )
            model.unk_token = unk_token
        if pad_token is not None and isinstance(pad_token, str):
            if model.pad_token is not None and model.pad_token != pad_token:
                logger.warning(
                    f"Overriding existing pad_token '{model.pad_token}' "
                    f"with the one from the HuggingFace tokenizer: '{pad_token}'."
                )
            if model.pad_token is None:
                logger.warning(
                    "HuggingFace tokenizer defines a pad_token, but the Skeletoken model does not. "
                    f"Setting it to '{pad_token}'."
                )
            model.pad_token = pad_token

        model._original_class = type(hf_tokenizer)
        return model

    @classmethod
    def from_transformers(cls, path: PathLike) -> TokenizerModel:  # pragma: nocover
        """Load a HuggingFace tokenizer from a local path or a model repo."""
        hf_tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path)
        return cls.from_transformers_tokenizer(hf_tokenizer)

    def to_transformers(self, tokenizer_class: type[PreTrainedTokenizerFast] | None = None) -> PreTrainedTokenizerFast:
        """Convert the TokenizerModel to a HuggingFace tokenizer."""
        if tokenizer_class is None:
            if self._original_class is not None:
                tokenizer_class = self._original_class
            else:
                tokenizer_class = PreTrainedTokenizerFast
        tok = tokenizer_class(tokenizer_object=self.to_tokenizer())
        tok.pad_token = self.pad_token
        tok.unk_token = self.unk_token
        if self.bos:
            if len(self.bos) > 1:
                logger.warning(f"Tokenizer has multiple bos tokens: {self.bos}. Not setting it automatically.")
            else:
                tok.bos_token = self.bos[0]
        if self.eos:
            if len(self.eos) > 1:
                logger.warning(f"Tokenizer has multiple eos tokens: {self.eos}. Not setting it automatically.")
            else:
                tok.eos_token = self.eos[0]

        return tok

    @property
    def model_delta(self) -> ModelDelta:
        """Get the model delta between the original tokenizer and the current one."""
        from skeletoken.model_delta import compute_model_delta

        return compute_model_delta(self._original_tokenizer, self)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to their corresponding IDs."""
        return [self.model.vocab[token] for token in tokens]

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert a list of IDs to their corresponding tokens."""
        inv_vocab = self.model.vocab.sorted_vocabulary
        return [inv_vocab[id] for id in ids]

    @property
    def vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.model.vocab)

    @property
    def vocabulary(self) -> dict[str, int]:
        """Get the vocabulary as a dictionary."""
        return self.model.vocab.vocabulary

    @property
    def sorted_vocabulary(self) -> list[str]:
        """Get the sorted vocabulary as a list of tokens."""
        return self.model.vocab.sorted_vocabulary

    @property
    def unk_token_id(self) -> int | None:
        """Get the ID of the unk token, if any."""
        unk_token = self.unk_token
        if unk_token is None:
            return None
        return self.model.vocab[unk_token]

    @property
    def pad_token_id(self) -> int | None:
        """Get the ID of the pad token, if any."""
        if self.pad_token is None:
            return None
        assert self.padding is not None
        return self.padding.pad_id

    @property
    def adds_prefix_space(self) -> bool | None:
        """Return True if the tokenizer insertsa prefix space.

        This property returns a `bool` if and only if it has a BytePreTokenizer. In any other case,
        None will be returned.
        """
        if self.pre_tokenizer is None:
            return None
        return get_add_prefix_space(self.pre_tokenizer)

    @adds_prefix_space.setter
    def adds_prefix_space(self, value: bool) -> None:
        """Set the prefix space."""
        if self.pre_tokenizer is None:
            raise ValueError("You are trying to set the prefix space, but don't have a pretokenizer.")
        relevant_tokenizer = get_pretokenizer_of_type(self.pre_tokenizer, ByteLevelPreTokenizer)
        if relevant_tokenizer is None:
            raise ValueError(
                "You are trying to set the prefix space, but your pretokenizer is not a ByteLevelPreTokenizer."
            )
        relevant_tokenizer.add_prefix_space = value

    def prune_added_tokens(self) -> TokenizerModel:
        """Prune all added tokens that don't play a role in the model."""
        # Collect all added tokens that are useful.
        tokens_to_keep = set()
        if eos := self.eos:
            tokens_to_keep.update(eos)
        if bos := self.bos:
            tokens_to_keep.update(bos)
        if unk_token := self.unk_token:
            tokens_to_keep.add(unk_token)
        if pad_token := self.pad_token:
            tokens_to_keep.add(pad_token)

        added_tokens_to_remove = [
            token.content for token in self.added_tokens.root if token.content not in tokens_to_keep
        ]
        model = self.remove_tokens_from_vocabulary(added_tokens_to_remove)

        return model
